import os, json, time, torch
import numpy as np
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from tailorflow.model import build_vae
from tailorflow.prior import GlobalPrior, LocalPrior, GlobalFlow, LocalFlow, OccEncoder
from tailorflow.occ_aug import smooth_occ

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPS = os.path.join(REPO, 'checkpoints')


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def save_config(args, out_dir, extra=None):
    cfg = {**vars(args), **(extra or {}), '_saved_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    return cfg


def ddp_sync_check(params, batch, step):
    """Raises unless all ranks hold identical weights, different batches and the same step."""
    import torch.distributed as dist
    w = torch.cat([q.detach().double().flatten() for q in params])
    v = torch.stack([w.sum(), w.abs().sum(), batch.detach().double().sum(), torch.tensor(float(step), device=w.device, dtype=torch.float64)])
    g = [torch.zeros_like(v) for _ in range(dist.get_world_size())]
    dist.all_gather(g, v)
    g = torch.stack(g)
    dw = ((g[:, :2] - g[0, :2]).abs() / g[0, :2].abs().clamp(min=1e-12)).max().item()
    same_batch = (g[:, 2] == g[0, 2]).all().item()
    ok = dw < 1e-6 and not same_batch and (g[:, 3] == g[0, 3]).all().item()
    msg = f'ddp check step {step}: ranks {len(g)} weight rel diff {dw:.2e} batches differ {not same_batch} steps {g[:, 3].tolist()}'
    if not ok:
        raise RuntimeError('DDP DESYNC ' + msg)
    return msg


def freeze(*mods):
    for m in mods:
        m.eval()
        for q in m.parameters():
            q.requires_grad_(False)


def load_vae(path, dev):
    """EMA weights of a VAE checkpoint; its config.json sits next to it."""
    path = resolve(path)
    vc = json.load(open(os.path.join(os.path.dirname(path), 'config.json')))
    vae = build_vae(vc).to(dev)
    ck = torch.load(path, map_location=dev, weights_only=False)
    vae.load_state_dict(ck['ema'])
    freeze(vae)
    return vae, vc, ck


def load_prior(prior_exp, ckpt, vc, dev):
    pdir = os.path.join(EXPS, prior_exp)
    pc = json.load(open(f'{pdir}/config.json'))
    S = torch.load(f'{pdir}/stats.pt', map_location=dev, weights_only=False)
    pck = torch.load(f'{pdir}/{ckpt}', map_location=dev, weights_only=False)
    ncls = len(vc['cates'].split(','))
    mods = {'g': GlobalPrior(dg=pc['dg'], d=pc['gd'], depth=pc['gdepth'], ncls=ncls, docc=pc['dg']),
            'l': LocalPrior(dl=pc['dl'], dg=pc['dg'], d=pc['d'], ncls=ncls, vres=pc['vres'], vbound=pc['vbound'], vd=pc['vd'],
                            docc=pc['dg'], k_sa=pc['k_sa'], nfreq=pc['nfreq'],
                            hier_schedule=tuple(int(x) for x in pc['hier_schedule'].split(','))),
            'occ': OccEncoder(d=pc['dg'])}
    for k, m in mods.items():
        m.to(dev).load_state_dict({n[len(k) + 1:]: v for n, v in pck['ema'].items() if n.startswith(k + '.')})
        freeze(m)
    return {'g': GlobalFlow(mods['g'], ncls=ncls), 'l': LocalFlow(mods['l'], ncls=ncls), 'occ': mods['occ'], 'S': S, 'pc': pc,
            'step': pck['step']}


def decode_occ(occ_raw, use_occ):
    return torch.where(use_occ.view(-1, 1, 1), occ_raw.float(), torch.full_like(occ_raw.float(), -1.0))


@torch.no_grad()
def sample_local_partitioned(lf, B, L, dev, cls, zgl, steps, occ_cond=None, pocc_fn=None, chunk=4096, solver='euler'):
    """L > chunk: every velocity evaluation denoises m = round(L / chunk) random subsets of ~chunk points."""
    D, m = lf.p.dpt, max(1, round(L / chunk))
    c = -(-L // m)
    rep = lambda x: None if x is None else x.repeat_interleave(m, 0)
    zpt = torch.randn(B, L, D, device=dev)

    def vel(z, t):
        pocc = pocc_fn(z) if pocc_fn is not None else None
        perm = torch.argsort(torch.rand(B, L, device=dev), 1)
        perm = torch.cat([perm, perm[:, :m * c - L]], 1)
        g = lambda x: x.gather(1, perm.unsqueeze(-1).expand(-1, -1, x.shape[-1])).reshape(B * m, c, x.shape[-1])
        x = lf.p(g(z), rep(zgl), torch.full((B * m,), t, device=dev), rep(cls), occ_cond=rep(occ_cond),
                 pocc=None if pocc is None else g(pocc))
        xt = torch.empty(B, L, D, device=dev, dtype=x.dtype)
        xt.scatter_(1, perm[:, :L].unsqueeze(-1).expand(-1, -1, D), x.reshape(B, m * c, D)[:, :L])
        return (xt.float() - z) / (1 - t)

    for i in range(steps):
        v = vel(zpt, i / steps)
        if solver == 'heun' and i < steps - 1:
            v = (v + vel(zpt + v / steps, (i + 1) / steps)) / 2
        zpt = zpt + v / steps
    return zpt


@torch.no_grad()
def sample_prior(prior, vae, cl, bv, occ_raw, use_occ, steps=64, cfg=1.0, L=4096, chunk=4096, solver='euler', zg=None):
    """-> raw latents (zg, [B, L, 3+dl]); coverage conditions the rows with use_occ; a given zg is reused (same garment)."""
    B, dev = cl.shape[0], cl.device
    S, pc = prior['S'], prior['pc']
    oc_t = None
    if occ_raw is not None and bool(use_occ.any()):
        oc_t = smooth_occ(occ_raw.float(), pc['occ_smooth'])
    bvn = vae.norm_pts(bv.float())

    def pocc_fn(z):
        po = vae.latent_occ((z * S['psd'] + S['pm'])[..., :3].float(), oc_t, bvn)
        return torch.where(use_occ.view(-1, 1, 1), po, torch.zeros_like(po))

    pf = pocc_fn if oc_t is not None else None
    with torch.autocast('cuda', dtype=torch.bfloat16):
        oc = prior['occ'](oc_t, drop=~use_occ)
        sgl = ((zg - S['gm']) / S['gsd']) if zg is not None else prior['g'].sample(B, dev, cl, steps=steps, cfg=cfg,
                                                                                   occ_cond=oc, solver=solver)
        if L > chunk:
            assert cfg == 1.0, 'partitioned sampling supports cfg 1.0 only'
            spt = sample_local_partitioned(prior['l'], B, L, dev, cl, sgl, steps, occ_cond=oc, pocc_fn=pf, chunk=chunk, solver=solver)
        else:
            spt = prior['l'].sample(B, L, dev, cl, sgl, steps=steps, cfg=cfg, occ_cond=oc, pocc_fn=pf, solver=solver)
    return sgl.float() * S['gsd'] + S['gm'], spt.float() * S['psd'] + S['pm']


def inside_pct(q, bv, bn):
    """% of points behind the nearest body vertex's normal."""
    idx = knn_points(q.float(), bv.float(), K=1).idx.expand(-1, -1, 3)
    nm = F.normalize(bn.float().gather(1, idx), dim=-1)
    return 100 * (((q.float() - bv.float().gather(1, idx)) * nm).sum(-1) < 0).float().mean().item()


_GEO = {}


def body_geodesic(vae, src):
    """[V] geodesic distance (m) on the canonical body from the vertex set src."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    if 'g' not in _GEO:
        cb = vae.denorm_pts(vae.canon(1)[0][0].float()).cpu().numpy()
        f = vae.smpl_faces.cpu().numpy()
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        w = np.linalg.norm(cb[e[:, 0]] - cb[e[:, 1]], axis=1)
        n = cb.shape[0]
        _GEO['g'] = sp.coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    d = dijkstra(_GEO['g'], indices=np.flatnonzero(src.cpu().numpy()), min_only=True)
    return torch.as_tensor(d, dtype=torch.float32, device=src.device)


def pca_normals(p, k=16):
    """[N, 3] unit normals of a cloud by local PCA."""
    nb = knn_points(p[None], p[None], K=k).idx[0]
    q = p[nb] - p[nb].mean(1, keepdim=True)
    return torch.linalg.eigh(q.transpose(1, 2) @ q).eigenvectors[..., 0]


@torch.no_grad()
def guided_local(prior, vae, cl, bv, occ, zg, old=None, hold=None, L=None, init=None, t0=0.0, known=None, occ_w=1.0, steps=64,
                 tau=0.03, seed=0, chunk=4096):
    """Local-prior sample under coverage occ, zg pinned, raw latents. known: rows held clamped (only the new rows are
    returned); init/t0: SDEdit from t0; old/hold: project predicted points near held rows of old onto their tangent planes;
    occ_w: guidance on the coverage condition."""
    S, lf = prior['S'], prior['l']
    dev, D, sd = bv.device, lf.p.dpt, float(vae.data_sd)
    oc_t = smooth_occ(occ.float(), prior['pc']['occ_smooth'])
    bvn = vae.norm_pts(bv.float())
    zgl = (zg.float() - S['gm']) / S['gsd']
    K = 0 if known is None else known.shape[1]
    N = K + (L if init is None else init.shape[1])
    m = max(1, round(N / chunk))
    c = -(-N // m)
    g = torch.Generator(device=dev).manual_seed(seed)
    rep = lambda x: x.repeat_interleave(m, 0)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        oc = prior['occ'](oc_t, drop=torch.tensor([False], device=dev))
        ocu = prior['occ'](oc_t, drop=torch.tensor([True], device=dev))
    kx = None if known is None else (known.float() - S['pm']) / S['psd']
    kmask = (torch.arange(N, device=dev) < K).view(1, -1, 1).float()
    if old is not None:
        P = old[0, :, :3].float()
        Pn = pca_normals(P)

    def pin(z):
        return z if kx is None else torch.cat([kx, z[:, K:]], 1)

    def guide(xw):
        if old is None:
            return xw
        x = xw[0] * S['psd'] + S['pm']
        r = knn_points(x[None, :, :3].float(), P[None], K=1)
        j = r.idx[0, :, 0]
        hit = hold[j] & (r.dists[0, :, 0].sqrt() * sd < tau)
        q = x[:, :3] - ((x[:, :3] - P[j]) * Pn[j]).sum(-1, keepdim=True) * Pn[j]
        x = torch.cat([torch.where(hit[:, None], q, x[:, :3]), x[:, 3:]], -1)
        return ((x - S['pm']) / S['psd'])[None]

    def vel(z, t):
        pocc = vae.latent_occ((z * S['psd'] + S['pm'])[..., :3].float(), oc_t, bvn)
        perm = torch.argsort(torch.rand(1, N, device=dev, generator=g), 1)
        perm = torch.cat([perm, perm[:, :m * c - N]], 1)
        gg = lambda x: x.gather(1, perm.unsqueeze(-1).expand(-1, -1, x.shape[-1])).reshape(m, c, x.shape[-1])
        kn = gg(kmask)[..., 0] > 0.5 if K else None
        tt = torch.full((m,), t, device=dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            x = lf.p(gg(z), rep(zgl), tt, rep(cl), occ_cond=rep(oc), pocc=gg(pocc), known=kn).float()
            if occ_w != 1.0:
                xu = lf.p(gg(z), rep(zgl), tt, rep(cl), occ_cond=rep(ocu), pocc=torch.zeros_like(gg(pocc)), known=kn).float()
                x = xu + occ_w * (x - xu)
        xt = torch.empty(1, N, D, device=dev)
        xt.scatter_(1, perm[:, :N].unsqueeze(-1).expand(-1, -1, D), x.reshape(1, m * c, D)[:, :N])
        return (guide(xt) - z) / (1 - t)

    noise = torch.randn(1, N, D, device=dev, generator=g)
    z = pin(noise if init is None else (1 - t0) * noise + t0 * torch.cat([torch.zeros(1, K, D, device=dev),
                                                                            (init.float() - S['pm']) / S['psd']], 1))
    ts = [t0 + (1 - t0) * i / steps for i in range(steps + 1)]
    for i in range(steps):
        h = ts[i + 1] - ts[i]
        v = vel(z, ts[i])
        if i < steps - 1:
            v = (v + vel(pin(z + h * v), ts[i + 1])) / 2
        z = pin(z + h * v)
    return (z * S['psd'] + S['pm'])[:, K:]


@torch.no_grad()
def edit_latent(prior, vae, cl, zg, lat, bv, bn, beta, occ_old, occ_new, band=0.08, t0=0.75, occ_w=1.0, refine_steps=32,
                steps=64, seed=0):
    """Coverage edit (grow, trim or both) of one garment; occ_* [1, V, 1], raw latents -> (latent, info).
    Rows > band (geodesic m) from every coverage change are held bit-exact; the rest is inpainted under occ_new (count matched
    to the held density), refined by SDEdit from t0 steered onto the held cloth, and new rows the keep head puts outside
    occ_new are dropped. occ_w > 1: guidance on the coverage condition."""
    o, n = occ_old.view(-1) > 0.5, occ_new.view(-1) > 0.5
    ch = o ^ n
    if not bool(ch.any()):
        return lat, {'held': lat.shape[1], 'new': 0}
    P = lat[0, :, :3].float()
    vi = knn_points(P[None], vae.canon(1)[0][0].float()[None], K=1).idx[0, :, 0]
    near = body_geodesic(vae, ch) <= band
    hold = (n[vi] | ~o[vi]) & ~near[vi]
    held = lat[:, hold]
    nh = int(hold.sum())
    occ = n.float().view(1, -1, 1)
    sp = knn_points(P[None], P[None], K=2).dists[0, :, 1].sqrt().median()
    kw = dict(occ_w=occ_w)

    def on_held(x, thr):
        r = knn_points(x[:, :, :3].float(), P[None], K=1)
        return hold[r.idx[0, :, 0]] & (r.dists[0, :, 0].sqrt() < thr)

    if nh:
        pre = guided_local(prior, vae, cl, bv, occ, zg, lat, hold, L=4096, steps=32, seed=seed + 11, **kw)
        g = max(int(round(nh / max(float(on_held(pre, 0.01 / float(vae.data_sd)).float().mean()), 0.02))) - nh, 0)
        g = max(g, int(0.5 * nh / max(int((n & ~near).sum()), 1) * int((n & near).sum())))
    else:
        g = lat.shape[1]
    if g:
        gen = guided_local(prior, vae, cl, bv, occ, zg, L=g, known=held, steps=steps, seed=seed + 3, **kw)
        ref = guided_local(prior, vae, cl, bv, occ, zg, lat, hold, init=torch.cat([held.float(), gen], 1), t0=t0,
                           steps=refine_steps, seed=seed + 5, **kw) if nh else gen
        new = ref[:, ~on_held(ref, 2 * sp)].to(lat.dtype)
    else:
        new = lat[:, :0]
    out = torch.cat([held, new], 1)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        keep = vae.trim_keep(zg, out[..., :3].contiguous(), out[..., 3:].contiguous(), bv, bn, occ, beta)[0]
    keep[:nh] = True
    return out[:, keep], {'held': nh, 'new': int(keep[nh:].sum()), 'trimmed': int((~keep).sum())}
