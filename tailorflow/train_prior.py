import sys, os, time, math, argparse, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

from tailorflow.metrics import one_nna, fps
from tailorflow.data_io import build_mesh_loader, MeshClothDataset
from tailorflow.occ_aug import CutDataset, smooth_occ
from tailorflow.prior import GlobalPrior, LocalPrior, GlobalFlow, LocalFlow, OccEncoder, FracHead
from tailorflow.common import EXPS, resolve, save_config, load_vae, sample_prior, inside_pct, ddp_sync_check
from tailorflow.vis import cloud_panel, save_grid

p = argparse.ArgumentParser()
p.add_argument('--exp', required=True)
p.add_argument('--vae', required=True)
p.add_argument('--d', type=int, default=512)
p.add_argument('--gd', type=int, default=256)
p.add_argument('--gdepth', type=int, default=4)
p.add_argument('--vres', type=int, default=32)
p.add_argument('--vbound', type=float, default=4.0)
p.add_argument('--vd', type=int, default=64)
p.add_argument('--k_sa', type=int, default=16)
p.add_argument('--nfreq', type=int, default=6)
p.add_argument('--bs', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--warmup', type=int, default=200)
p.add_argument('--steps', type=int, default=40000)
p.add_argument('--grad_clip', type=float, default=1.0)
p.add_argument('--ema_decay', type=float, default=0.999)
p.add_argument('--amp', type=int, default=1)
p.add_argument('--flow_denom_min', type=float, default=0.05)
p.add_argument('--cfg_drop', type=float, default=0.1)
p.add_argument('--occ_smooth', type=int, default=3, help='SMPL-ring smoothing of the coverage mask the prior sees')
p.add_argument('--occ_cond_drop', type=float, default=0.25)
p.add_argument('--limit', type=int, default=0)
p.add_argument('--num_workers', type=int, default=6)
p.add_argument('--save_every', type=int, default=1000)
p.add_argument('--val', type=int, default=1, help='0: checkpoints only, no validation / renders')
p.add_argument('--ckpt_keep_after', type=int, default=10000)
p.add_argument('--ckpt_keep_every', type=int, default=1000)
p.add_argument('--cfg', type=float, default=1.0, help='CFG scale for val/render sampling')
p.add_argument('--val_n_per_class', type=int, default=64)
p.add_argument('--val_steps', type=int, default=15)
p.add_argument('--inpaint_p', type=float, default=0.0, help='probability of an inpainting row (EditAug cut: kept rows clean, cut rows generated)')
p.add_argument('--frac_w', type=float, default=1.0, help='FracHead BCE weight')
p.add_argument('--init', default='', help='prior exp to fine-tune from (EMA weights of --init_ckpt; its stats.pt is copied)')
p.add_argument('--init_ckpt', default='ckpt_latest.pt')
p.add_argument('--seed', type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed)
ws, rank, local_rank = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0)), int(os.environ.get('LOCAL_RANK', 0))
distributed = ws > 1
if distributed:
    dist.init_process_group('nccl')
    torch.cuda.set_device(local_rank)
main = rank == 0
dev = torch.device(f'cuda:{local_rank}' if distributed else 'cuda')
amp_dtype = torch.bfloat16 if args.amp else None
out = os.path.join(EXPS, args.exp)
if main:
    os.makedirs(f'{out}/renders', exist_ok=True)

vae, vc, vck = load_vae(args.vae, dev)
cates = vc['cates'].split(',')
ncls, L, dl, dg = len(cates), vc['L'], vc['dl'], vc['dg']
loader, sampler = build_mesh_loader(cates, 'train', args.bs, limit=args.limit, distributed=distributed, world_size=ws, rank=rank,
                                    num_workers=args.num_workers, wrap=(lambda d: CutDataset(d, args.inpaint_p)) if args.inpaint_p > 0 else None)
if main:
    print(f"loaded VAE {args.vae} step {vck['step']}", flush=True)
    save_config(args, out, extra={'world_size': ws, 'vae_exp': vc.get('exp'), 'vae_step': int(vck['step']), 'L': L, 'dl': dl, 'dg': dg,
                                  'cates': vc['cates'], 'hier_schedule': vc['hier_schedule']})

stat = f'{out}/stats.pt'
if main and args.init and not os.path.exists(stat):
    shutil.copy(os.path.join(EXPS, args.init, 'stats.pt'), stat)
key = {'vae': os.path.abspath(resolve(args.vae)), 'vae_step': int(vck['step']), 'dl': dl, 'dg': dg, 'lat': 'abs_pt+feat/blockstd',
       'lat_sample': 1, 'data_source': 'c3d', 'pt_whiten': 'axis'}
if main and os.path.exists(stat) and {k: torch.load(stat, map_location='cpu', weights_only=False).get(k) for k in key} != key:
    print('STATS CACHE STALE -- recomputing', flush=True)
    os.remove(stat)
if main and not os.path.exists(stat):
    ps, gs = [], []
    stat_loader, _ = build_mesh_loader(cates, 'train', args.bs, limit=args.limit, num_workers=args.num_workers)
    with torch.no_grad(), torch.autocast('cuda', dtype=amp_dtype, enabled=bool(args.amp)):
        for b in stat_loader:
            zg, pt, ft = vae.encode(b['tr_points'].to(dev), b['body_verts'].to(dev), sample=True)
            ps.append(torch.cat([pt.float(), ft.float()], -1).reshape(-1, 3 + dl).cpu())
            gs.append(zg.float().cpu())
    ps, gs = torch.cat(ps), torch.cat(gs)
    psd = torch.cat([ps[:, :3].std(0)] + ([ps[:, 3:].reshape(-1).std().view(1).expand(dl)] if dl else [])) + 1e-6
    gsd = (gs.reshape(-1).std() + 1e-6).expand(gs.shape[1]).clone()
    torch.save({'pm': ps.mean(0), 'psd': psd, 'gm': gs.mean(0), 'gsd': gsd, **key}, stat)
    print(f'whitening: pt std {psd[:3].tolist()} feat {psd[3:].tolist()} global {gsd[0]:.3f}', flush=True)
if distributed:
    dist.barrier()
S = torch.load(stat, map_location=dev, weights_only=False)

hier_schedule = tuple(int(x) for x in vc['hier_schedule'].split(','))
net = nn.ModuleDict({'g': GlobalPrior(dg=dg, d=args.gd, depth=args.gdepth, ncls=ncls, docc=dg),
                     'l': LocalPrior(dl=dl, dg=dg, d=args.d, ncls=ncls, vres=args.vres, vbound=args.vbound, vd=args.vd, docc=dg,
                                     k_sa=args.k_sa, nfreq=args.nfreq, hier_schedule=hier_schedule),
                     'occ': OccEncoder(d=dg), 'frac': FracHead(dg=dg, ncls=ncls)}).to(dev)
raw = net
global_flow = GlobalFlow(raw['g'], drop=args.cfg_drop, ncls=ncls)
local_flow = LocalFlow(raw['l'], drop=args.cfg_drop, ncls=ncls, t_hi_p=0.1, denom_min=args.flow_denom_min)


class FlowLoss(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, zgl, zgl_cond, zpt, cls, occ_in, occ_drop, bv, known, inp, occ_old, frac_t):
        occ_cond = self.net['occ'](occ_in, drop=occ_drop)
        loss_g = global_flow.loss(zgl, cls, occ_cond=occ_cond)
        loss_l = local_flow.loss(zpt, zgl_cond, cls, occ_cond=occ_cond, pocc_fn=pocc_fn(occ_in, bv), pocc_drop=occ_drop, known=known)
        fl = nn.functional.binary_cross_entropy_with_logits(self.net['frac'](occ_old, occ_in, zgl_cond, cls), frac_t, reduction='none')
        loss_f = (fl * inp).sum() / inp.sum().clamp(min=1)
        return loss_g, loss_l, loss_f


train_step = FlowLoss(net)
if distributed:
    train_step = DDP(train_step, device_ids=[local_rank], find_unused_parameters=False)
torch.manual_seed(args.seed + 1000 * rank)
torch.cuda.manual_seed_all(args.seed + 1000 * rank)
prior = {'g': global_flow, 'l': local_flow, 'occ': raw['occ'], 'S': S, 'pc': {'occ_smooth': args.occ_smooth}}
opt = Adam(net.parameters(), lr=args.lr)
if main:
    print('params(M): global %.2f local %.2f occ %.2f' % tuple(sum(q.numel() for q in raw[k].parameters()) / 1e6 for k in ('g', 'l', 'occ')),
          flush=True)
if args.init and not os.path.exists(f'{out}/ckpt_latest.pt'):
    ick = torch.load(os.path.join(EXPS, args.init, args.init_ckpt), map_location=dev, weights_only=False)
    raw.load_state_dict(ick['ema'])
    if main:
        print(f"init from {args.init}/{args.init_ckpt} (step {ick['step']}, EMA)", flush=True)
ema = {k: v.detach().clone() for k, v in raw.state_dict().items()}


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    f = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * min(f, 1.0)))


_ema_pairs = None


@torch.no_grad()
def ema_update():
    global _ema_pairs
    if _ema_pairs is None:
        sd = raw.state_dict()
        fl = [k for k, v in sd.items() if v.dtype.is_floating_point]
        _ema_pairs = ([ema[k] for k in fl], [sd[k] for k in fl], [(ema[k], sd[k]) for k, v in sd.items() if not v.dtype.is_floating_point])
    e, v, other = _ema_pairs
    torch._foreach_lerp_(e, v, 1 - args.ema_decay)
    for a, b in other:
        a.copy_(b)


def pocc_fn(occ, bv):
    bvn = vae.norm_pts(bv.float())
    return lambda z: vae.latent_occ((z * S['psd'] + S['pm'])[..., :3].float(), occ, bvn)


def decode(zg, lat, bv, bn, occ, beta):
    with torch.autocast('cuda', dtype=amp_dtype, enabled=bool(args.amp)):
        return vae.decode(zg, lat[..., :3].contiguous(), lat[..., 3:].contiguous(), bv, bn, occ, beta=beta).float()


step = 0
cpath = f'{out}/ckpt_latest.pt'
if os.path.exists(cpath):
    c = torch.load(cpath, map_location=dev, weights_only=False)
    raw.load_state_dict(c['model'])
    opt.load_state_dict(c['opt'])
    step = c['step']
    ema = {k: v.to(dev) for k, v in c['ema'].items()}
    if main:
        print('resumed', step, flush=True)


@torch.no_grad()
def inpaint_rows(b, bv, zgl, zpt, occ_drop):
    """Items with a loader-side cut: known = latent rows on kept faces, zg of the cut garment as local conditioning."""
    B = zgl.shape[0]
    known = torch.zeros(B, L, dtype=torch.bool, device=dev)
    inp = torch.zeros(B, dtype=torch.bool, device=dev)
    occ_old = b['body_occupancy'].to(dev)[..., 0].clone()
    zgl_cond = zgl.clone()
    if 'cut' not in b:
        return zgl_cond, known, inp, occ_old, torch.zeros(B, device=dev)
    keep = b['cut_keep'].to(dev)
    nu = (~keep).sum(1)
    rows = (b['cut'].to(dev) & ~occ_drop & (nu >= 16) & (nu <= L - 64)).nonzero(as_tuple=True)[0]
    if len(rows):
        crops = fps(b['cut_dense'].to(dev)[rows], L, random_start=True)[0]
        with torch.autocast('cuda', dtype=amp_dtype, enabled=bool(args.amp)):
            zg_c = vae.encode(crops, bv[rows], sample=True)[0]
        known[rows] = keep[rows]
        zgl_cond[rows] = (zg_c.float() - S['gm']) / S['gsd']
        inp[rows] = True
        occ_old[rows] = b['cut_m'].to(dev)[rows]
    frac_t = (~known).float().mean(1)
    return zgl_cond, known, inp, occ_old, frac_t


val_ds = MeshClothDataset(cates, 'val', limit=args.limit)
first = {}
for i, (c, _, _) in enumerate(val_ds.items):
    first.setdefault(c, i)
fb = {k: torch.stack([val_ds[first[c]][k] for c in cates if c in first]).to(dev)
      for k in ('tr_points', 'body_verts', 'body_normals', 'body_occupancy', 'beta', 'cate_idx')}


@torch.no_grad()
def run_validation(bs=16):
    """Per-class 1-NNA-CD and inside% of coverage-free samples."""
    raw.eval()
    res = {}
    for ci, c in enumerate(cates):
        idx = [i for i, it in enumerate(val_ds.items) if it[0] == c][:args.val_n_per_class]
        gens, refs, ins = [], [], []
        for s in range(0, len(idx), bs):
            b = torch.utils.data.default_collate([val_ds[i] for i in idx[s:s + bs]])
            bv, bn = b['body_verts'].to(dev), b['body_normals'].to(dev)
            cl = torch.full((bv.shape[0],), ci, dtype=torch.long, device=dev)
            zg, lat = sample_prior(prior, vae, cl, bv, None, torch.zeros_like(cl, dtype=torch.bool), steps=args.val_steps, cfg=args.cfg, L=L)
            y = decode(zg, lat, bv, bn, None, b['beta'].to(dev))
            gens.append(y)
            refs.append(b['tr_points'].to(dev))
            ins.append(inside_pct(y, bv, bn))
        res[c] = (one_nna(torch.cat(gens), torch.cat(refs), 32) * 100, float(np.mean(ins)))
    raw.train()
    return res


@torch.no_grad()
def save_render(step):
    n = fb['tr_points'].shape[0]
    cl = fb['cate_idx'].view(-1).long()
    bv, bn, beta, occ = fb['body_verts'], fb['body_normals'], fb['beta'], fb['body_occupancy']
    roll = torch.roll(torch.arange(n, device=dev), 1)
    rec = vae.decode(*vae.encode(fb['tr_points'], bv, sample=True), bv, bn, occ, beta=beta)
    zg0, lat0 = sample_prior(prior, vae, cl, bv, occ, torch.zeros(n, dtype=torch.bool, device=dev), cfg=args.cfg, L=L)
    zg1, lat1 = sample_prior(prior, vae, cl, bv, occ, torch.ones(n, dtype=torch.bool, device=dev), cfg=args.cfg, L=L)
    panels = [('gt', fb['tr_points'], bv, None), ('vaeRecon_occ', rec, bv, occ), ('prior_noOcc', decode(zg0, lat0, bv, bn, None, beta), bv, None),
              ('prior_occ', decode(zg1, lat1, bv, bn, occ, beta), bv, occ),
              ('prior_occ_otherBody', decode(zg1, lat1, bv[roll], bn[roll], occ, beta[roll]), bv[roll], occ)]
    rows = [np.concatenate([cloud_panel(pc[i], bd[i], f'{cates[cl[i]]}_{nm}', occ=None if oc is None else oc[i]) for nm, pc, bd, oc in panels], axis=1)
            for i in range(n)]
    save_grid(rows, f'{out}/renders/step_{step:06d}.png')


progress_path = f'{out}/progress.csv'
if main and not os.path.exists(progress_path):
    with open(progress_path, 'w') as f:
        f.write('step,loss,' + ','.join(f'{c}_1nna,{c}_inside' for c in cates) + ',1nna,inside\n')

ep = 0
t0 = time.time()
step0 = step
while step < args.steps:
    if sampler is not None:
        sampler.set_epoch(ep)
    for b in loader:
        x_in = b['tr_points'].to(dev)
        bv = b['body_verts'].to(dev)
        occ_in = smooth_occ(b['body_occupancy'].to(dev), args.occ_smooth)
        with torch.no_grad(), torch.autocast('cuda', dtype=amp_dtype, enabled=bool(args.amp)):
            zg, pt, ft = vae.encode(x_in, bv, sample=True)
            zpt = (torch.cat([pt.float(), ft.float()], -1) - S['pm']) / S['psd']
            zgl = (zg.float() - S['gm']) / S['gsd']
        cls = b['cate_idx'].to(dev).view(-1).long()
        occ_drop = torch.rand(cls.shape[0], device=dev) < args.occ_cond_drop
        if 'cut' in b:
            occ_drop = occ_drop & ~b['cut'].to(dev)
        zgl_cond, known, inp, occ_old, frac_t = inpaint_rows(b, bv, zgl, zpt, occ_drop)
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        opt.zero_grad()
        with torch.autocast('cuda', dtype=amp_dtype, enabled=bool(args.amp)):
            loss_g, loss_l, loss_f = train_step(zgl, zgl_cond, zpt, cls, occ_in, occ_drop, bv, known, inp.float(), occ_old, frac_t)
            loss = loss_g + loss_l + args.frac_w * loss_f
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        opt.step()
        ema_update()
        if distributed and (step in (step0, step0 + 1, step0 + 100) or step % args.save_every == 0):
            msg = ddp_sync_check(net.parameters(), x_in, step)
            if main and step <= step0 + 100:
                print('  ' + msg, flush=True)
        if main and step % 50 == 0:
            pp = local_flow.last_parts
            print(f'step {step} loss {loss.item():.4f} gl {loss_g.item():.4f} [pos {pp[0]:.4f} feat {pp[1]:.4f}] frac {loss_f.item():.4f} '
                  f'inp {int(inp.sum())} ({time.time() - t0:.1f}s)', flush=True)
        if main and step % args.save_every == 0:
            backup = {k: v.detach().clone() for k, v in raw.state_dict().items()}
            raw.load_state_dict(ema)
        if main and step % args.save_every == 0 and args.val:
            save_render(step)
            res = run_validation()
            nna, ins = float(np.mean([v[0] for v in res.values()])), float(np.mean([v[1] for v in res.values()]))
            print(f'  [val step {step}] 1-NNA-CD {nna:.1f} inside {ins:.2f}% | ' + ' '.join(f'{c} {v[0]:.1f}/{v[1]:.2f}' for c, v in res.items()),
                  flush=True)
            with open(progress_path, 'a') as f:
                f.write(f'{step},{loss.item():.4f},' + ','.join(f'{v[0]:.2f},{v[1]:.2f}' for v in res.values()) + f',{nna:.2f},{ins:.2f}\n')
        if main and step % args.save_every == 0:
            ck = {'model': backup, 'ema': ema, 'opt': opt.state_dict(), 'step': step}
            torch.save(ck, cpath)
            if step >= args.ckpt_keep_after and step % args.ckpt_keep_every == 0:
                torch.save(ck, f'{out}/ckpt_{step:06d}.pt')
            raw.load_state_dict(backup)
        step += 1
        if step >= args.steps:
            break
    ep += 1
if main:
    torch.save({'model': raw.state_dict(), 'ema': ema, 'opt': opt.state_dict(), 'step': step}, f'{out}/ckpt_{step:06d}.pt')
    print(f'saved final ckpt_{step:06d}.pt', flush=True)
