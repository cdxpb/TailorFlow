import numpy as np, torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree, breadth_first_order
from pytorch3d.ops import knn_points, estimate_pointcloud_normals
from pytorch3d.ops.marching_cubes import marching_cubes


def nn_spacing(p):
    return knn_points(p[None], p[None], K=2).dists[0, :, 1].clamp_min(0).sqrt()


def oriented_normals(pts, body, k_pca=12, k_mst=8, sigma=0.02):
    nrm = estimate_pointcloud_normals(pts[None], neighborhood_size=k_pca, use_symeig_workaround=False)[0]
    N = pts.shape[0]
    bi = knn_points(pts[None], body[None], K=1)
    db = bi.dists[0, :, 0].clamp_min(0).sqrt()
    radial = (pts - body[bi.idx[0, :, 0]]) / db.clamp_min(1e-6)[:, None]
    w = torch.exp(-db / sigma) + 1e-6
    r = knn_points(pts[None], pts[None], K=k_mst + 1)
    ii = torch.arange(N, device=pts.device)[:, None].expand(N, k_mst).flatten()
    jj = r.idx[0, :, 1:].flatten()
    ew = (1.0 - (nrm[ii] * nrm[jj]).sum(-1).abs()) + 1e-4
    g = coo_matrix((ew.cpu().numpy(), (ii.cpu().numpy(), jj.cpu().numpy())), shape=(N, N))
    t = minimum_spanning_tree(g.maximum(g.T))
    t = t + t.T
    ncomp, lab = connected_components(t, directed=False)
    n = nrm.cpu().numpy().copy()
    wn, ru = w.cpu().numpy(), radial.cpu().numpy()
    for c in range(ncomp):
        members = np.flatnonzero(lab == c)
        order, pred = breadth_first_order(t, int(members[np.argmax(wn[members])]), directed=False)
        for v in order[1:]:
            if (n[v] * n[pred[v]]).sum() < 0:
                n[v] = -n[v]
        if (wn[members] * (n[members] * ru[members]).sum(1)).sum() < 0:
            n[members] = -n[members]
    return nrm, torch.from_numpy(n).to(pts.device)


def edge_keys(Fc, N):
    e = torch.stack([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]], 1).sort(-1).values
    return e[..., 0] * N + e[..., 1]


def face_components(Fc, N):
    ekf = edge_keys(Fc, N).flatten()
    fi = torch.arange(Fc.shape[0], device=Fc.device).repeat_interleave(3)
    o = torch.argsort(ekf)
    eks, fis = ekf[o], fi[o]
    pair = eks[1:] == eks[:-1]
    f1, f2 = fis[:-1][pair].cpu().numpy(), fis[1:][pair].cpu().numpy()
    n, lab = connected_components(coo_matrix((np.ones(len(f1)), (f1, f2)), shape=(Fc.shape[0],) * 2), directed=False)
    return torch.from_numpy(lab).to(Fc.device), n


def _keep_faces(V, Fc, mask):
    Fc = Fc[mask]
    used = torch.unique(Fc)
    remap = torch.full((V.shape[0],), -1, dtype=torch.long, device=V.device)
    remap[used] = torch.arange(used.shape[0], device=V.device)
    return V[used], remap[Fc]


def _edge_counts(Fc, N):
    e = torch.stack([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]], 1).sort(-1).values
    _, inv, cnt = torch.unique((e[..., 0] * N + e[..., 1]).reshape(-1), return_inverse=True, return_counts=True)
    return cnt[inv].view(-1, 3), e


@torch.no_grad()
def remove_ears(V, Fc, iters=10):
    for _ in range(iters):
        c, _ = _edge_counts(Fc, V.shape[0])
        ear = (c == 1).sum(1) >= 2
        if not ear.any():
            break
        Fc = Fc[~ear]
    return _keep_faces(V, Fc, torch.ones(Fc.shape[0], dtype=torch.bool, device=V.device))


@torch.no_grad()
def smooth_boundary(V, Fc, iters=50, lam=0.5, mu=-0.53):
    N = V.shape[0]
    c, e = _edge_counts(Fc, N)
    be = e[c == 1]
    if be.shape[0] == 0:
        return V
    fn = torch.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]], dim=-1)
    vn = torch.nn.functional.normalize(torch.zeros_like(V).index_add_(0, Fc.reshape(-1), fn.repeat_interleave(3, 0)), dim=-1)
    one = torch.ones(be.shape[0], device=V.device)
    deg = torch.zeros(N, device=V.device).index_add_(0, be[:, 0], one).index_add_(0, be[:, 1], one)
    ok = (deg == 2)[:, None]
    V = V.clone()
    for _ in range(iters):
        for w in (lam, mu):
            nsum = torch.zeros_like(V).index_add_(0, be[:, 0], V[be[:, 1]]).index_add_(0, be[:, 1], V[be[:, 0]])
            dlt = nsum / 2 - V
            dlt = dlt - (dlt * vn).sum(-1, keepdim=True) * vn
            V = torch.where(ok, V + w * dlt, V)
    return V


def boundary_loops(Fc, N):
    e = torch.cat([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]])
    k = e.sort(1).values
    _, inv, cnt = torch.unique(k[:, 0] * N + k[:, 1], return_inverse=True, return_counts=True)
    nxt, bad = {}, set()
    for u, v in e[cnt[inv] == 1].cpu().numpy().tolist():
        if u in nxt:
            bad.add(u)
        nxt[u] = v
    loops, seen = [], set()
    for start in list(nxt):
        if start in seen or start in bad:
            continue
        loop, u, ok = [start], nxt[start], True
        while u != start:
            if u in seen or u not in nxt or u in bad or len(loop) > len(nxt):
                ok = False
                break
            loop.append(u)
            u = nxt[u]
        seen.update(loop)
        if ok and len(loop) >= 3:
            loops.append(loop)
    return loops


@torch.no_grad()
def smooth_loops(V, Fc, sigma=0.015, min_perim_sigmas=8.0, band_rings=3, band_iters=10):
    N, dev = V.shape[0], V.device
    fn = torch.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]], dim=-1)
    vn = torch.nn.functional.normalize(torch.zeros_like(V).index_add_(0, Fc.reshape(-1), fn.repeat_interleave(3, 0)), dim=-1)
    Vn = V.clone()
    moved = torch.zeros(N, dtype=torch.bool, device=dev)
    for lp in boundary_loops(Fc, N):
        idx = torch.tensor(lp, device=dev)
        P = V[idx]
        perim = (P.roll(-1, 0) - P).norm(dim=-1).sum().item()
        if perim < min_perim_sigmas * sigma:
            continue
        L = P.shape[0]
        h = perim / L
        r = min(int(3 * sigma / h) + 1, (L - 1) // 2)
        off = torch.arange(-r, r + 1, device=dev)
        w = torch.exp(-0.5 * (off * h / sigma) ** 2)
        d = (P[(torch.arange(L, device=dev)[:, None] + off[None]) % L] * (w / w.sum())[None, :, None]).sum(1) - P
        n = vn[idx]
        Vn[idx] = P + d - (d * n).sum(-1, keepdim=True) * n
        moved[idx] = True
    if band_rings and moved.any():
        ae = torch.unique(torch.cat([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]]).sort(1).values, dim=0)
        band = moved.clone()
        for _ in range(band_rings):
            nb = band.clone()
            nb[ae[band[ae[:, 0]], 1]] = True
            nb[ae[band[ae[:, 1]], 0]] = True
            band = nb
        free_v = (band & ~moved)[:, None]
        one = torch.ones(ae.shape[0], device=dev)
        deg = torch.zeros(N, device=dev).index_add_(0, ae[:, 0], one).index_add_(0, ae[:, 1], one).clamp_min(1)[:, None]
        for _ in range(band_iters):
            s_ = torch.zeros_like(Vn).index_add_(0, ae[:, 0], Vn[ae[:, 1]]).index_add_(0, ae[:, 1], Vn[ae[:, 0]])
            Vn = torch.where(free_v, 0.5 * Vn + 0.5 * s_ / deg, Vn)
    return Vn


@torch.no_grad()
def mesh_garment_poisson(points, body_verts, h=None, pad=12, grad_sigma=3.0, iso_sigma=6.0, k_pca=48, trim=4.0,
                         rim_thr=0.5, rim_k=16, rim_d=1.0, rim_iters=20, comp_frac=0.05, min_faces=200,
                         ear_iters=10, bnd_iters=50, loop_sigma=0.025, interior_iters=30, interior_lam=0.5):
    pts, body = points.float(), body_verts.float()
    dev = pts.device
    s = nn_spacing(pts).mean().item()
    h = max(0.004, 0.8 * s) if h is None else h
    _, vn = oriented_normals(pts, body, k_pca=k_pca)
    lo = pts.min(0).values - pad * h
    dims = ((pts.max(0).values + pad * h - lo) / h).ceil().long() + 2
    dims = [int(x) + int(x) % 2 for x in dims]
    g = (pts - lo) / h
    i0 = g.floor().long()
    f = g - i0
    NV = dims[0] * dims[1] * dims[2]
    corners = []
    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                w = (f[:, 0] if ox else 1 - f[:, 0]) * (f[:, 1] if oy else 1 - f[:, 1]) * (f[:, 2] if oz else 1 - f[:, 2])
                corners.append(((i0[:, 0] + ox) * dims[1] * dims[2] + (i0[:, 1] + oy) * dims[2] + (i0[:, 2] + oz), w))

    def splat(val):
        out = torch.zeros(val.shape[1], NV, device=dev)
        for lin, w in corners:
            out.index_add_(1, lin, (val * w[:, None]).T.contiguous())
        return out.view(val.shape[1], *dims)

    K = [2 * np.pi * torch.fft.fftfreq(d, d=h, device=dev) for d in dims]
    KX, KY, KZ = torch.meshgrid(*K, indexing='ij')
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    Vh = torch.fft.fftn(splat(vn), dim=(1, 2, 3))
    chi_h = 1j * (KX * Vh[0] + KY * Vh[1] + KZ * Vh[2]) * torch.exp(-0.5 * (grad_sigma * h) ** 2 * K2)
    del Vh, KX, KY, KZ
    chi_h = -chi_h / torch.where(K2 == 0, torch.ones_like(K2), K2)
    chi_h[0, 0, 0] = 0
    chi = torch.fft.ifftn(chi_h).real.float()
    del chi_h
    cv = sum(chi.reshape(-1)[lin] * w for lin, w in corners)
    G = torch.exp(-0.5 * (iso_sigma * h) ** 2 * K2)
    blur = lambda grid, G=G: torch.fft.ifftn(torch.fft.fftn(grid) * G).real.float()
    den = blur(splat(torch.ones_like(cv)[:, None])[0])
    chi = chi - blur(splat(cv[:, None])[0]) / den.clamp_min(1e-4 * den.max())
    del den, G, K2
    Vm, Fm = marching_cubes(chi[None], 0.0, return_local_coords=False)
    del chi
    V, Fc = lo + Vm[0][:, [2, 1, 0]] * h, Fm[0][:, [0, 2, 1]]
    if Fc.shape[0] == 0:
        return V, Fc
    cen = V[Fc].mean(1)
    kn = knn_points(cen[None], pts[None], K=rim_k)
    d = kn.dists[0, :, 0].clamp_min(0).sqrt()
    keep = d < trim * s
    if not keep.any():
        return V[:0], Fc[:0]
    if rim_thr > 0:
        nb = pts[kn.idx[0]]
        mu = nb.mean(1)
        nrm = torch.nn.functional.normalize(vn[kn.idx[0]].mean(1), dim=-1)
        off = cen - mu
        off = off - (off * nrm).sum(-1, keepdim=True) * nrm
        spread = (nb - mu[:, None]).norm(dim=-1).mean(1).clamp_min(1e-9)
        cand = ((off.norm(dim=-1) / spread > rim_thr) & (d > rim_d * s))[keep]
        Fc = Fc[keep]
        for _ in range(rim_iters):
            e = torch.stack([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]], 1).sort(-1).values
            _, inv, cnt = torch.unique((e[..., 0] * V.shape[0] + e[..., 1]).reshape(-1), return_inverse=True, return_counts=True)
            rm = (cnt[inv].view(-1, 3) == 1).any(1) & cand
            if not rm.any():
                break
            Fc, cand = Fc[~rm], cand[~rm]
        keep = torch.ones(Fc.shape[0], dtype=torch.bool, device=dev)
    V, Fc = _keep_faces(V, Fc, keep)
    lab, nl = face_components(Fc, V.shape[0])
    csz = torch.bincount(lab, minlength=nl)
    V, Fc = _keep_faces(V, Fc, csz[lab] >= max(min_faces, comp_frac * csz.max().item()))
    if ear_iters:
        V, Fc = remove_ears(V, Fc, ear_iters)
    if bnd_iters:
        V = smooth_boundary(V, Fc, bnd_iters)
    if loop_sigma:
        V = smooth_loops(V, Fc, sigma=loop_sigma)
    if interior_iters:
        V = smooth_interior(V, Fc, interior_iters, interior_lam)
    return V, Fc


def smooth_interior(V, Fc, iters=30, lam=0.5):
    N = V.shape[0]
    he = torch.cat([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]])
    k = he.sort(1).values
    _, inv, cnt = torch.unique(k[:, 0] * N + k[:, 1], return_inverse=True, return_counts=True)
    bv = torch.zeros(N, dtype=torch.bool, device=V.device)
    bv[he[cnt[inv] == 1].flatten()] = True
    e = torch.cat([he, he.flip(1)])
    deg = torch.zeros(N, device=V.device).index_add_(0, e[:, 0], torch.ones(e.shape[0], device=V.device)).clamp_min(1)
    for _ in range(iters):
        avg = torch.zeros_like(V).index_add_(0, e[:, 0], V[e[:, 1]]) / deg[:, None]
        V = torch.where(bv[:, None], V, V + lam * (avg - V))
    return V


POISSON_PRESETS = {
    'b3x_lap30': dict(grad_sigma=3.0, iso_sigma=6.0, interior_iters=30),
    'b3x_lap15': dict(grad_sigma=3.0, iso_sigma=6.0, interior_iters=15),
}

@torch.no_grad()
def mesh_garment(points, body_verts, preset='b3x_lap30', **kw):
    return mesh_garment_poisson(points, body_verts, **{**POISSON_PRESETS[preset], **kw})


def weld(V, Fc, tol=1e-7):
    _, inv = torch.unique(torch.round(V / tol).long(), dim=0, return_inverse=True)
    Vn = torch.zeros(int(inv.max()) + 1, 3, device=V.device, dtype=V.dtype).index_copy_(0, inv, V)
    Fn = inv[Fc]
    return Vn, Fn[(Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])]


def area(V, Fc):
    return 0.5 * torch.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]], dim=-1).norm(dim=-1).sum().item()


def topo(V, Fc):
    V, Fc = weld(V, Fc)
    e = torch.cat([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]]).sort(1).values
    ue, cnt = torch.unique(e, dim=0, return_counts=True)
    b = ue[cnt == 1]
    nv = V.shape[0]
    bn = b.cpu().numpy()
    loops = 0
    if len(bn):
        _, lab = connected_components(coo_matrix((np.ones(len(bn)), (bn[:, 0], bn[:, 1])), shape=(nv, nv)), directed=False)
        loops = len(np.unique(lab[np.unique(bn)]))
    fe = Fc.cpu().numpy()
    _, lab = connected_components(coo_matrix((np.ones(len(fe) * 2), (np.r_[fe[:, 0], fe[:, 1]], np.r_[fe[:, 1], fe[:, 2]])), shape=(nv, nv)), directed=False)
    return dict(bnd_len=(V[b[:, 0]] - V[b[:, 1]]).norm(dim=-1).sum().item(), loops=loops, comps=len(np.unique(lab[np.unique(fe)])))
