import os
import torch
from torch.utils.cpp_extension import load

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party')
_EXT = {}


def _ext(name):
    if name not in _EXT:
        src = {'chamfer': ['chamfer3d/chamfer_cuda.cpp', 'chamfer3d/chamfer3D.cu'], 'emdf': ['emd/emd_fast.cpp', 'emd/emd_fast_kernel.cu'], 'fps': ['fps/fps_kernel.cu']}[name]
        _EXT[name] = load(name=f'tailorflow_{name}', sources=[os.path.join(_DIR, s) for s in src], extra_cflags=['-O3', '-std=c++17'])
    return _EXT[name]


class _Chamfer(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, xyz1, xyz2):
        B, n, _ = xyz1.shape
        m = xyz2.shape[1]
        dev = xyz1.device
        dist1, dist2 = torch.zeros(B, n, device=dev), torch.zeros(B, m, device=dev)
        idx1, idx2 = torch.zeros(B, n, dtype=torch.int32, device=dev), torch.zeros(B, m, dtype=torch.int32, device=dev)
        torch.cuda.set_device(dev)
        _ext('chamfer').forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, g1, g2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        gx1, gx2 = torch.zeros_like(xyz1), torch.zeros_like(xyz2)
        _ext('chamfer').backward(xyz1, xyz2, gx1, gx2, g1.contiguous(), g2.contiguous(), idx1, idx2)
        return gx1, gx2


class _EMD(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, xyz1, xyz2):
        xyz1, xyz2 = xyz1.contiguous(), xyz2.contiguous()
        match = _ext('emdf').approxmatch_forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, match)
        return _ext('emdf').matchcost_forward(xyz1, xyz2, match)

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        return tuple(_ext('emdf').matchcost_backward(grad_cost.contiguous(), xyz1, xyz2, match))


def fps(pos, K, random_start=False):
    """Farthest point sampling [B, P, 3] -> (points [B, K, 3], idx [B, K]); same selection as pytorch3d."""
    B, P, _ = pos.shape
    if not pos.is_cuda or P < 64 or P > 16384:
        from pytorch3d.ops import sample_farthest_points
        return sample_farthest_points(pos, K=K, random_start_point=random_start)
    start = torch.randint(P, (B,), device=pos.device) if random_start else torch.zeros(B, dtype=torch.long, device=pos.device)
    idx = _ext('fps').fps(pos.float().contiguous(), K, start)
    return pos.gather(1, idx.unsqueeze(-1).expand(-1, -1, pos.shape[-1])), idx


def chamfer(x, y):
    """Squared nearest-neighbour distances [B, N] (x -> y) and [B, M] (y -> x)."""
    return _Chamfer.apply(x.contiguous(), y.contiguous())


def emd(x, y):
    """Approximate EMD per pair [B], normalized by the point count."""
    return _EMD.apply(x, y) / float(x.shape[1])


@torch.no_grad()
def pairwise_cd(X, Y, batch_size=64):
    out = torch.zeros(X.shape[0], Y.shape[0], device=X.device)
    for i in range(X.shape[0]):
        for s in range(0, Y.shape[0], batch_size):
            yb = Y[s:s + batch_size]
            d1, d2 = chamfer(X[i:i + 1].expand(yb.shape[0], -1, -1), yb)
            out[i, s:s + yb.shape[0]] = d1.mean(1) + d2.mean(1)
    return out


@torch.no_grad()
def pairwise_emd(X, Y, batch_size=16):
    """Pairwise approximate EMD [n_x, n_y]; heavier than CD, so the default batch is smaller."""
    out = torch.zeros(X.shape[0], Y.shape[0], device=X.device)
    for i in range(X.shape[0]):
        for s in range(0, Y.shape[0], batch_size):
            yb = Y[s:s + batch_size]
            out[i, s:s + yb.shape[0]] = emd(X[i:i + 1].expand(yb.shape[0], -1, -1), yb)
    return out


def pairwise(X, Y, dist='cd', batch_size=None):
    """Pairwise distance matrix [n_x, n_y] under 'cd' (Chamfer) or 'emd' (Earth Mover's)."""
    if dist == 'cd':
        return pairwise_cd(X, Y, batch_size or 64)
    if dist == 'emd':
        return pairwise_emd(X, Y, batch_size or 16)
    raise ValueError(dist)


@torch.no_grad()
def one_nna(gen, ref, batch_size=None, dist='cd'):
    """1-NN leave-one-out accuracy (fraction) under `dist`; 0.5 is ideal. Rows of `gen` and `ref`
    must already be resampled to the same point count."""
    return one_nna_mat(pairwise(gen, gen, dist, batch_size), pairwise(ref, ref, dist, batch_size),
                       pairwise(gen, ref, dist, batch_size))


def one_nna_mat(Dgg, Drr, Dgr):
    """1-NNA from precomputed [g,g], [r,r], [g,r] distance matrices (see `one_nna`)."""
    n_g, n_r = Dgg.shape[0], Drr.shape[0]
    M = torch.cat([torch.cat([Dgg, Dgr], 1), torch.cat([Dgr.t(), Drr], 1)], 0)
    M.fill_diagonal_(float('inf'))
    lab = torch.cat([torch.zeros(n_g), torch.ones(n_r)]).to(M)
    nn = M.topk(1, 0, False).indices[0]
    return (lab.index_select(0, nn) == lab).float().mean().item()


@torch.no_grad()
def coverage(gen, ref, dist='cd', batch_size=None):
    """COV(S_g, S_r): fraction of reference clouds matched to at least one generated cloud."""
    return coverage_mat(pairwise(gen, ref, dist, batch_size))


def coverage_mat(Dgr):
    """COV from the precomputed [g,r] matrix."""
    return Dgr.argmin(0).unique().numel() / Dgr.shape[1]


@torch.no_grad()
def mmd(gen, ref, dist='cd', batch_size=None):
    """MMD(S_g, S_r): mean over reference clouds of the distance to the nearest generated cloud."""
    return mmd_mat(pairwise(gen, ref, dist, batch_size))


def mmd_mat(Dgr):
    """MMD from the precomputed [g,r] matrix."""
    return Dgr.min(0).values.mean().item()


@torch.no_grad()
def jsd(gen, ref, bins=28, eps=1e-12):
    """Jensen-Shannon divergence (nats) between the marginal point distributions of two sets, following
    Achlioptas et al. / PointFlow: both sets are quantized into a `bins^3` grid over their joint bounding
    box and the histograms are compared."""
    g, r = gen.reshape(-1, 3).float(), ref.reshape(-1, 3).float()
    lo = torch.minimum(g.amin(0), r.amin(0))
    ext = (torch.maximum(g.amax(0), r.amax(0)) - lo).clamp(min=eps)

    def hist(x):
        idx = ((x - lo) / ext * bins).long().clamp(0, bins - 1)
        flat = idx[:, 0] * bins * bins + idx[:, 1] * bins + idx[:, 2]
        return torch.bincount(flat, minlength=bins ** 3).float()

    p, q = hist(g), hist(r)
    p, q = p / p.sum().clamp(min=eps), q / q.sum().clamp(min=eps)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * ((a + eps).log() - (b + eps).log())).sum()
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).item()
