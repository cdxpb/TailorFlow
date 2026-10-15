"""Shape interpolation between two garments of one class. The latent is the garment's point cloud on the canonical body,
so the morph is done there: points are matched by optimal transport (squared distance, on 4k FPS subsets, extended to all
points by an inverse-distance kNN field), both clouds are warped along the matched straight lines and cross-dissolved,
and zg is lerped. A light SDEdit pass of the prior (strength s*4a(1-a), so the endpoints stay exact) cleans the
in-between frames. Frames advance evenly because every point moves at constant speed."""
import torch
from scipy.optimize import linear_sum_assignment
from pytorch3d.ops import sample_farthest_points, knn_points
from tailorflow.common import guided_local


def _field(X, A, D, k=8):
    kn = knn_points(X[None], A[None], K=k)
    w = 1.0 / (kn.dists[0].sqrt() + 1e-4)
    return (D[kn.idx[0]] * (w / w.sum(1, keepdim=True))[..., None]).sum(1)


def morph(A, B, alphas, n_ot=4096, seed=0):
    """A [N, C], B [M, C] latents (xyz first) -> one latent per alpha, min(N, M) rows each."""
    A4 = sample_farthest_points(A[None, :, :3], K=n_ot, random_start_point=False)[1][0]
    B4 = sample_farthest_points(B[None, :, :3], K=n_ot, random_start_point=False)[1][0]
    A4, B4 = A[A4], B[B4]
    col = torch.as_tensor(linear_sum_assignment(torch.cdist(A4[:, :3], B4[:, :3]).pow(2).cpu().numpy())[1], device=A.device)
    inv = torch.empty_like(col)
    inv[col] = torch.arange(len(col), device=A.device)
    DA, DB = _field(A[:, :3], A4[:, :3], B4[col] - A4), _field(B[:, :3], B4[:, :3], A4[inv] - B4)
    g = torch.Generator().manual_seed(seed)
    n = min(len(A), len(B))
    u, pb = torch.rand(n, generator=g).to(A.device), torch.randperm(len(B), generator=g)[:n].to(A.device)
    return [torch.where((u >= a)[:, None], A[:n] + a * DA[:n], B[pb] + (1 - a) * DB[pb]) for a in alphas]


@torch.no_grad()
def interpolate(prior, vae, cl, bv, zga, la, zgb, lb, alphas, strength=0.5, steps=32, seed=0, cl_b=None):
    """cl [1], bv [1, V, 3], zg [G], latents [N, C] -> [(zg [1, G], lat [1, n, C])] per alpha. cl_b: B's class when the two
    garments differ in category (the prior is conditioned on the nearer endpoint's class)."""
    out = []
    for a, lat in zip(alphas, morph(la, lb, alphas, seed=seed)):
        zg = ((1 - a) * zga + a * zgb)[None]
        s = strength * 4 * a * (1 - a)
        if s > 0:
            occ = torch.zeros(1, bv.shape[1], 1, device=bv.device)
            c = cl if cl_b is None or a < 0.5 else cl_b
            lat = guided_local(prior, vae, c, bv, occ, zg, init=lat[None], t0=1.0 - s, occ_w=0.0, steps=steps, seed=seed)[0]
        out.append((zg, lat[None]))
    return out
