"""Dataset body model: beta [10] + gender -> the exact body preprocessing/build.py writes. For a fixed pose SMPL is affine
in beta, so each gender is stored as an [11, 6890*3] matrix. Beta sampling uses per-gender Gaussians fit to the train set;
`occ[cls]` holds the train coverage masks, used to draw in-distribution random coverage. Both come from body_stats.npz
(written by preprocessing/body_stats.py), so the demo needs only SMPL and the checkpoints."""
import os
import numpy as np
import torch
from tailorflow.preprocessing.smpl import GENDERS, NB, load_smpl, body

STATS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'body_stats.npz')


class BodyModel:
    def __init__(self, cates, stats=STATS, device='cpu'):
        W = []
        for g in GENDERS:
            m = load_smpl(g)
            b0 = body(m, np.zeros(NB))
            W.append(np.concatenate([np.stack([body(m, e) - b0 for e in np.eye(NB)]), b0[None]]).reshape(NB + 1, -1))
        self.W = torch.from_numpy(np.stack(W).astype(np.float32)).to(device)
        self.faces = m['faces']
        z = np.load(stats)
        self.occ = {c: np.unpackbits(z[f'occ_{c}'], axis=1, count=len(b0)).astype(bool) for c in cates}
        self.names, self.k = list(GENDERS), len(GENDERS)
        self.n, self.mean, self._chol = z['n'], z['mean'], list(z['chol'])

    def verts(self, beta, space=0):
        b = torch.as_tensor(beta, dtype=self.W.dtype, device=self.W.device).reshape(-1, NB)
        x = torch.cat([b, torch.ones(len(b), 1, dtype=b.dtype, device=b.device)], 1)
        return (x @ self.W[space]).reshape(len(b), -1, 3)

    def normals(self, verts):
        """trimesh vertex_normals, as written by preprocessing/build.py."""
        import trimesh
        out = [trimesh.Trimesh(v.detach().cpu().numpy().astype(np.float64), self.faces, process=False).vertex_normals
               for v in verts]
        return torch.from_numpy(np.stack(out).astype(np.float32)).to(verts.device)

    def sample_beta(self, space=0, rng=None, scale=1.0):
        rng = rng or np.random
        return (self.mean[space] + scale * (self._chol[space] @ rng.standard_normal(NB))).astype(np.float32)
