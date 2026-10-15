import os
import random
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import torch
from pytorch3d.ops import sample_farthest_points, sample_points_from_meshes, knn_points
from pytorch3d.structures import Meshes

_ADJ = {}


def smpl_adjacency(device):
    key = str(device)
    if key not in _ADJ:
        f = torch.from_numpy(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smpl_faces.npy')).astype(np.int64))
        e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
        n = int(f.max()) + 1
        e = torch.cat([e, e.flip(1), torch.arange(n).view(-1, 1).repeat(1, 2)], 0).t()
        a = torch.sparse_coo_tensor(e, torch.ones(e.shape[1]), (n, n)).coalesce()
        a = torch.sparse_coo_tensor(a.indices(), torch.ones_like(a.values()), (n, n)).coalesce()
        deg = torch.sparse.sum(a, 1).to_dense()
        an = torch.sparse_coo_tensor(a.indices(), 1.0 / deg[a.indices()[0]], (n, n)).coalesce()
        _ADJ[key] = (a.to(device), an.to(device))
    return _ADJ[key]


def smooth_occ(occ, rings):
    """SMPL-ring averaging of a known coverage mask (unknown = -1 is kept)."""
    if occ is None or rings <= 0:
        return occ
    _, an = smpl_adjacency(occ.device)
    x = occ[..., 0].float()
    m = (x > 0.5).float()
    for _ in range(rings):
        m = torch.sparse.mm(an, m.t().contiguous()).t()
    return torch.where(x > -0.5, m, x).unsqueeze(-1).to(occ.dtype)


VERT, LAT = 1, 0
CUT_SMOOTH, CUT_MIN_ISLAND = 15, 0.1


def cut_garment(V, F, owner, removed, smooth=CUT_SMOOTH, min_island=CUT_MIN_ISLAND):
    """Garment faces kept by a coverage crop (numpy). A garment vertex is cut if the body vertex owning it (raycast owner)
    was removed; the cut label is averaged over mesh neighbours `smooth` times (threshold 0.5) for a clean cut line, and
    kept pieces smaller than min_island of the largest kept piece are dropped."""
    n = len(owner)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr() + sp.eye(n)
    A = sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A
    x = removed[owner].astype(np.float64)
    for _ in range(smooth):
        x = A @ x
    keep = ~(x > 0.5)[F].any(1)
    kf = np.where(keep)[0]
    if len(kf):
        ke = np.concatenate([F[kf][:, [0, 1]], F[kf][:, [1, 2]]])
        lab = connected_components(sp.coo_matrix((np.ones(len(ke)), (ke[:, 0], ke[:, 1])), shape=(n, n)), directed=False)[1]
        area = 0.5 * np.linalg.norm(np.cross(V[F[kf, 1]] - V[F[kf, 0]], V[F[kf, 2]] - V[F[kf, 0]]), axis=1)
        comp_area = np.bincount(lab[F[kf, 0]], weights=area, minlength=n)
        keep[kf[comp_area[lab[F[kf, 0]]] < min_island * comp_area.max()]] = False
    return keep


def arm_regions(template):
    """Per-arm vertex masks [2, V] (SMPL part map, split by side) and the shoulder->hand parameter of each arm vertex:
    projection onto the axis from the arm/torso boundary ring centre to the farthest arm vertex."""
    t = template.float()
    pm = torch.from_numpy(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smpl_part_map.npy')) == 1).to(t.device)
    a, _ = smpl_adjacency(t.device)
    touches_torso = torch.sparse.mm(a, (~pm).float().view(-1, 1)).view(-1) > 0
    arm = torch.zeros(2, t.shape[0], dtype=torch.bool, device=t.device)
    armt = torch.zeros(t.shape[0], device=t.device)
    for s, sgn in enumerate((1, -1)):
        m = pm & (t[:, LAT] * sgn > 0)
        sh = t[m & touches_torso].mean(0)
        hand = t[m][(t[m] - sh).norm(dim=-1).argmax()]
        v = hand - sh
        arm[s] = m
        armt[m] = (((t[m] - sh) @ v) / (v * v).sum().clamp(min=1e-8)).clamp(0, 1)
    return arm, armt


def smaller_partners(O, thresh=0.85, min_ratio=0.55, max_ratio=0.97, hops=3, min_ratio_chain=0.15):
    """Same-class garments whose coverage lies (>thresh) inside garment i's, chained up to `hops`."""
    Of = O.float()
    sz = Of.sum(1)
    contain = (Of @ Of.t()) / sz.view(1, -1).clamp(min=1)
    ratio = sz.view(1, -1) / sz.view(-1, 1).clamp(min=1)
    ok = (contain.t() > thresh) & (ratio > min_ratio) & (ratio < max_ratio)
    ok.fill_diagonal_(False)
    reach = ok.clone()
    for _ in range(hops - 1):
        reach = reach | (reach.float() @ ok.float() > 0)
    reach.fill_diagonal_(False)
    ok = reach & (contain.t() > thresh) & (ratio > min_ratio_chain) & (ratio < max_ratio)
    return [ok[i].nonzero(as_tuple=True)[0] for i in range(O.shape[0])]


class EditAug:
    """Coverage cuts of a GT garment: a crop composes 1-2 ops, hem(u), per-arm sleeve(side, u) (one arm, or both at
    equal/different u) and pair(j) = occ_i & occ_j of a real smaller same-class garment. The cut is a subset of the
    item's GT mesh (faces whose raycast owner survives), so points on kept faces keep their positions."""

    def __init__(self, items, template, n=4096, u_range=(0.15, 0.75), min_removed=30, device='cpu'):
        self.items, self.n = items, n
        self.u_range, self.min_removed = u_range, min_removed
        self.arm, self.armt = arm_regions(template.to(device))
        self.row, self.path, occs, cls = {}, {}, [], []
        for cat, cdir, stem in items:
            self.row[(cat, stem)] = len(occs)
            self.path[(cat, stem)] = os.path.join(cdir, f'{stem}.npz')
            occs.append(np.load(self.path[(cat, stem)])['occ'])
            cls.append(cat)
        self.O = torch.from_numpy(np.stack(occs))
        self.has_sleeve = torch.stack([(self.O & self.arm[s].cpu()).sum(1) >= 32 for s in range(2)], 1)
        self.partners = [None] * len(occs)
        for c in sorted(set(cls)):
            idx = torch.tensor([i for i, cc in enumerate(cls) if cc == c])
            for k, js in zip(idx.tolist(), smaller_partners(self.O[idx])):
                self.partners[k] = idx[js]

    def mesh(self, cat, stem, device):
        z = np.load(self.path[(cat, stem)])
        return z['verts'], z['faces'].astype(np.int64), z['owner'].astype(np.int64)

    def _u(self):
        return random.uniform(*self.u_range)

    def random_spec(self, cat, stem):
        r = self.row[(cat, stem)]
        avail = ['hem']
        if bool(self.has_sleeve[r].any()):
            avail.append('sleeve')
        if self.partners[r] is not None and len(self.partners[r]):
            avail.append('pair')
        ops = random.sample(avail, min(len(avail), 2 if random.random() < 0.4 else 1))
        spec = []
        for op in ops:
            if op == 'hem':
                spec.append(('hem', self._u()))
            elif op == 'pair':
                js = self.partners[self.row[(cat, stem)]]
                if js is not None and len(js):
                    spec.append(('pair', int(js[random.randrange(len(js))])))
            else:
                sides = self.has_sleeve[r].nonzero(as_tuple=True)[0].tolist()
                mode = random.choice(['sym', 'asym', 'one']) if len(sides) == 2 else 'one'
                u = self._u()
                if mode == 'one':
                    spec.append(('sleeve', random.choice(sides), u))
                else:
                    spec += [('sleeve', 0, u), ('sleeve', 1, u if mode == 'sym' else self._u())]
        return spec

    def mask(self, occ, bv, spec):
        o = occ.view(-1) > 0.5
        m = o.clone()
        for op in spec:
            if op[0] == 'hem':
                yf = bv[:, VERT]
                lo, hi = yf[o].min(), yf[o].max()
                m &= ~(yf < lo + op[1] * (hi - lo))
            elif op[0] == 'sleeve':
                sub = o & self.arm[op[1]]
                if int(sub.sum()) < 32:
                    continue
                lo, hi = self.armt[sub].min(), self.armt[sub].max()
                m &= ~(self.arm[op[1]] & (self.armt > lo + op[2] * (hi - lo)))
            elif op[0] == 'pair':
                m &= self.O[op[1]].to(m.device)
        return m

    def fps(self, dense):
        return sample_farthest_points(torch.stack(dense), K=self.n, random_start_point=False)[0]

    def cut(self, x, bv, occ, cat, stem, spec):
        """Coverage crop as a subset of the garment mesh. -> dict(m = kept body vertices, keep = [n] which points of x lie
        on kept faces, kept_frac = kept / full cloth area, V, F, fk = kept faces) or None if degenerate."""
        o = occ.view(-1) > 0.5
        m = self.mask(occ, bv, spec)
        if int((o & ~m).sum()) < self.min_removed:
            return None
        V, F, owner = self.mesh(cat, stem, x.device)
        fk = cut_garment(V, F, owner, (o & ~m).cpu().numpy())
        if bool(fk.all()) or int(fk.sum()) < 16:
            return None
        kv = np.zeros(len(V), dtype=bool)
        kv[F[fk].reshape(-1)] = True
        Vt = torch.from_numpy(V).float().to(x.device)
        if x.is_cuda:
            keep = torch.from_numpy(kv).to(x.device)[knn_points(x[None].float(), Vt[None], K=1).idx[0, :, 0]]
        else:
            keep = torch.from_numpy(kv[cKDTree(V.astype(np.float32)).query(x.float().numpy(), k=1)[1]])
        fa = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
        return dict(m=m, keep=keep, kept_frac=float(fa[fk].sum() / fa.sum()), V=V, F=F, fk=fk)

    def cut_points(self, c, device):
        """The kept garment resampled like the dataset (4n area samples + FPS n)."""
        d = sample_points_from_meshes(Meshes([torch.from_numpy(c['V']).float().to(device)],
                                             [torch.from_numpy(c['F'][c['fk']]).to(device)]), 4 * self.n)[0]
        return self.fps([d])[0]


class CutDataset(torch.utils.data.Dataset):
    """Wraps MeshClothDataset: with probability p an item also carries a random EditAug coverage cut computed in the loader
    worker (CPU): cut (bool), cut_m [V] kept body vertices, cut_keep [n] points of the garment on kept faces, cut_frac
    cut_dense [4n, 3] area samples of the kept garment (FPS to n on the GPU)."""

    def __init__(self, base, p):
        self.base, self.p = base, p
        tmpl = torch.stack([base[i]['body_verts'] for i in range(0, len(base), max(1, len(base) // 300))]).mean(0)
        self.aug = EditAug(base.items, tmpl, device='cpu')
        self.items = base.items

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        it = dict(self.base[i])
        n, V = it['tr_points'].shape[0], it['body_occupancy'].shape[0]
        c = None
        if random.random() < self.p:
            c = self.aug.cut(it['tr_points'], it['body_verts'], it['body_occupancy'], it['sid'], it['mid'],
                             self.aug.random_spec(it['sid'], it['mid']))
        if c is None:
            it.update(cut=False, cut_m=torch.zeros(V), cut_keep=torch.ones(n, dtype=torch.bool), cut_dense=torch.zeros(4 * n, 3))
        else:
            d = sample_points_from_meshes(Meshes([torch.from_numpy(c['V']).float()], [torch.from_numpy(c['F'][c['fk']])]), 4 * n)[0]
            it.update(cut=True, cut_m=c['m'].float(), cut_keep=c['keep'], cut_dense=d)
        return it
