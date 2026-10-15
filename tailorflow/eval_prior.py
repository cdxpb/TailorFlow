"""Protocol-A evaluation of a prior. Per class, the references are all female val+test garments; the prior is sampled
without coverage conditioning on each reference's own body. 1-NNA / COV / MMD (CD and EMD) on 2048-point clouds and
equal-size sets (mean over 5 random subsets when the sets differ), JSD, and the % of garment points inside the body mesh.
`floor` = evenly spaced female train garments (the same-distribution reference); `ref` rows give the data's own inside%."""
import sys, os, json, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import trimesh
from tailorflow.metrics import pairwise_cd, pairwise_emd, jsd
from tailorflow.data_io import MeshClothDataset, C3D_ROOT
from tailorflow.common import EXPS, load_vae, load_prior, sample_prior

p = argparse.ArgumentParser()
p.add_argument('--prior_exp', default='prior')
p.add_argument('--prior_ckpt', default='ckpt_020000.pt')
p.add_argument('--classes', default='Upper,Trousers,Skirt,Dress,Jumpsuit,DressJumpsuit')
p.add_argument('--steps', type=int, default=64)
p.add_argument('--solver', default='heun')
p.add_argument('--cfg', type=float, default=1.0)
p.add_argument('--bs', type=int, default=16)
p.add_argument('--k', type=int, default=5)
p.add_argument('--floor', type=int, default=1)
p.add_argument('--out', default='')
args = p.parse_args()

dev = torch.device('cuda')
pc = json.load(open(os.path.join(EXPS, args.prior_exp, 'config.json')))
vae, vc, _ = load_vae(pc['vae'], dev)
prior = load_prior(args.prior_exp, args.prior_ckpt, vc, dev)
cates = vc['cates'].split(',')
out = args.out or os.path.join(EXPS, args.prior_exp, f'eval_A_{os.path.splitext(args.prior_ckpt)[0]}')
os.makedirs(out, exist_ok=True)
FACES = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smpl_faces.npy'))
DS = {sp: MeshClothDataset(cates, sp) for sp in ('train', 'val', 'test')}
npz = lambda C, sp, stem: np.load(os.path.join(C3D_ROOT, sp, C, stem + '.npz'))


def female(C, splits):
    return [(C, sp, stem) for sp in splits for c, _, stem in DS[sp].items if c == C and int(npz(C, sp, stem)['gender']) == 0]


def floor(C, n):
    it = female(C, ('train',))
    return [it[j] for j in np.linspace(0, len(it) - 1, min(n, len(it))).round().astype(int)]


def points(keys):
    return np.stack([np.asarray(npz(*k)['points'], np.float32) for k in keys])


@torch.no_grad()
def generate(C, keys):
    f = os.path.join(out, f'gen_{C}.pt')
    if os.path.exists(f):
        return torch.load(f)['raw'].float().numpy()
    pos = {sp: {it[2]: i for i, it in enumerate(DS[sp].items) if it[0] == C} for sp in ('val', 'test')}
    order = keys + [(C, 'test', st) for st in pos['test'] if (C, 'test', st) not in keys]   # the paper's run also sampled
    items = [DS[sp][pos[sp][stem]] for _, sp, stem in order]                                # the male test bodies
    G = torch.zeros(len(items), vc['L'], 3)
    torch.manual_seed(cates.index(C))
    for s in range(0, len(items), args.bs):
        ch = items[s:s + args.bs]
        bv, bn, beta = (torch.stack([it[k] for it in ch]).to(dev) for k in ('body_verts', 'body_normals', 'beta'))
        cl = torch.full((len(ch),), cates.index(C), dtype=torch.long, device=dev)
        zg, lat = sample_prior(prior, vae, cl, bv, None, torch.zeros_like(cl, dtype=torch.bool), steps=args.steps, cfg=args.cfg,
                               L=vc['L'], solver=args.solver)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = vae.decode(zg, lat[..., :3].contiguous(), lat[..., 3:].contiguous(), bv, bn, None, beta=beta)
        G[s:s + len(ch)] = y.float().cpu()
        print(f'{C}: sampled {s + len(ch)}/{len(items)}', flush=True)
    G = G[:len(keys)].half()
    torch.save({'raw': G, 'names': [f'{sp}/{stem}' for _, sp, stem in keys]}, f)
    return G.float().numpy()


def sets(C):
    """{name: (clouds [N, P, 3], body keys [(cate, split, stem)])}; DressJumpsuit = Dress ++ Jumpsuit evenly cut to 500."""
    if C == 'DressJumpsuit':
        D, J = sets('Dress'), sets('Jumpsuit')
        out_ = {}
        for m in D:
            X, k = np.concatenate([D[m][0], J[m][0]]), D[m][1] + J[m][1]
            sel = np.linspace(0, len(X) - 1, min(500, len(X))).round().astype(int)
            out_[m] = (X[sel], [k[i] for i in sel])
        return out_
    r = female(C, ('val', 'test'))
    S = {'ref': (points(r), r), 'ours': (generate(C, r), r)}
    if args.floor:
        S['floor'] = (points(floor(C, len(r))), floor(C, len(r)))
        S['floor_inside'] = (points(floor(C, 500)), floor(C, 500))
    return S


def sub(X):
    X = torch.as_tensor(X).float()
    idx = torch.stack([torch.randperm(X.shape[1], generator=torch.Generator().manual_seed(1 + i))[:2048] for i in range(len(X))])
    return torch.gather(X, 1, idx.unsqueeze(-1).expand(-1, -1, 3)).contiguous().to(dev)


def nna(Mrr, Mrg, Mgg):
    M = torch.cat([torch.cat([Mrr, Mrg], 1), torch.cat([Mrg.t(), Mgg], 1)], 0).clone()
    M.fill_diagonal_(float('inf'))
    lab = torch.cat([torch.ones(len(Mrr)), torch.zeros(len(Mgg))]).to(dev)
    return (lab[M.argmin(0)] == lab).float().mean().item() * 100


def score(R, G, rr):
    n, res = min(len(R), len(G)), {}
    rg = (pairwise_cd(R, G), pairwise_emd(R, G, batch_size=32))
    gg = (pairwise_cd(G, G), pairwise_emd(G, G, batch_size=32))
    for t, d in enumerate(('cd', 'emd')):
        cov, mmd, nn = [], [], []
        for s in range(args.k):
            g = torch.Generator().manual_seed(1000 + s)
            ir, ig = torch.randperm(len(R), generator=g)[:n].to(dev), torch.randperm(len(G), generator=g)[:n].to(dev)
            D = rg[t][ir][:, ig].t()
            cov.append(D.argmin(0).unique().numel() / D.shape[1] * 100)
            mmd.append(D.min(0).values.mean().item())
            nn.append(nna(rr[t][ir][:, ir], rg[t][ir][:, ig], gg[t][ig][:, ig]))
            if len(R) == n == len(G):
                break
        res |= {f'nna_{d}': np.mean(nn), f'cov_{d}': np.mean(cov), f'mmd_{d}': np.mean(mmd)}
    return res | {'jsd': jsd(G, R)}


def inside(X, keys):
    v = [trimesh.Trimesh(np.asarray(npz(*k)['body'], np.float64), FACES, process=False).contains(x.astype(np.float64)).mean() * 100
         for x, k in zip(X, keys)]
    return {'inside': np.mean(v), 'inside_std': np.std(v)}


rows, cols = [], ['cls', 'method', 'n', 'nna_cd', 'nna_emd', 'cov_cd', 'cov_emd', 'mmd_cd', 'mmd_emd', 'jsd', 'inside', 'inside_std']
for C in args.classes.split(','):
    S = sets(C)
    R = sub(S['ref'][0])
    rr = (pairwise_cd(R, R), pairwise_emd(R, R, batch_size=32))
    for m in [m for m in ('ours', 'floor', 'ref') if m in S]:
        res = (score(R, sub(S[m][0]), rr) if m != 'ref' else {}) | inside(*S['floor_inside' if m == 'floor' else m])
        rows.append({'cls': C, 'method': m, 'n': len(S[m][0]), **{k: round(float(v), 5) for k, v in res.items()}})
        print(' '.join(f'{k} {v}' for k, v in rows[-1].items()), flush=True)
        with open(os.path.join(out, 'metrics.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
print('->', out, flush=True)
