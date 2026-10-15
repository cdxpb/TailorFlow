"""Sanity check of a built dataset -> <root>/sanity/:
  stats.txt       per split/class stats
  <Category>.png  train items: GT points with occupied body vertices in red, garment mesh, occupancy on the body"""
import os, sys, json, argparse
from multiprocessing import Pool
import numpy as np
import torch
from PIL import Image
from pytorch3d.ops import knn_points

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tailorflow.preprocessing.smpl import DATA, DATASET, SCALE, load_smpl
from tailorflow.common import inside_pct
from tailorflow.vis import render_points, render_mesh, label

p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
p.add_argument('--n', type=int, default=8, help='rendered train items per class')
p.add_argument('--data', default=DATA)
p.add_argument('--workers', type=int, default=16)
a = p.parse_args()
root = os.path.join(a.data, DATASET)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out = os.path.join(root, 'sanity')
os.makedirs(out, exist_ok=True)
meta = json.load(open(os.path.join(root, 'meta.json')))
T = lambda x: torch.from_numpy(np.asarray(x)).float().to(dev)
FEET = np.isin(load_smpl('female')['weights'].argmax(1), [10, 11])
OCC_RGB = torch.tensor([[0.85, 0.83, 0.81], [0.9, 0.25, 0.25]])


def files(split, c):
    d = os.path.join(root, split, c)
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith('.npz') and '.' not in f[:-4]]


def file_stats(f):
    z = np.load(f)
    x, bv, bn = (torch.from_numpy(z[k])[None].float() for k in ('points', 'body', 'body_normals'))
    return (inside_pct(x, bv, bn), knn_points(x, x, K=2).dists[0, :, 1].sqrt().mean().item() * SCALE * 1000, z['occ'].mean() * 100,
            np.ptp(z['body'][:, 1]) * SCALE, z['occ'][FEET].sum(), z['occ'].any())


def init_worker():
    torch.set_num_threads(1)


lines = [f"{root} | pose {meta['pose']} | 1 unit = {SCALE:.4f} m",
         f"{'split':5s} {'class':9s} {'n':>5s} {'inside%':>8s} {'nn_mm':>6s} {'occ%':>6s} {'height_m':>8s} "
         + f"{'feet_v':>6s} {'empty_occ':>9s}"]
pool = Pool(a.workers, initializer=init_worker)
for split, cats in meta['counts'].items():
    for c in cats:
        fs = files(split, c)
        sub = fs[::max(1, len(fs) // 200)]
        ins, nn, occ, hgt, feet, nonempty = zip(*pool.map(file_stats, sub, chunksize=4))
        lines.append(f"{split:5s} {c:9s} {len(fs):5d} {np.mean(ins):8.2f} {np.mean(nn):6.2f} {np.mean(occ):6.1f} {np.mean(hgt):8.3f} "
                     + f'{np.mean(feet):6.1f} {len(sub) - sum(nonempty):9d}')
        print(lines[-1], flush=True)
pool.close()
open(os.path.join(out, 'stats.txt'), 'w').write('\n'.join(lines) + '\n')

for c in meta['counts']['train']:
    fs = files('train', c)
    rows = []
    for f in fs[::max(1, len(fs) // a.n)][:a.n]:
        z = np.load(f)
        bv, x = T(z['body']), T(z['points'])
        allp = torch.cat([bv, T(z['verts'])])
        bb = (allp.min(0).values, allp.max(0).values)
        mesh = (T(z['verts']), torch.from_numpy(z['faces'].astype(np.int64)).to(dev))
        name = f'{c} {os.path.basename(f)[:-4]}'
        cells = []
        for back in (False, True):
            side = 'back' if back else 'front'
            cells.append(label(render_points(x, bv, bb, back=back, occ=T(z['occ'])[:, None]), f'{name} points+occ {side}'))
            cells.append(label(render_mesh(mesh, bv, bb, back=back), f'mesh {side}'))
            cells.append(label(render_mesh(None, bv, bb, back=back, body_rgb=OCC_RGB[torch.from_numpy(z['occ'].astype(np.int64))]),
                               f'occ {side}: {z["occ"].sum()} verts, feet {z["occ"][FEET].sum()}'))
        rows.append(np.concatenate(cells, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(out, f'{c}.png'))
    print('wrote', os.path.join(out, f'{c}.png'), flush=True)
