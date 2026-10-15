"""EditAug coverage-cut sanity renders -> <root>/sanity/<Category>_edit.png, one row per train item:
garment | cuts: hem, sleeve (asymmetric), pair, random spec. Each cut shows the kept GT points with the cut coverage in red
and the cut garment resampled like the dataset."""
import os, sys, json, argparse, random
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tailorflow.preprocessing.smpl import DATA, DATASET
from tailorflow.data_io import MeshClothDataset
from tailorflow.occ_aug import EditAug
from tailorflow.vis import render_points, render_mesh, label

p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
p.add_argument('--n', type=int, default=8, help='rendered train items per class')
p.add_argument('--data', default=DATA)
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
root = os.path.join(a.data, DATASET)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out = os.path.join(root, 'sanity')
os.makedirs(out, exist_ok=True)
cates = list(json.load(open(os.path.join(root, 'meta.json')))['counts']['train'])
random.seed(a.seed)
torch.manual_seed(a.seed)

ds = MeshClothDataset(cates, 'train', root=root)
tmpl = torch.stack([ds[i]['body_verts'] for i in range(0, len(ds), max(1, len(ds) // 300))]).mean(0)
ea = EditAug(ds.items, tmpl, device=dev)
print(f'{root}: {len(ds)} train items, sleeve-eligible {ea.has_sleeve.any(1).float().mean().item() * 100:.0f}%, '
      f'with pair partners {np.mean([p is not None and len(p) > 0 for p in ea.partners]) * 100:.0f}%', flush=True)


def fmt(spec):
    return '+'.join(f'hem {o[1]:.2f}' if o[0] == 'hem' else f'sleeve {"LR"[o[1]]} {o[2]:.2f}' if o[0] == 'sleeve'
                    else f'pair {ds.items[int(o[1])][2]}' for o in spec) or 'none'


for c in cates:
    idx = [i for i, it in enumerate(ds.items) if it[0] == c]
    rows = []
    for i in idx[::max(1, len(idx) // a.n)][:a.n]:
        it = ds[i]
        cat, stem = it['sid'], it['mid']
        x, bv, occ = it['tr_points'].to(dev), it['body_verts'].to(dev), it['body_occupancy'].to(dev)
        r = ea.row[(cat, stem)]
        z = np.load(os.path.join(ds.items[i][1], f'{stem}.npz'))
        allp = torch.cat([bv, x])
        bb = (allp.min(0).values, allp.max(0).values)
        mesh = (torch.from_numpy(z['verts']).to(dev), torch.from_numpy(z['faces'].astype(np.int64)).to(dev))
        cells = [label(render_mesh(mesh, bv, bb), f'{cat} {stem}'), label(render_points(x, bv, bb, occ=occ), 'GT points + occ')]
        sides = ea.has_sleeve[r].nonzero(as_tuple=True)[0].tolist()
        partners = ea.partners[r]
        edits = [[('hem', random.uniform(*ea.u_range))],
                 [('sleeve', 0, random.uniform(*ea.u_range)), ('sleeve', 1, random.uniform(*ea.u_range))][:len(sides)]
                 if len(sides) == 2 else [('sleeve', s, random.uniform(*ea.u_range)) for s in sides],
                 [('pair', int(partners[random.randrange(len(partners))]))] if partners is not None and len(partners) else [],
                 ea.random_spec(cat, stem)]
        for spec in edits:
            cut = ea.cut(x, bv, occ, cat, stem, spec) if spec else None
            if cut is None:
                blank = np.full_like(cells[0], 255)
                cells += [label(blank, f'{fmt(spec)} (n/a)'), blank]
                continue
            cells += [label(render_points(x[cut['keep']], bv, bb, occ=cut['m'].float().view(-1, 1)), f'kept + occ: {fmt(spec)}'),
                      label(render_points(ea.cut_points(cut, dev), bv, bb), f"resampled ({cut['kept_frac'] * 100:.0f}% area)")]
        rows.append(np.concatenate(cells, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(out, f'{c}_edit.png'))
    print('wrote', os.path.join(out, f'{c}_edit.png'), flush=True)
