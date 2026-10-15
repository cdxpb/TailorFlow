import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, trimesh
from PIL import Image
from tailorflow.data_io import MeshClothDataset, C3D_ROOT
from tailorflow.common import EXPS, load_vae, load_prior, sample_prior, decode_occ, inside_pct
from tailorflow.garment_mesh import mesh_garment, nn_spacing, topo, area
from tailorflow.vis import render_points, render_mesh, label

p = argparse.ArgumentParser(description='prior sample -> VAE decode -> mesh, with renders and mesh stats')
p.add_argument('--prior_exp', default='prior')
p.add_argument('--prior_ckpt', default='ckpt_020000.pt')
p.add_argument('--split', default='val')
p.add_argument('--classes', default='')
p.add_argument('--n_per_class', type=int, default=3)
p.add_argument('--occ', default='0,1')
p.add_argument('--L', type=int, default=32768, help='points per garment; above 4096 the prior samples in 4096-point partitions')
p.add_argument('--steps', type=int, default=64)
p.add_argument('--cfg', type=float, default=1.0)
p.add_argument('--presets', default='b3x_lap30,b3x_lap15', help='garment_mesh.POISSON_PRESETS names')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', default='')
args = p.parse_args()

dev = torch.device('cuda')
prior_cfg_vae = __import__('json').load(open(os.path.join(EXPS, args.prior_exp, 'config.json')))['vae']
vae, vc, vck = load_vae(prior_cfg_vae, dev)
prior = load_prior(args.prior_exp, args.prior_ckpt, vc, dev)
cates = vc['cates'].split(',')
classes = args.classes.split(',') if args.classes else cates
presets = args.presets.split(',')
out = args.out or os.path.join(EXPS, args.prior_exp, f"pipeline_L{args.L}_{os.path.splitext(args.prior_ckpt)[0]}")
os.makedirs(f'{out}/meshes', exist_ok=True)
print(f"prior {args.prior_exp}/{args.prior_ckpt} step {prior['step']} | VAE step {vck['step']} | "
      f"L {args.L} steps {args.steps} cfg {args.cfg} | "
      f"presets {presets} | data {C3D_ROOT}", flush=True)

ds = MeshClothDataset(cates, args.split)
log = open(f'{out}/stats.txt', 'a')
for cls in classes:
    ci = cates.index(cls)
    b = torch.utils.data.default_collate([ds[j] for j in [j for j in range(len(ds)) if ds.items[j][0] == cls][:args.n_per_class]])
    bv, bn = b['body_verts'].to(dev).float(), b['body_normals'].to(dev).float()
    occ_raw, beta = b['body_occupancy'].to(dev).float(), b['beta'].to(dev)
    n = bv.shape[0]
    cl = torch.full((n,), ci, dtype=torch.long, device=dev)
    for o in [int(x) for x in args.occ.split(',')]:
        use_occ = torch.full((n,), bool(o), dtype=torch.bool, device=dev)
        t0 = time.time()
        torch.manual_seed(args.seed * 1000 + ci * 2 + o)
        zg, lat = sample_prior(prior, vae, cl, bv, occ_raw, use_occ, steps=args.steps, cfg=args.cfg, L=args.L)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            y = vae.decode(zg, lat[..., :3].contiguous(), lat[..., 3:].contiguous(), bv, bn, decode_occ(occ_raw, use_occ), beta=beta)
        y = y.float()
        print(f'{cls} occ{o}: sampled+decoded in {time.time() - t0:.1f}s', flush=True)
        rows_f, rows_b = [], []
        for i in range(n):
            cells = []
            pts = y[i]
            name = f'{cls}_occ{o}_{i}_L{args.L}'
            np.save(f'{out}/meshes/{name}_points.npy', pts.cpu().numpy())
            cells.append((f'points N={pts.shape[0]}', 'pts', pts))
            for pre in presets:
                t1 = time.time()
                V, Fc = mesh_garment(pts, bv[i], preset=pre)
                trimesh.Trimesh(V.cpu().numpy(), Fc.cpu().numpy(), process=False).export(f'{out}/meshes/{name}_{pre}.ply')
                st = topo(V, Fc)
                line = (f"{name}_{pre} mesh_verts {V.shape[0]} | area {area(V, Fc):.3f} loops {st['loops']} comps {st['comps']} "
                        f"bnd {st['bnd_len']:.2f}m | inside {inside_pct(pts[None], bv[i:i + 1], bn[i:i + 1]):.2f}% | "
                        f"nn {nn_spacing(pts).mean().item() * 1000:.2f}mm "
                        f"| mesh {time.time() - t1:.1f}s")
                print(line, flush=True)
                log.write(line + '\n')
                log.flush()
                cells.append((f'mesh ({pre})', 'mesh', (V, Fc)))
            bb = (pts.min(0).values, pts.max(0).values)
            rf, rb = [], []
            for lab, kind, M in cells:
                fn = render_points if kind == 'pts' else render_mesh
                rf.append(label(fn(M, bv[i], bb), f'{cls}#{i} occ{o} {lab}'))
                rb.append(label(fn(M, bv[i], bb, back=True), f'{lab} back'))
            rows_f.append(np.concatenate(rf, 1))
            rows_b.append(np.concatenate(rb, 1))
        Image.fromarray(np.concatenate(rows_f + rows_b, 0)).save(f'{out}/{cls}_occ{o}.png')
print('pipeline done ->', out, flush=True)
