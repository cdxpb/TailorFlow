"""VAE reconstruction of unseen garments (all female val+test garments of a class): encode, decode on the garment's own
body with its own coverage. CD = sqrt(mean sq. NN distance recon->GT + GT->recon), P2S = mean distance of the decoded
points to the GT garment mesh; both in cm."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import trimesh
from pytorch3d.ops import knn_points
from tailorflow.common import load_vae
from tailorflow.data_io import MeshClothDataset, C3D_ROOT
from tailorflow.preprocessing.smpl import SCALE

p = argparse.ArgumentParser()
p.add_argument('--vae', default='checkpoints/vae/ckpt_020000.pt')
p.add_argument('--classes', default='Upper,Trousers,Skirt,Dress,Jumpsuit')
p.add_argument('--bs', type=int, default=8)
p.add_argument('--seed', type=int, default=1234)
args = p.parse_args()

dev = torch.device('cuda')
vae, vc, _ = load_vae(args.vae, dev)
DS = {sp: MeshClothDataset(vc['cates'].split(','), sp) for sp in ('val', 'test')}
cm = 100 * SCALE
for C in args.classes.split(','):
    npz = lambda sp, stem: np.load(os.path.join(C3D_ROOT, sp, C, stem + '.npz'))
    items = [(sp, DS[sp][i]) for sp in ('val', 'test') for i, it in enumerate(DS[sp].items)
             if it[0] == C and int(npz(sp, it[2])['gender']) == 0]
    torch.manual_seed(args.seed)
    cd, p2s = [], []
    for s in range(0, len(items), args.bs):
        ch = items[s:s + args.bs]
        x, bv, bn, beta, occ = (torch.stack([it[k] for _, it in ch]).to(dev)
                                for k in ('tr_points', 'body_verts', 'body_normals', 'beta', 'body_occupancy'))
        with torch.no_grad():
            z_body, kv, lv = vae.encode_body(bv, bn, beta)
            zg, pt, feat = vae.encode(x, bv, sample=True)
            y = vae.decode_from(zg, pt, feat, bv, bn, z_body, kv, lv, occ).float()
        for j, (sp, it) in enumerate(ch):
            a, b = y[j:j + 1], x[j:j + 1].float()
            cd.append(float(knn_points(a, b, K=1).dists.mean() + knn_points(b, a, K=1).dists.mean()) ** 0.5 * cm)
            z = npz(sp, it['mid'])
            q = y[j].cpu().numpy().astype(np.float64)
            m = trimesh.Trimesh(np.asarray(z['verts'], np.float64), np.asarray(z['faces'], np.int64), process=False)
            p2s.append(float(np.linalg.norm(q - trimesh.proximity.closest_point(m, q)[0], axis=1).mean()) * cm)
    print(f'{C:9s} n {len(items):3d}  CD {np.mean(cd):.3f} cm  P2S {np.mean(p2s):.3f} cm', flush=True)
