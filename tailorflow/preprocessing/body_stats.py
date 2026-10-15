"""Writes viz/body_stats.npz from the built train split: per-gender beta Gaussians and the coverage masks of each class
(bit-packed), so the demo runs without the dataset."""
import os, sys, json
import numpy as np
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PKG))
from tailorflow.common import EXPS
from tailorflow.data_io import C3D_ROOT
from tailorflow.preprocessing.smpl import GENDERS, NB
from tailorflow.viz.bodymodel import STATS

cates = json.load(open(os.path.join(EXPS, 'vae', 'config.json')))['cates'].split(',')
B, G, occ = [], [], {}
for c in cates:
    O = []
    d = os.path.join(C3D_ROOT, 'train', c)
    for f in sorted(os.listdir(d)):
        if f.endswith('.npz') and '.' not in f[:-4]:
            z = np.load(os.path.join(d, f))
            B.append(z['betas'])
            G.append(int(z['gender']))
            O.append(z['occ'])
    occ[c] = np.packbits(np.stack(O).astype(bool), axis=1)
B, G = np.stack(B).astype(np.float64), np.asarray(G)
k = len(GENDERS)
np.savez_compressed(STATS, n=np.array([(G == g).sum() for g in range(k)]),
                    mean=np.stack([B[G == g].mean(0) for g in range(k)]),
                    chol=np.stack([np.linalg.cholesky(np.cov(B[G == g].T) + 1e-6 * np.eye(NB)) for g in range(k)]),
                    **{f'occ_{c}': o for c, o in occ.items()})
print('wrote', STATS)
