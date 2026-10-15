"""data/smpl/SMPL_FEMALE.pkl -> tailorflow/smpl_faces.npy (SMPL triangles) and tailorflow/smpl_part_map.npy (0 torso, 1 arms, 2 legs, 3 head)."""
import os, sys
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PKG))
from tailorflow.preprocessing.smpl import load_smpl

m = load_smpl('female')
np.save(os.path.join(PKG, 'smpl_faces.npy'), m['faces'].astype(np.uint32))
joint = m['weights'].argmax(1)
part = np.zeros(joint.shape[0], dtype=np.int64)
part[np.isin(joint, [16, 17, 18, 19, 20, 21, 22, 23])] = 1
part[np.isin(joint, [1, 2, 4, 5, 7, 8, 10, 11])] = 2
part[np.isin(joint, [12, 15])] = 3
np.save(os.path.join(PKG, 'smpl_part_map.npy'), part)
print('wrote smpl_faces.npy', m['faces'].shape, 'and smpl_part_map.npy')
