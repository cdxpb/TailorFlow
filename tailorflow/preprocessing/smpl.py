"""SMPL (v1.0/v1.1 pkl, first 10 betas) forward kinematics and the fixed pose/normalization of the C3D dataset."""
import os, pickle
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
SMPL_DIR = os.path.join(DATA, 'smpl')
GENDERS = ('female', 'male')
SCALE = 1.044149153487508
NB = 10
POSE = 'tapose'
DATASET = 'C3D_TAPOSE'


def pose_vec():
    """CLOTH3D rest pose: SMPL zero pose (arms in T) with the hips abducted +-0.15 rad."""
    p = np.zeros(72)
    p[5], p[8] = 0.15, -0.15
    return p


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        self.__dict__.update(s if isinstance(s, dict) else {})


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _Stub if module.startswith('chumpy') else super().find_class(module, name)


def _arr(x):
    x = x.x if isinstance(x, _Stub) else x
    return np.asarray(x.toarray() if hasattr(x, 'toarray') else x, dtype=np.float64)


def load_smpl(gender, smpl_dir=SMPL_DIR):
    d = _Unpickler(open(os.path.join(smpl_dir, f'SMPL_{gender.upper()}.pkl'), 'rb'), encoding='latin1').load()
    return {'v_template': _arr(d['v_template']), 'shapedirs': _arr(d['shapedirs'])[:, :, :NB], 'posedirs': _arr(d['posedirs']),
            'J_regressor': _arr(d['J_regressor']), 'weights': _arr(d['weights']),
            'parents': np.asarray(d['kintree_table'])[0].astype(np.int64), 'faces': np.asarray(d['f']).astype(np.int64)}


def rodrigues(r):
    th = np.linalg.norm(r, axis=-1, keepdims=True)[..., None]
    k = r / np.maximum(th[..., 0], 1e-12)
    K = np.zeros(r.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def forward(m, betas, pose):
    """-> posed verts [6890,3], rest joints [24,3], per-joint skinning transforms [24,4,4]."""
    v = m['v_template'] + m['shapedirs'] @ np.asarray(betas, dtype=np.float64).reshape(-1)[:NB]
    J = m['J_regressor'] @ v
    R = rodrigues(np.asarray(pose, dtype=np.float64).reshape(24, 3))
    v = v + m['posedirs'] @ (R[1:] - np.eye(3)).reshape(-1)
    G = np.zeros((24, 4, 4))
    for i in range(24):
        L = np.eye(4)
        L[:3, :3] = R[i]
        L[:3, 3] = J[i] - (J[m['parents'][i]] if i else 0)
        G[i] = G[m['parents'][i]] @ L if i else L
    A = G.copy()
    A[:, :3, 3] -= np.einsum('jab,jb->ja', G[:, :3, :3], J)
    return skin(v, m['weights'], A), J, A


def skin(x, w, A):
    T = np.einsum('vj,jab->vab', w, A)
    return np.einsum('vab,vb->va', T[:, :3, :3], x) + T[:, :3, 3]


def body(m, betas):
    """Normalized dataset body: SMPL(betas) in the dataset pose, pelvis at the origin, divided by SCALE."""
    v, J, _ = forward(m, betas, pose_vec())
    return (v - J[0]) / SCALE
