"""data/C3D_RAW/{split}/<id> -> data/C3D_TAPOSE/{split}/<Category>/<id>.npz

Per garment: CLOTH3D's rest garment (Z-up, pelvis at the origin) is rotated to Y-up and kept in CLOTH3D's rest pose (arms in
T-pose, hips abducted +-0.15 rad); the body is SMPL(betas) in the same pose. Body and garment are normalized by (x - pelvis) / SCALE.
npz keys: points [4096,3], body [6890,3], body_normals [6890,3], occ [6890] (bool, raycast), betas [10], gender (0 f / 1 m),
verts/faces (garment mesh), owner (body vertex of each garment vertex, raycast).
"""
import os, sys, json, argparse, zlib, time
from multiprocessing import Pool
import numpy as np
import scipy.io as sio
import torch
import trimesh
from scipy.spatial import cKDTree
from pytorch3d.ops import sample_farthest_points

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tailorflow.preprocessing.smpl import DATA, DATASET, POSE, GENDERS, SCALE, load_smpl, forward, pose_vec

CATEGORY = {'Top': 'Upper', 'Tshirt': 'Upper', 'Dress': 'Dress', 'Jumpsuit': 'Jumpsuit', 'Skirt': 'Skirt', 'Trousers': 'Trousers'}
N_POINTS, N_DENSE = 4096, 16384


def read_obj(path):
    v, f = [], []
    for line in open(path):
        if line.startswith('v '):
            v.append(line.split()[1:4])
        elif line.startswith('f '):
            q = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            f += [[q[0], q[i], q[i + 1]] for i in range(1, len(q) - 1)]
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)


RING = np.stack([np.cos(np.arange(8) * np.pi / 4), np.zeros(8), np.sin(np.arange(8) * np.pi / 4)], 1)


def raycast(verts, faces, body, normals, near=0.05, max_up=0.5, min_ring=7):
    """-> (occ [6890] bool, owner [n_verts] int32). A body vertex is covered if the ray along its normal hits the garment
    (far hits only for rays with |n_y| < max_up: steep rays from feet/shoulders otherwise hit long hems or fabric far
    above/below), or if it is enclosed: >= min_ring of 8 horizontal rays hit the garment (the crotch under a skirt, which
    no normal ray reaches). owner = body vertex whose kept normal-ray hit is nearest each garment vertex."""
    body, normals = np.asarray(body, np.float64), np.asarray(normals, np.float64)
    ray = trimesh.Trimesh(verts, faces, process=False).ray
    loc, ir, _ = ray.intersects_location(body + normals * 1e-4, normals, multiple_hits=False)
    k = (np.linalg.norm(loc - body[ir], axis=1) < near) | (np.abs(normals[ir, 1]) < max_up)
    occ = np.zeros(len(body), dtype=bool)
    occ[ir[k]] = True
    free = np.where(~occ)[0]
    ring = ray.intersects_any(np.repeat(body[free], len(RING), 0), np.tile(RING, (len(free), 1))).reshape(len(free), -1)
    occ[free[ring.sum(1) >= min_ring]] = True
    owner = ir[k][cKDTree(loc[k]).query(verts)[1]] if k.any() else cKDTree(body).query(verts)[1]
    return occ, owner.astype(np.int32)


def occ_raycast(verts, faces, body, normals):
    return raycast(verts, faces, body, normals)[0]


def occupancy(rec):
    """occ and owner, computed from the stored float32 arrays so --occ_only equals a full build."""
    occ, owner = raycast(rec['verts'].astype(np.float64), rec['faces'].astype(np.int64), rec['body'], rec['body_normals'])
    return {'occ': occ, 'owner': owner}


def sample_surface(verts, faces, seed):
    pts, _ = trimesh.sample.sample_surface(trimesh.Trimesh(verts, faces, process=False), N_DENSE, seed=seed)
    idx = sample_farthest_points(torch.from_numpy(pts).float()[None], K=N_POINTS, random_start_point=False)[1][0].numpy()
    return pts[idx].astype(np.float32)


def garment_items(split, sid, m, smpl):
    """-> [(category, garment verts, body verts, faces)] in the normalized frame."""
    sm = smpl[GENDERS[int(m['gender'])]]
    bv, J, _ = forward(sm, m['shape'], pose_vec())
    out = []
    for g in m['outfit']._fieldnames:
        x, f = read_obj(os.path.join(RAW, split, sid, f'{g}.obj'))
        x = np.stack([x[:, 0], x[:, 2], -x[:, 1]], 1)
        out.append((CATEGORY[g], x / SCALE, (bv - J[0]) / SCALE, f))
    return out


def process(job):
    split, sid = job
    try:
        m = sio.loadmat(os.path.join(RAW, split, sid, 'info.mat'), struct_as_record=False, squeeze_me=True)
        n = 0
        for cat, gx, bv, faces in garment_items(split, sid, m, SMPL):
            dst = os.path.join(OUT, DATASET, split, cat, f'{sid}.npz')
            if os.path.exists(dst):
                continue
            seed = zlib.crc32(f'{split}/{cat}/{sid}'.encode())
            bmesh = trimesh.Trimesh(bv, SMPL['female']['faces'], process=False)
            rec = {'points': sample_surface(gx, faces, seed), 'body': bv.astype(np.float32),
                   'body_normals': np.asarray(bmesh.vertex_normals, dtype=np.float32),
                   'betas': np.asarray(m['shape'], dtype=np.float32), 'gender': np.int64(m['gender']),
                   'verts': gx.astype(np.float32), 'faces': faces.astype(np.int32)}
            rec.update(occupancy(rec))
            save(rec, dst)
            n += 1
        return split, sid, n, None
    except Exception as e:
        return split, sid, 0, f'{type(e).__name__}: {e}'


def save(rec, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst[:-4] + f'.tmp{os.getpid()}.npz'
    np.savez(tmp, **rec)
    os.replace(tmp, dst)


def update_occ(path):
    """--occ_only: recompute occ and owner of an existing item in place."""
    try:
        z = np.load(path)
        rec = {k: z[k] for k in z.files if k not in ('occ', 'owner') and not k.startswith(('points_', 'occ_'))}
        rec.update(occupancy(rec))
        save(rec, path)
        return path, None
    except Exception as e:
        return path, f'{type(e).__name__}: {e}'


def init(raw, out):
    global RAW, OUT, SMPL
    RAW, OUT = raw, out
    SMPL = {g: load_smpl(g) for g in GENDERS}
    torch.set_num_threads(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--splits', default='train,val,test')
    p.add_argument('--raw', default=os.path.join(DATA, 'C3D_RAW'))
    p.add_argument('--out', default=DATA)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--limit', type=int, default=0, help='samples per split (smoke test)')
    p.add_argument('--occ_only', action='store_true', help='only recompute occ and owner of existing items')
    a = p.parse_args()
    root = os.path.join(a.out, DATASET)
    for d, _, fs in os.walk(root):
        for f in fs:
            if '.tmp' in f:
                os.remove(os.path.join(d, f))
    if a.occ_only:
        paths = [os.path.join(d, f) for s in a.splits.split(',')
                 for d, _, fs in os.walk(os.path.join(root, s)) for f in sorted(fs) if f.endswith('.npz') and '.' not in f[:-4]]
        print(f'occ_only: {len(paths)} items | {a.workers} workers', flush=True)
        with Pool(a.workers, initializer=init, initargs=(a.raw, a.out)) as pool:
            errs = [e for _, e in pool.imap_unordered(update_occ, paths, chunksize=8) if e]
        print(f'done, {len(errs)} errors', *errs[:10], sep='\n', flush=True)
        sys.exit(0)
    jobs = []
    for s in a.splits.split(','):
        ids = sorted(d for d in os.listdir(os.path.join(a.raw, s)) if os.path.isdir(os.path.join(a.raw, s, d)))
        jobs += [(s, i) for i in (ids[:a.limit] if a.limit else ids)]
    print(f'{len(jobs)} samples -> {root} | {a.workers} workers', flush=True)
    t0, done, errs = time.time(), 0, []
    with Pool(a.workers, initializer=init, initargs=(a.raw, a.out)) as pool:
        for k, (s, sid, n, err) in enumerate(pool.imap_unordered(process, jobs, chunksize=4)):
            done += n
            if err:
                errs.append(f'{s}/{sid}: {err}')
                print('ERROR', errs[-1], flush=True)
            if (k + 1) % 500 == 0 or k + 1 == len(jobs):
                print(f'{k + 1}/{len(jobs)} samples, {done} files written ({time.time() - t0:.0f}s)', flush=True)
    counts = {s: {c: len([f for f in os.listdir(os.path.join(root, s, c)) if f.endswith('.npz') and '.' not in f[:-4]])
                  for c in sorted(os.listdir(os.path.join(root, s)))} for s in a.splits.split(',')}
    json.dump({'pose': POSE, 'scale': SCALE, 'counts': counts}, open(os.path.join(root, 'meta.json'), 'w'), indent=1)
    print(json.dumps(counts))
    print(f'done, {len(errs)} errors', flush=True)
