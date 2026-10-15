"""Interactive sampler: sample, coverage-edit, transfer and mesh garments with the final VAE + prior on one GPU.

  python tailorflow/viz/server.py --port 8000 --gpu 0
"""
import os
import sys
import json
import time
import struct
import threading
import argparse

for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_i + 1]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from scipy.spatial import cKDTree
from flask import Flask, request, Response, send_from_directory

from tailorflow.common import EXPS, load_vae, load_prior, sample_prior, decode_occ, inside_pct, edit_latent
from tailorflow.preprocessing.build import occ_raycast
from tailorflow.garment_mesh import mesh_garment, nn_spacing, topo, area, POISSON_PRESETS
from tailorflow.viz.bodymodel import BodyModel

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--port', type=int, default=8000)
p.add_argument('--host', default='127.0.0.1')
p.add_argument('--gpu', type=int, default=0)
args = p.parse_args()

PRIOR_EXP, PRIOR_CKPT = 'prior', 'ckpt_020000.pt'

HERE = os.path.dirname(os.path.abspath(__file__))
dev = torch.device('cuda')
GPU = threading.Lock()

pc = json.load(open(os.path.join(EXPS, PRIOR_EXP, 'config.json')))
vae, vc, vck = load_vae(pc['vae'], dev)
prior = load_prior(PRIOR_EXP, PRIOR_CKPT, vc, dev)
MODEL = {'vae': vae, 'prior': prior, 'cates': vc['cates'].split(','),
         'label': f"{PRIOR_EXP} @{prior['step']} · {pc['vae_exp']} @{vck['step']}"}
CATES = MODEL['cates']
ANY_CLASS = 'Any'
bm = BodyModel(CATES, device='cpu')
FACES = bm.faces.astype(np.uint32)
_cb = vae.denorm_pts(vae.canon(1)[0][0]).cpu().numpy()
MIRROR = cKDTree(_cb).query(_cb * np.array([-1.0, 1.0, 1.0]))[1].astype(np.uint32)
_covered = np.zeros(6890, dtype=bool)
for _o in bm.occ.values():
    _covered |= (_o > 0.5).any(0)
LOCKED = (np.load(os.path.join(os.path.dirname(HERE), 'smpl_part_map.npy')) == 3) & ~_covered
print(f"loaded {MODEL['label']}", flush=True)
print(f"classes {CATES} | body spaces {bm.names}", flush=True)

DEFAULTS = {'cls': 'Dress', 'L': 32768, 'steps': 64, 'solver': 'heun', 'cfg': 1.0, 'preset': 'b3x_lap30', 'n': 4}
S = {'body': None, 'occ': None, 'items': {}, 'next': 0}

def pack(meta, arrays=None):
    blob, meta = b'', dict(meta)
    meta['arrays'] = {}
    for k, a in (arrays or {}).items():
        a = np.ascontiguousarray(a)
        meta['arrays'][k] = {'dtype': str(a.dtype), 'shape': list(a.shape), 'offset': len(blob), 'bytes': int(a.nbytes)}
        blob += a.tobytes()
    head = json.dumps(meta).encode()
    return Response(struct.pack('<I', len(head)) + head + blob, mimetype='application/octet-stream')


def M():
    return MODEL


def body_tensors(b):
    return b['verts'].to(dev), b['normals'].to(dev), torch.as_tensor(b['beta'], dtype=torch.float32, device=dev)[None]


def make_body(space, beta):
    v = bm.verts(np.asarray(beta, dtype=np.float32), int(space))
    return {'space': int(space), 'beta': [float(x) for x in beta], 'verts': v, 'normals': bm.normals(v)}


def random_body(rng, space=None):
    sp = int(rng.integers(bm.k)) if space is None else int(space)
    return make_body(sp, bm.sample_beta(sp, rng))


def body_meta(b):
    return {'space': b['space'], 'space_name': bm.names[b['space']], 'beta': b['beta']}


def signed_mm(x, bv, bn):
    """Signed distance of each point to the nearest body vertex along its normal, in mm (negative = inside)."""
    from pytorch3d.ops import knn_points
    import torch.nn.functional as F
    idx = knn_points(x.float(), bv.float(), K=1).idx.expand(-1, -1, 3)
    return ((x.float() - bv.float().gather(1, idx)) * F.normalize(bn.float().gather(1, idx), dim=-1)).sum(-1) * 1000


def decode(it):
    """Decode the item's latent on its body under its coverage; fills cloud / signed / stats and drops any stale mesh."""
    bv, bn, beta = body_tensors(it['body'])
    use = it['occ'] is not None
    o = torch.as_tensor(it['occ'] if use else np.zeros(6890), dtype=torch.float32, device=dev).view(1, -1, 1)
    vae = MODEL['vae']
    pt, ft = it['lat'][..., :3].contiguous(), it['lat'][..., 3:].contiguous()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        y = vae.decode(it['zg'], pt, ft, bv, bn, decode_occ(o, torch.tensor([use], device=dev)), beta=beta).float()
    it['cloud'] = y
    it['signed'] = signed_mm(y, bv, bn)[0].cpu().numpy().astype(np.float32)
    it['stats'] = {'n': int(y.shape[1]), 'inside': round(inside_pct(y, bv, bn), 3),
                   'spacing_mm': round(nn_spacing(y[0]).mean().item() * 1000, 2)}
    it['mesh'] = None


def new_item(kind, **kw):
    S['next'] += 1
    it = {'key': f"{kind[0]}{S['next']}", 'kind': kind, 'mesh': None, 'parent': None, **kw}
    S['items'][it['key']] = it
    return it


def item_meta(it):
    m = {k: it[k] for k in ('key', 'kind', 'cls', 'seed', 'L', 'parent', 'occ_src', 'stats')}
    m['body'] = body_meta(it['body'])
    m['cond'] = it['occ'] is not None
    m['covered'] = int((it['occ'] > 0.5).sum()) if it['occ'] is not None else None
    if it['mesh'] is not None:
        m['mesh'] = it['mesh']['meta']
    return m


def item_arrays(it, prefix=''):
    a = {prefix + 'points': it['cloud'][0].cpu().numpy().astype(np.float32), prefix + 'signed': it['signed'],
         prefix + 'verts': it['body']['verts'][0].numpy().astype(np.float32)}
    if it['occ'] is not None:
        a[prefix + 'occ'] = it['occ'].astype(np.float32)
    if it['mesh'] is not None:
        a[prefix + 'mverts'], a[prefix + 'mfaces'] = it['mesh']['V'], it['mesh']['F']
    return a


def one(it):
    return pack({'item': item_meta(it)}, item_arrays(it))


def get_item(key):
    it = S['items'].get(key)
    if it is None:
        raise KeyError(f'unknown item {key} - sample again')
    return it


def cls_id_of(cls):
    """Class index for the prior, mapping the unconditional 'Any' class to the null-class embedding."""
    return len(CATES) if cls == ANY_CLASS else CATES.index(cls)


def mesh_item(it, preset):
    t0 = time.time()
    V, F = mesh_garment(it['cloud'][0], it['body']['verts'][0].to(dev), preset=preset)
    torch.cuda.synchronize()
    st = topo(V, F)
    if F.shape[0] == 0:
        alt = 'b3x_lap15' if preset == 'b3x_lap30' else 'b3x_lap30'
        V2, F2 = mesh_garment(it['cloud'][0], it['body']['verts'][0].to(dev), preset=alt)
        torch.cuda.synchronize()
        if F2.shape[0] > 0:
            V, F, preset, st = V2, F2, alt, topo(V2, F2)
    it['mesh'] = {'V': V.cpu().numpy().astype(np.float32), 'F': F.cpu().numpy().astype(np.uint32),
                  'meta': {'verts': int(V.shape[0]), 'faces': int(F.shape[0]), 'loops': int(st['loops']),
                           'comps': int(st['comps']), 'area': round(area(V, F), 3), 'preset': preset,
                           'ms': int((time.time() - t0) * 1000)}}


app = Flask(__name__, static_folder=None)


@app.errorhandler(Exception)
def on_error(e):
    import traceback
    traceback.print_exc()
    return pack({'error': f'{type(e).__name__}: {e}'})


@app.get('/')
def index():
    return send_from_directory(os.path.join(HERE, 'static'), 'index.html')


@app.get('/static/<path:f>')
def static_file(f):
    return send_from_directory(os.path.join(HERE, 'static'), f)


@app.get('/api/init')
def api_init():
    return pack({'classes': CATES, 'any_class': ANY_CLASS, 'presets': list(POISSON_PRESETS), 'spaces': bm.names,
                 'defaults': DEFAULTS, 'levels': [4096, 8192, 16384, 32768], 'nbetas': 10, 'model': MODEL['label'],
                 'body': body_meta(S['body'])},
                {'faces': FACES, 'verts': S['body']['verts'][0].numpy().astype(np.float32), 'occ': S['occ'],
                 'locked': LOCKED.astype(np.uint32), 'mirror': MIRROR})


@app.post('/api/body')
def api_body():
    """Set the working body. With `key`, that item's garment is re-decoded on the new body (same latent)."""
    q = request.get_json()
    rng = np.random.default_rng(q.get('seed'))
    with GPU:
        b = random_body(rng, q.get('space')) if q.get('randomize') else make_body(q.get('space', 0), q['beta'])
        S['body'] = b
        if q.get('key') and q['key'] in S['items']:
            it = S['items'][q['key']]
            it['body'] = b
            decode(it)
            torch.cuda.synchronize()
            return one(it)
    return pack({'body': body_meta(b)}, {'verts': b['verts'][0].numpy().astype(np.float32)})


@app.post('/api/reset')
def api_reset():
    with GPU:
        S['body'] = random_body(np.random.default_rng())
        S['occ'] = np.zeros(6890, dtype=np.float32)
        S['items'] = {}
    return pack({'body': body_meta(S['body'])}, {'verts': S['body']['verts'][0].numpy().astype(np.float32), 'occ': S['occ']})


@app.post('/api/forget')
def api_forget():
    for k in request.get_json().get('keys', []):
        S['items'].pop(k, None)
    return pack({'n': len(S['items'])})


@app.post('/api/occ')
def api_occ():
    """The painted coverage mask. `from_key` seeds it from an item: its conditioning mask, or a raycast of its mesh."""
    q = request.get_json()
    if q.get('clear'):
        S['occ'][:] = 0.0
    elif q.get('from_key'):
        it = get_item(q['from_key'])
        if q.get('source') == 'cond' and it['occ'] is not None:
            S['occ'] = it['occ'].astype(np.float32).copy()
        else:
            with GPU:
                if it['mesh'] is None:
                    mesh_item(it, DEFAULTS['preset'])
            S['occ'] = occ_raycast(it['mesh']['V'], it['mesh']['F'], it['body']['verts'][0].numpy(),
                                   it['body']['normals'][0].numpy()).astype(np.float32)
    elif q.get('random'):
        cls = q['cls']
        if cls == ANY_CLASS or cls not in bm.occ:
            cls = CATES[int(np.random.default_rng().integers(len(CATES)))]
        m = bm.occ[cls]
        S['occ'] = m[np.random.default_rng().integers(len(m))].astype(np.float32)
    else:
        S['occ'][np.asarray(q['idx'], dtype=np.int64)] = float(q.get('value', 1.0))
        S['occ'][LOCKED] = 0.0
    return pack({'covered': int((S['occ'] > 0.5).sum())}, {'occ': S['occ']})


@app.post('/api/sample')
def api_sample():
    """One new garment on the working body. cond: 'none' | 'painted' | 'random' (a train mask of this class)."""
    q = request.get_json()
    L, solver, steps, cfg = int(q['L']), q['solver'], int(q['steps']), float(q['cfg'])
    if L > 4096:
        cfg = 1.0
    cls, cond, seed = q['cls'], q.get('cond', 'none'), int(q['seed'])
    rng = np.random.default_rng(seed)
    any_cls = cls == ANY_CLASS
    cls_id = len(CATES) if any_cls else CATES.index(cls)
    occ, occ_src = None, None
    if cond == 'painted':
        occ, occ_src = S['occ'].copy(), 'painted'
    elif cond == 'random':
        src_cls = cls if not any_cls else CATES[int(rng.integers(len(CATES)))]
        j = int(rng.integers(len(bm.occ[src_cls])))
        occ, occ_src = bm.occ[src_cls][j].astype(np.float32), f'train mask {j}'
    with GPU:
        t0 = time.time()
        if q.get('body'):
            S['body'] = make_body(q['body']['space'], q['body']['beta'])
        m, b = M(), S['body']
        bv, _, _ = body_tensors(b)
        torch.manual_seed(seed)
        o = torch.as_tensor(occ if occ is not None else np.zeros(6890), dtype=torch.float32, device=dev).view(1, -1, 1)
        zg, lat = sample_prior(m['prior'], m['vae'], torch.tensor([cls_id], device=dev), bv, o,
                               torch.tensor([occ is not None], device=dev), steps=steps, cfg=cfg, L=L, solver=solver)
        it = new_item('garment', zg=zg, lat=lat, cls=cls, seed=seed, L=L, occ=occ, occ_src=occ_src, body=b,
                      solver=solver, cfg=cfg, steps=steps)
        decode(it)
        torch.cuda.synchronize()
        it['stats']['ms'] = int((time.time() - t0) * 1000)
    arrays = item_arrays(it)
    return pack({'item': item_meta(it)}, arrays)


@app.post('/api/edit')
def api_edit():
    """Coverage edit from the item's mask (condition, or mesh raycast for a free sample) to the painted mask."""
    q = request.get_json()
    src = get_item(q['key'])
    with GPU:
        if src['occ'] is not None:
            old = src['occ']
        else:
            if src['mesh'] is None:
                mesh_item(src, DEFAULTS['preset'])
            old = occ_raycast(src['mesh']['V'], src['mesh']['F'], src['body']['verts'][0].numpy(),
                              src['body']['normals'][0].numpy()).astype(np.float32)
        bv, bn, beta = body_tensors(src['body'])
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev).view(1, -1, 1)
        lat, info = edit_latent(MODEL['prior'], MODEL['vae'], torch.tensor([cls_id_of(src['cls'])], device=dev), src['zg'],
                                src['lat'], bv, bn, beta, t(old), t(S['occ']), steps=DEFAULTS['steps'], seed=int(src['seed']),
                                occ_w=float(q.get('occ_w', 1.0)))
        it = new_item(src['kind'], **{k: src[k] for k in ('cls', 'seed', 'body')}, L=int(lat.shape[1]), zg=src['zg'], lat=lat,
                      occ=S['occ'].copy(), occ_src='edited')
        it['parent'] = src['key']
        decode(it)
        it['stats'].update(info)
        torch.cuda.synchronize()
    return one(it)


@app.post('/api/transfer')
def api_transfer():
    """The item's latent decoded on n random bodies - the body swap the decoder is built for."""
    q = request.get_json()
    src = get_item(q['key'])
    n, space = int(q.get('n', 4)), q.get('space')
    rng = np.random.default_rng(q.get('seed'))
    out, arrays = [], {}
    with GPU:
        for i in range(n):
            it = new_item('transfer', **{k: src[k] for k in ('zg', 'lat', 'cls', 'seed', 'L', 'occ', 'occ_src')},
                          body=random_body(rng, space))
            it['parent'] = src['key']
            decode(it)
            out.append(item_meta(it))
            arrays.update(item_arrays(it, f'{i}_'))
        torch.cuda.synchronize()
    return pack({'items': out}, arrays)


@app.post('/api/mesh')
def api_mesh():
    q = request.get_json()
    it = get_item(q['key'])
    with GPU:
        mesh_item(it, q.get('preset', DEFAULTS['preset']))
    return one(it)


def warm_mesh():
    """The first Poisson mesh in a process pays ~20 s of CUDA/pytorch3d warm-up; spend it before the user clicks."""
    t0 = time.time()
    with GPU:
        bv = S['body']['verts'][0].to(dev)
        bn = S['body']['normals'][0].to(dev)
        pts = (bv + bn * 0.02)[::2][None]
        mesh_garment(pts[0], bv, preset=DEFAULTS['preset'])
        torch.cuda.synchronize()
    print(f'mesh path warm ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    S['body'] = random_body(np.random.default_rng(0), 0)
    S['occ'] = np.zeros(6890, dtype=np.float32)
    threading.Thread(target=warm_mesh, daemon=True).start()
    print(f'  -> http://{args.host}:{args.port}', flush=True)
    app.run(host=args.host, port=args.port, threaded=True)
