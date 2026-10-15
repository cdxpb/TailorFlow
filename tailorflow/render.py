"""Path-traced figure renders (Mitsuba 3): body + garment on a white floor with a soft ground shadow. Every cell is lit
on its own (an area key light above and in front, a constant sky fill); the floor is a shadow catcher, so only the
shadow is kept, and it fades out before the cell edge.

  python tailorflow/render.py a.npz b.npz ... --color '#FF9EB5' --out row.png [--points]
an npz holds `bverts` plus `mV`/`mF` (garment mesh) or `cloud` (garment points), as stored for the paper figures."""
import os, argparse, tempfile
import numpy as np
import trimesh
import mitsuba as mi
from PIL import Image

mi.set_variant('cuda_ad_rgb' if 'cuda_ad_rgb' in mi.variants() else 'llvm_ad_rgb')
T = mi.ScalarTransform4f
FACES = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smpl_faces.npy'))
BODY = (0.88, 0.88, 0.90)
LIGHT = dict(pos=(0.948, 3.546, 2.603), size=0.8, key=14.42, sky=0.3)
CLOTH = None
TMP = tempfile.mkdtemp()
lin = lambda c: [((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c]
rgb = lambda c: [int(c.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)] if isinstance(c, str) else list(c)


def scene(objs, cam, size, floor_y, c, spp, light, floor=True, rfilter='gaussian'):
    pos = [float(c[0] + light['pos'][0]), float(floor_y + light['pos'][1]), float(c[2] + light['pos'][2])]
    d = {'type': 'scene', 'integrator': {'type': 'path', 'max_depth': 6},
         'sensor': {'type': 'perspective', 'fov': cam['fov'], 'fov_axis': 'y',
                    'to_world': T().look_at(origin=cam['origin'], target=cam['target'], up=[0, 1, 0]),
                    'sampler': {'type': 'independent', 'sample_count': spp},
                    'film': {'type': 'hdrfilm', 'width': size[1], 'height': size[0], 'rfilter': {'type': rfilter}}},
         'sky': {'type': 'constant', 'radiance': {'type': 'rgb', 'value': light['sky']}},
         'key': {'type': 'rectangle', 'to_world': T().look_at(origin=pos, target=[float(c[0]), float(floor_y), float(c[2])],
                                                               up=[0, 1, 0]).scale(light['size']),
                 'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': light['key']}}}}
    for i, (V, F, col, two) in enumerate(objs):
        f = os.path.join(TMP, f'o{i}.ply')
        trimesh.Trimesh(np.asarray(V, np.float64), np.asarray(F, np.int64), process=False).export(f)
        b = {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': lin(rgb(col))}}
        if two and CLOTH:
            b = dict(CLOTH, type='principled', base_color={'type': 'rgb', 'value': lin(rgb(col))})
        d[f'o{i}'] = {'type': 'ply', 'filename': f, 'bsdf': {'type': 'twosided', 'bsdf': b} if two else b}
    if floor:
        d['floor'] = {'type': 'rectangle', 'to_world': T().translate([float(c[0]), floor_y, float(c[2])]).rotate([1, 0, 0], -90).scale(20),
                      'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': 0.8}}}
    return d


def aov(d, name, size, ss=1, spp=4):
    d = dict(d, integrator={'type': 'aov', 'aovs': name})
    d['sensor'] = dict(d['sensor'], sampler={'type': 'independent', 'sample_count': spp},
                       film={'type': 'hdrfilm', 'width': size[1] * ss, 'height': size[0] * ss, 'rfilter': {'type': 'box'}})
    return mi.render(mi.load_dict(d)).numpy()


def render(objs, lo, hi, H, floor_y, spp=1024, elev=9.0, fov=14.0, light=LIGHT, exposure=0.85, fade=(0.55, 1.0)):
    """objs: [(V, F, colour, twosided)]; lo/hi: world box framed by the cell (height H px) -> HxWx3 uint8."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    size = (H, int(round(H * (hi[0] - lo[0]) / (hi[1] - lo[1]))))
    c, e = (lo + hi) / 2, np.radians(elev)
    dist = (hi[1] - lo[1]) / 2 / np.tan(np.radians(fov / 2))
    cam = {'fov': fov, 'target': [float(x) for x in c], 'origin': [float(c[0]), float(c[1] + dist * np.sin(e)), float(c[2] + dist * np.cos(e))]}
    full = mi.render(mi.load_dict(scene(objs, cam, size, floor_y, c, spp, light))).numpy()[..., :3]
    bare = mi.render(mi.load_dict(scene([], cam, size, floor_y, c, spp, light))).numpy()[..., :3]
    a = aov(scene(objs, cam, size, floor_y, c, 1, light, floor=False), 'd:depth', size, ss=4)[..., -1] > 0
    a = a.reshape(size[0], 4, size[1], 4).mean((1, 3))[..., None]
    shade = np.clip(full.mean(-1, keepdims=True) / np.maximum(bare.mean(-1, keepdims=True), 1e-6), 0, 1)
    P = aov(scene([], cam, size, floor_y, c, 1, light), 'p:position', size)[..., -3:]
    w = np.clip((fade[1] - np.linalg.norm(P[..., [0, 2]] - c[[0, 2]], axis=-1, keepdims=True)) / (fade[1] - fade[0]), 0, 1)
    shade = 1 - (1 - shade) * w * w * (3 - 2 * w)
    x = np.clip(a * full * exposure + (1 - a) * shade, 0, None)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


_ICO = trimesh.creation.icosphere(subdivisions=1)


def objects(st, color, points=False, body=BODY, dy=0.0, radius=None):
    sh = np.array([0, dy, 0], np.float32)
    o = [(st['bverts'] - sh, FACES, body, False)]
    if points:
        P = np.asarray(st['cloud'], np.float32) - sh
        r = (radius or 0.0055) * (32768 / len(P)) ** 0.5
        V = (P[:, None] + _ICO.vertices[None].astype(np.float32) * r).reshape(-1, 3)
        F = (_ICO.faces[None] + (np.arange(len(P)) * len(_ICO.vertices))[:, None, None]).reshape(-1, 3)
        return o + [(V, F, color, True)]
    return o + ([(st['mV'] - sh, st['mF'], color, True)] if len(st['mV']) else [])


def row(states, colors, H=640, points=False, gap=10, body=BODY, **kw):
    """States side by side, one shared frame and scale, feet on one floor; returns HxWx3 uint8."""
    feet = [float(s['bverts'][:, 1].min()) for s in states]
    floor_y = min(feet)
    objs = [objects(s, c, points, body, f - floor_y) for s, c, f in zip(states, colors, feet)]
    pts = np.concatenate([V for o in objs for V, *_ in o])
    lo, hi = pts.min(0), pts.max(0)
    wy = hi[1] - lo[1]
    lo, hi = lo - [0.12 * wy, 0.04 * wy, 0], hi + [0.04 * wy, 0.015 * wy, 0]
    cells = [render(o, lo, hi, H, floor_y, **kw) for o in objs]
    sep = np.full((H, gap, 3), 255, np.uint8)
    return np.concatenate(sum([[c, sep] for c in cells], [])[:-1], 1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('npz', nargs='+')
    p.add_argument('--color', default='#FF9EB5')
    p.add_argument('--points', action='store_true')
    p.add_argument('--H', type=int, default=640)
    p.add_argument('--spp', type=int, default=1024)
    p.add_argument('--out', default='row.png')
    a = p.parse_args()
    S = [dict(np.load(f)) for f in a.npz]
    Image.fromarray(row(S, [a.color] * len(S), a.H, a.points, spp=a.spp)).save(a.out)
