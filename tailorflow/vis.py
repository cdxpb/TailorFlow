import os, numpy as np, torch
from PIL import Image, ImageDraw
from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene
from pytorch3d.renderer import (FoVOrthographicCameras, look_at_view_transform, RasterizationSettings, MeshRenderer, MeshRasterizer,
                                HardPhongShader, PointLights, TexturesVertex, PointsRasterizationSettings, PointsRasterizer,
                                PointsRenderer, AlphaCompositor)

_SMPL_F = {}


def smpl_faces(dev):
    if dev not in _SMPL_F:
        f = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smpl_faces.npy'))
        _SMPL_F[dev] = torch.from_numpy(f.astype(np.int64)).to(dev)
    return _SMPL_F[dev]


def _camera(bounds, back, size, dev):
    lo, hi = bounds
    ctr = ((lo + hi) / 2).tolist()
    half = ((hi - lo).max() / 2 * 1.08).item()
    s = -1.0 if back else 1.0
    aspect = size[1] / size[0]
    R, T = look_at_view_transform(eye=((ctr[0], ctr[1], ctr[2] + 3.0 * s),), at=(tuple(ctr),), up=((0, 1, 0),))
    cam = FoVOrthographicCameras(device=dev, R=R, T=T, min_x=-half * aspect, max_x=half * aspect, min_y=-half, max_y=half,
                                 znear=0.1, zfar=10.0)
    return cam, ctr, s


@torch.no_grad()
def render_mesh(garment, body_v, bounds, back=False, size=(480, 340), body_rgb=None):
    """garment (verts, faces) or None; body_rgb: optional per-vertex body colours [V,3]."""
    dev = body_v.device
    bc = torch.tensor([0.82, 0.80, 0.78], device=dev).expand(body_v.shape[0], 3) if body_rgb is None else body_rgb.to(dev).float()
    meshes = [Meshes([body_v.float()], [smpl_faces(dev)], textures=TexturesVertex([bc]))]
    if garment is not None:
        gv, gf = garment
        meshes.append(Meshes([gv.float()], [gf], textures=TexturesVertex([torch.tensor([0.25, 0.5, 0.85], device=dev).expand(gv.shape[0], 3)])))
    scene = join_meshes_as_scene(meshes)
    cam, ctr, s = _camera(bounds, back, size, dev)
    lights = PointLights(device=dev, location=((ctr[0] + 1.0, ctr[1] + 1.5, ctr[2] + 3.0 * s),),
                         ambient_color=((0.45, 0.45, 0.45),), diffuse_color=((0.6, 0.6, 0.6),), specular_color=((0.05, 0.05, 0.05),))
    rend = MeshRenderer(MeshRasterizer(cameras=cam, raster_settings=RasterizationSettings(image_size=size, blur_radius=0.0, faces_per_pixel=1)),
                        HardPhongShader(device=dev, cameras=cam, lights=lights))
    return (rend(scene)[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def render_points(pts, body_v, bounds, back=False, size=(480, 340), rgb=(0.12, 0.35, 0.75), occ=None):
    dev = pts.device
    cam, _, s = _camera(bounds, back, size, dev)
    g, b = pts.float(), body_v.float().to(dev)
    dz = g[:, 2] * s
    dn = ((dz - dz.min()) / (dz.max() - dz.min()).clamp_min(1e-6)).unsqueeze(-1)
    gc = torch.tensor(rgb, device=dev) * (0.35 + 0.65 * dn)
    bc = torch.tensor([0.85, 0.83, 0.81], device=dev).expand(b.shape[0], 3)
    if occ is not None:
        bc = torch.where(occ[:, :1].to(dev) > 0.5, torch.tensor([0.95, 0.35, 0.35], device=dev), bc)
    radius = min(0.012, 0.012 * (4096 / g.shape[0]) ** 0.5) * 1.5
    rend = PointsRenderer(rasterizer=PointsRasterizer(cameras=cam, raster_settings=PointsRasterizationSettings(
        image_size=size, radius=radius, points_per_pixel=4)), compositor=AlphaCompositor(background_color=(1.0, 1.0, 1.0)))
    img = rend(Pointclouds(points=[torch.cat([g, b])], features=[torch.cat([gc, bc])]))[0, ..., :3]
    return (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def label(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((6, 4), text, fill=(0, 0, 0))
    return np.asarray(im)


def cloud_panel(pc, body, title, rgb=(0.12, 0.35, 0.75), occ=None, size=(320, 240)):
    """Garment + body (occupied vertices red) point render, framed on both; returns HxWx3 uint8."""
    pc, body = pc.detach().float(), body.detach().float().to(pc.device)
    allp = torch.cat([pc, body])
    return label(render_points(pc, body, (allp.min(0).values, allp.max(0).values), size=size, rgb=rgb, occ=occ), title)


def save_grid(rows, path):
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)
