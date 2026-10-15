import math
import torch
import torch.nn as nn
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
from tailorflow.model import mlp, FFBlock, FiLM, VoxelUNet, fourier_enc, SADown, FPUp, _level_vres


def temb(t, d):
    h = d // 2
    f = torch.exp(-math.log(10000) * torch.arange(h, device=t.device) / (h - 1))
    a = t.view(-1, 1) * f.view(1, -1) * 1000.0
    return torch.cat([a.sin(), a.cos()], -1)


class OccEncoder(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.proj = nn.Linear(6890, d)
        self.null = nn.Parameter(torch.zeros(d))

    def forward(self, occ, drop=None):
        if occ is None:
            B = drop.shape[0] if drop is not None else 1
            return self.null.unsqueeze(0).expand(B, -1)
        h = self.proj(occ.view(occ.shape[0], -1))
        if drop is not None:
            h = torch.where(drop.view(-1, 1), self.null, h)
        return h


class FracHead(nn.Module):
    """Fraction of a grown garment that is new cloth (trained with the inpainting rows; not used at inference)."""

    def __init__(self, dg=128, d=256, ncls=5, nvert=6890):
        super().__init__()
        self.occ = nn.Linear(2 * nvert, d)
        self.cls_emb = nn.Embedding(ncls + 1, d)
        self.net = mlp([d + dg, d, d, 1])

    def forward(self, occ_old, occ_new, zgl, cls):
        o, n = (occ_old.view(occ_old.shape[0], -1) > 0.5).float(), (occ_new.view(occ_new.shape[0], -1) > 0.5).float()
        h = torch.relu(self.occ(torch.cat([n, n * (1 - o)], -1)) + self.cls_emb(cls))
        return self.net(torch.cat([h, zgl.float()], -1))[..., 0]


class SEGate(nn.Module):
    def __init__(self, d, r=4):
        super().__init__()
        self.fc = mlp([d, max(d // r, 4), d])

    def forward(self, x):
        return x * torch.sigmoid(self.fc(x))


class GlobalPrior(nn.Module):
    def __init__(self, dg=128, d=256, depth=4, ncls=5, tdim=128, docc=None):
        super().__init__()
        self.dg = dg
        self.tdim = tdim
        self.in_proj = nn.Linear(dg, d)
        self.t_mlp = mlp([tdim, d, d])
        self.cls_emb = nn.Embedding(ncls + 1, d)
        self.occ_proj = nn.Linear(docc, d) if docc else None
        self.blocks = nn.ModuleList([nn.ModuleDict({'film': FiLM(d, d), 'ff': FFBlock(d), 'se': SEGate(d)}) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, dg)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, zgl, t, cls, occ_cond=None):
        h = self.in_proj(zgl).unsqueeze(1)
        c = self.t_mlp(temb(t, self.tdim)) + self.cls_emb(cls)
        if occ_cond is not None and self.occ_proj is not None:
            c = c + self.occ_proj(occ_cond)
        for b in self.blocks:
            h = b['se'](b['ff'](b['film'](h, c)))
        return self.out(self.norm(h).squeeze(1))


class LocalPrior(nn.Module):
    """Per-point latent denoiser; (t, class, coverage) via FiLM, the global latent via AdaGN + FiLM."""

    def __init__(self, dl=1, dg=128, d=256, ncls=5, tdim=128, nfreq=0, vres=12, vbound=4.0, vd=64, k_sa=16, k_fp=3, docc=None,
                 hier_schedule=(2048, 512, 256, 128)):
        super().__init__()
        self.dpt = 3 + dl
        self.dg = dg
        self.nfreq = nfreq
        self.tdim = tdim
        self.in_pt = nn.Linear(self.dpt + 6 * nfreq + 2, d)
        self.known_proj = nn.Linear(1, d, bias=False)
        nn.init.zeros_(self.known_proj.weight)
        self.gl_proj = nn.Linear(dg, d)
        self.t_mlp = mlp([tdim, d, d])
        self.cls_emb = nn.Embedding(ncls + 1, d)
        self.occ_proj = nn.Linear(docc, d) if docc else None
        self.hier_schedule = tuple(hier_schedule)
        self.n_levels = len(self.hier_schedule)
        vres_i = _level_vres(vres, self.n_levels)
        self.stem = VoxelUNet(d, d=vd, levels=1, res=vres_i[0], bound=vbound, d_out=d, cond_dim=d)
        self.downs = nn.ModuleList([SADown(d, k=k_sa, vres=vres_i[i + 1], vbound=vbound, vd=vd, cond_dim=d) for i in range(self.n_levels)])
        self.ups = nn.ModuleList([FPUp(d, k=k_fp) for _ in range(self.n_levels)])
        self.film_down = nn.ModuleList([FiLM(d, d) for _ in range(self.n_levels + 1)])
        self.ff_down = nn.ModuleList([FFBlock(d) for _ in range(self.n_levels + 1)])
        self.film_up = nn.ModuleList([FiLM(d, d) for _ in range(self.n_levels)])
        self.ff_up = nn.ModuleList([FFBlock(d) for _ in range(self.n_levels)])
        self.style_down = nn.ModuleList([FiLM(d, d) for _ in range(self.n_levels + 1)])
        self.style_up = nn.ModuleList([FiLM(d, d) for _ in range(self.n_levels)])
        self.pocc_projs = nn.ModuleList([nn.Linear(2, d) for _ in range(self.n_levels + 1)])
        self.pocc_projs_up = nn.ModuleList([nn.Linear(2, d) for _ in range(self.n_levels)])
        self.norm_pt = nn.LayerNorm(d)
        self.out_pt = nn.Linear(d, self.dpt)
        nn.init.zeros_(self.out_pt.weight)
        nn.init.zeros_(self.out_pt.bias)

    def forward(self, zpt, zgl_cond, t, cls, occ_cond=None, pocc=None, known=None):
        pos0 = zpt[..., :3]
        if pocc is None:
            pocc = torch.zeros_like(pos0[..., :2])
        feats_in = [zpt]
        if self.nfreq > 0:
            feats_in.append(fourier_enc(pos0, self.nfreq)[..., 3:])
        feats_in.append(pocc)
        hp = self.in_pt(torch.cat(feats_in, -1))
        if known is not None:
            kf = known.unsqueeze(-1).to(hp.dtype)
            hp = hp + self.known_proj(kf)
        c = self.t_mlp(temb(t, self.tdim)) + self.cls_emb(cls)
        if occ_cond is not None and self.occ_proj is not None:
            c = c + self.occ_proj(occ_cond)
        s = self.gl_proj(zgl_cond)

        g, l = self.stem(pos0, hp, cond=s)
        hp = self.ff_down[0](self.style_down[0](self.film_down[0](hp + l + g.unsqueeze(1), c), s))
        hp = hp + self.pocc_projs[0](pocc)
        positions, feats, poccs = [pos0], [hp], [pocc]
        cur_pos, cur_feat, cur_pocc = pos0, hp, pocc
        for i, down in enumerate(self.downs):
            cur_pos, cur_feat, _, idx = down(cur_pos, cur_feat, min(self.hier_schedule[i], cur_pos.shape[1]), s)
            cur_pocc = cur_pocc.gather(1, idx.unsqueeze(-1).expand(-1, -1, 2))
            poccs.append(cur_pocc)
            cur_feat = cur_feat + self.pocc_projs[i + 1](cur_pocc)
            cur_feat = self.ff_down[i + 1](self.style_down[i + 1](self.film_down[i + 1](cur_feat, c), s))
            positions.append(cur_pos)
            feats.append(cur_feat)
        for i in reversed(range(self.n_levels)):
            cur_feat = self.ups[i](positions[i], feats[i], positions[i + 1], cur_feat)
            cur_feat = cur_feat + self.pocc_projs_up[i](poccs[i])
            cur_feat = self.ff_up[i](self.style_up[i](self.film_up[i](cur_feat, c), s))
        return self.out_pt(self.norm_pt(cur_feat))


def _gather(x):
    out = torch.empty((dist.get_world_size() * x.shape[0],) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x.contiguous())
    return out


def ot_pair(z, n, cls):
    """Minibatch OT coupling of data and noise per class, over the global batch under DDP."""
    B, r0 = z.shape[0], 0
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        r0 = dist.get_rank() * B
        z, n, cls = _gather(z.detach()), _gather(n), _gather(cls)
    zf, nf = z.flatten(1).float(), n.flatten(1).float()
    perm = torch.arange(z.shape[0], device=z.device)
    for c in cls.unique():
        idx = (cls == c).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        _, col = linear_sum_assignment(torch.cdist(zf[idx], nf[idx]).detach().cpu().numpy())
        perm[idx] = idx[col]
    return n[perm[r0:r0 + B]]


def sample_t(B, dev):
    return torch.sigmoid(torch.randn(B, device=dev))


class GlobalFlow:
    def __init__(self, prior, drop=0.1, ncls=5):
        self.p, self.drop, self.ncls = prior, drop, ncls

    def loss(self, zgl, cls, occ_cond=None):
        B = zgl.shape[0]
        t = sample_t(B, zgl.device).clamp(1e-3, 1 - 1e-3)
        ngl = ot_pair(zgl, torch.randn_like(zgl), cls)
        tg = t.view(B, 1)
        xgl = (1 - tg) * ngl + tg * zgl
        c = torch.where(torch.rand(B, device=zgl.device) < self.drop, torch.full_like(cls, self.ncls), cls)
        x_theta = self.p(xgl, t, c, occ_cond=occ_cond)
        v_theta = (x_theta - xgl) / (1 - tg).clamp(min=0.05)
        return (v_theta - (zgl - ngl)).pow(2).mean()

    @torch.no_grad()
    def sample(self, B, dev, cls, steps=64, cfg=1.0, occ_cond=None, solver='euler'):
        zgl = torch.randn(B, self.p.dg, device=dev)
        un = torch.full_like(cls, self.ncls)

        def vel(z, tt):
            t = torch.full((B,), tt, device=dev)
            tg = t.view(B, 1)
            v = (self.p(z, t, cls, occ_cond=occ_cond) - z) / (1 - tg)
            if cfg != 1.0:
                u = (self.p(z, t, un, occ_cond=occ_cond) - z) / (1 - tg)
                v = u + cfg * (v - u)
            return v

        for i in range(steps):
            v = vel(zgl, i / steps)
            if solver == 'heun' and i < steps - 1:
                v = (v + vel(zgl + v / steps, (i + 1) / steps)) / 2
            zgl = zgl + v / steps
        return zgl


class LocalFlow:
    def __init__(self, prior, drop=0.1, ncls=5, t_hi_p=0.0, denom_min=0.05):
        self.p, self.drop, self.ncls = prior, drop, ncls
        self.t_hi_p, self.denom_min = t_hi_p, denom_min

    def loss(self, zpt, zgl_cond, cls, occ_cond=None, pocc_fn=None, pocc_drop=None, known=None):
        """x-prediction loss; known [B, N] rows (inpainting) are held clean and excluded from the loss."""
        B = zpt.shape[0]
        t = sample_t(B, zpt.device)
        if self.t_hi_p > 0:
            hi = torch.rand(B, device=zpt.device) < self.t_hi_p
            t = torch.where(hi, 0.9 + 0.1 * torch.rand(B, device=zpt.device), t)
        t = t.clamp(1e-3, 1 - 1e-3)
        npt = ot_pair(zpt, torch.randn_like(zpt), cls)
        tp = t.view(B, 1, 1)
        xpt = (1 - tp) * npt + tp * zpt
        if known is not None:
            xpt = torch.where(known.unsqueeze(-1), zpt, xpt)
        pocc = None
        if pocc_fn is not None:
            pocc = pocc_fn(xpt)
            if pocc_drop is not None:
                pocc = torch.where(pocc_drop.view(-1, 1, 1), torch.zeros_like(pocc), pocc)
        c = torch.where(torch.rand(B, device=zpt.device) < self.drop, torch.full_like(cls, self.ncls), cls)
        x_theta = self.p(xpt, zgl_cond, t, c, occ_cond=occ_cond, pocc=pocc, known=known)
        ept = ((x_theta - zpt) / (1 - tp).clamp(min=self.denom_min)).pow(2)
        if known is None:
            self.last_parts = (ept[..., :3].mean().item(), ept[..., 3:].mean().item() if ept.shape[-1] > 3 else 0.0)
            return ept.mean()
        w = (~known).unsqueeze(-1).float()
        nw = w.sum().clamp(min=1)
        self.last_parts = ((ept[..., :3] * w).sum().item() / nw.item() / 3, (ept[..., 3:] * w).sum().item() / nw.item() / max(ept.shape[-1] - 3, 1))
        return (ept * w).sum() / (nw * ept.shape[-1])

    @torch.no_grad()
    def sample(self, B, L, dev, cls, zgl_cond, steps=64, cfg=1.0, occ_cond=None, pocc_fn=None, solver='euler'):
        zpt = torch.randn(B, L, self.p.dpt, device=dev)
        un = torch.full_like(cls, self.ncls)

        def vel(z, tt):
            t = torch.full((B,), tt, device=dev)
            tp = t.view(B, 1, 1)
            pocc = pocc_fn(z) if pocc_fn is not None else None
            v = (self.p(z, zgl_cond, t, cls, occ_cond=occ_cond, pocc=pocc) - z) / (1 - tp)
            if cfg != 1.0:
                u = (self.p(z, zgl_cond, t, un, occ_cond=occ_cond, pocc=pocc) - z) / (1 - tp)
                v = u + cfg * (v - u)
            return v

        for i in range(steps):
            v = vel(zpt, i / steps)
            if solver == 'heun' and i < steps - 1:
                v = (v + vel(zpt + v / steps, (i + 1) / steps)) / 2
            zpt = zpt + v / steps
        return zpt
