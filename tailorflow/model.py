import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points, knn_points
from tailorflow.metrics import fps


def rot_a_to_b(a, b, v):
    c = (a * b).sum(-1, keepdim=True)
    k = torch.cross(a, b, dim=-1)
    den = (1.0 + c).clamp_min(1e-6)
    out = v * c + torch.cross(k, v, dim=-1) + k * ((k * v).sum(-1, keepdim=True) / den)
    return torch.where(c < -0.999, v, out)


def reanchor_pts(p, s, sn, t, tn, k, sigma):
    """Move points from body s to body t, keeping each point's offset to its k nearest vertices."""
    with torch.autocast('cuda', enabled=False):
        p32, s, sn, t, tn = p.float(), s.float(), sn.float(), t.float(), tn.float()
        r = knn_points(p32, s, K=k)
        g = r.idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        take = lambda v: torch.gather(v.unsqueeze(1).expand(-1, p32.shape[1], -1, -1), 2, g)
        sp, snn, tp, tnn = take(s), take(sn), take(t), take(tn)
        off = p32.unsqueeze(2) - sp
        moved = (tp - sp) + (rot_a_to_b(snn, tnn, off) - off)
        d = r.dists.clamp_min(0).sqrt()
        w = torch.softmax(-(d - d[..., :1]) / sigma, dim=-1)
        delta = (w.unsqueeze(-1) * moved).sum(2)
    return p + delta.to(p.dtype)


def fourier_enc(x, n=6):
    f = (2.0 ** torch.arange(n, device=x.device, dtype=x.dtype)) * math.pi
    a = x.unsqueeze(-1) * f
    return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)


def mlp(dims, act=nn.ReLU):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class FiLM(nn.Module):
    def __init__(self, d, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(cond_dim, 2 * d)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class CrossAttnBlock(nn.Module):
    def __init__(self, d, kv_dim, heads=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, kdim=kv_dim, vdim=kv_dim, batch_first=True)

    def forward(self, q, kv):
        out, _ = self.attn(self.norm_q(q), kv, kv, need_weights=False)
        return q + out


class FFBlock(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net = mlp([d, d * mult, d], act=nn.GELU)

    def forward(self, x):
        return x + self.net(self.norm(x))


class _Gather1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat, idx):
        ctx.save_for_backward(idx)
        ctx.shape, ctx.dtype = feat.shape, feat.dtype
        return feat.gather(1, idx)

    @staticmethod
    def backward(ctx, g):
        idx, = ctx.saved_tensors
        out = torch.zeros(ctx.shape, device=g.device, dtype=torch.float32)
        out.scatter_add_(1, idx, g.float())
        return out.to(ctx.dtype), None


def gather_rows(feat, idx):
    """feat [B, N, C], idx [B, M] -> [B, M, C]; backward accumulates in float32 (bf16 scatter atomics are slow)."""
    return _Gather1.apply(feat, idx.unsqueeze(-1).expand(-1, -1, feat.shape[-1]))


def gather_nbrs(feat, idx):
    B, Nq, k = idx.shape
    return gather_rows(feat, idx.reshape(B, Nq * k)).reshape(B, Nq, k, feat.shape[-1])


def gather_idx(feat, idx):
    return gather_rows(feat, idx.view(1, -1).expand(feat.shape[0], -1))


def gather_wc(feat, idx, w):
    return (gather_nbrs(feat, idx) * w.unsqueeze(-1)).sum(2)


def knn_w(q, ref, k=8, tau=0.2):
    res = knn_points(q, ref, K=k)
    return torch.softmax(-torch.sqrt(res.dists) / tau, dim=-1), res.idx


def voxelize(pos, feat, res, bound):
    B, L, C = feat.shape
    idx = ((pos.detach().clamp(-bound, bound) + bound) / (2 * bound) * (res - 1e-3)).long().clamp(0, res - 1)
    flat = idx[..., 0] * res * res + idx[..., 1] * res + idx[..., 2]
    grid = torch.zeros(B, res ** 3, C, device=feat.device, dtype=torch.float32)
    cnt = torch.zeros(B, res ** 3, 1, device=feat.device, dtype=torch.float32)
    grid = grid.scatter_add(1, flat.unsqueeze(-1).expand(-1, -1, C), feat.float())
    cnt.scatter_add_(1, flat.unsqueeze(-1), torch.ones_like(flat, dtype=torch.float32).unsqueeze(-1))
    grid = (grid / cnt.clamp(min=1)).to(feat.dtype)
    return grid.transpose(1, 2).view(B, C, res, res, res), flat


class _Devoxelize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, flat):
        B, C = grid.shape[:2]
        ctx.save_for_backward(flat)
        ctx.shape, ctx.dtype = grid.shape, grid.dtype
        return grid.reshape(B, C, -1).gather(2, flat.unsqueeze(1).expand(-1, C, -1)).transpose(1, 2)

    @staticmethod
    def backward(ctx, g):
        flat, = ctx.saved_tensors
        B, C = ctx.shape[:2]
        out = torch.zeros(B, ctx.shape[2] * ctx.shape[3] * ctx.shape[4], C, device=g.device, dtype=torch.float32)
        out.scatter_add_(1, flat.unsqueeze(-1).expand(-1, -1, C), g.float())
        return out.transpose(1, 2).reshape(ctx.shape).to(ctx.dtype), None


def devoxelize(grid, flat):
    return _Devoxelize.apply(grid, flat)


class ResBlock3D(nn.Module):
    def __init__(self, d, cond_dim=None, groups=8):
        super().__init__()
        g = groups if d % groups == 0 else 1
        self.norm1 = nn.GroupNorm(g, d)
        self.conv1 = nn.Conv3d(d, d, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, d)
        self.conv2 = nn.Conv3d(d, d, 3, padding=1)
        self.act = nn.GELU()
        self.cond_proj = nn.Linear(cond_dim, 2 * d) if cond_dim else None
        if self.cond_proj is not None:
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x, cond=None):
        h = self.conv1(self.act(self.norm1(x)))
        if self.cond_proj is not None and cond is not None:
            gamma, beta = self.cond_proj(cond).chunk(2, dim=-1)
            h = h * (1 + gamma.view(*gamma.shape, 1, 1, 1)) + beta.view(*beta.shape, 1, 1, 1)
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class SEGate3D(nn.Module):
    def __init__(self, d, r=8):
        super().__init__()
        self.fc = mlp([d, max(d // r, 4), d])

    def forward(self, x):
        gate = torch.sigmoid(self.fc(x.mean(dim=(2, 3, 4))))
        return x * gate.view(*gate.shape, 1, 1, 1)


class VoxelUNet(nn.Module):
    def __init__(self, d_in, d=64, levels=2, res=16, bound=0.6, d_out=None, cond_dim=None, use_se=False, global_only=False):
        super().__init__()
        self.res = res
        self.bound = bound
        self.cond_dim = cond_dim
        self.use_se = use_se
        self.global_only = global_only
        d_out = d_in if d_out is None else d_out
        self.in_proj = nn.Linear(d_in, d)
        self.down = nn.ModuleList([nn.Conv3d(d, d, 3, stride=2, padding=1) for _ in range(levels)])
        self.down_res = nn.ModuleList([ResBlock3D(d, cond_dim) for _ in range(levels)])
        self.bottleneck = ResBlock3D(d, cond_dim)
        if not global_only:
            self.up = nn.ModuleList([nn.ConvTranspose3d(d, d, 4, stride=2, padding=1) for _ in range(levels)])
            self.up_res = nn.ModuleList([ResBlock3D(d, cond_dim) for _ in range(levels)])
            self.fuse = nn.ModuleList([nn.Conv3d(2 * d, d, 3, padding=1) for _ in range(levels)])
        if use_se:
            self.down_se = nn.ModuleList([SEGate3D(d) for _ in range(levels)])
            self.bottleneck_se = SEGate3D(d)
            if not global_only:
                self.up_se = nn.ModuleList([SEGate3D(d) for _ in range(levels)])
        self.global_proj = nn.Linear(d, d_out)
        self.local_proj = None if global_only else nn.Linear(d, d_out)

    def forward(self, pos, feat, cond=None):
        c = cond if self.cond_dim else None
        x, idx = voxelize(pos, self.in_proj(feat), self.res, self.bound)
        skips = []
        for i, down in enumerate(self.down):
            skips.append(x)
            x = self.down_res[i](down(x), c)
            if self.use_se:
                x = self.down_se[i](x)
        x = self.bottleneck(x, c)
        if self.use_se:
            x = self.bottleneck_se(x)
        g = self.global_proj(x.amax(dim=(2, 3, 4)))
        if self.global_only:
            return g, None
        for i, (up, fuse, skip) in enumerate(zip(self.up, self.fuse, reversed(skips))):
            x = self.up_res[i](up(x), c)
            if self.use_se:
                x = self.up_se[i](x)
            x = fuse(torch.cat([x, skip], dim=1))
        return g, self.local_proj(devoxelize(x, idx))


class SADown(nn.Module):
    def __init__(self, d, k=16, vres=8, vbound=4.0, vd=64, cond_dim=None, use_se=False):
        super().__init__()
        self.k = k
        self.agg = mlp([2 * d, d, d])
        self.unet = VoxelUNet(d, d=vd, levels=1, res=vres, bound=vbound, d_out=d, cond_dim=cond_dim, use_se=use_se)

    def forward(self, pos, feat, n_out, cond):
        new_pos, idx = fps(pos.detach(), min(n_out, pos.shape[1]))
        nbr = gather_nbrs(feat, knn_points(new_pos, pos.detach(), K=min(self.k, pos.shape[1])).idx)
        center = gather_rows(feat, idx)
        h = self.agg(torch.cat([center.unsqueeze(2).expand_as(nbr), nbr - center.unsqueeze(2)], -1)).amax(2)
        g, l = self.unet(new_pos, h, cond=cond)
        return new_pos, h + l + g.unsqueeze(1), g, idx


class FPUp(nn.Module):
    def __init__(self, d, k=3, tau=0.85):
        super().__init__()
        self.k = k
        self.tau = tau
        self.fuse = mlp([2 * d, d, d])

    def forward(self, pos_fine, feat_skip, pos_coarse, feat_coarse):
        r = knn_points(pos_fine.detach().float(), pos_coarse.detach().float(), K=min(self.k, pos_coarse.shape[1]))
        dk, idx = r.dists.clamp(min=0).sqrt(), r.idx
        w = torch.softmax(-dk / self.tau, dim=-1)
        up = (gather_nbrs(feat_coarse, idx) * w.unsqueeze(-1)).sum(2)
        return self.fuse(torch.cat([up, feat_skip], -1))


def _level_vres(vres, n_levels):
    return [max(4, 2 * round(vres * (4 ** (-i / n_levels)) / 2)) for i in range(n_levels + 1)]


class HierEncoder(nn.Module):
    def __init__(self, d_in, d=256, d_out=None, cond_dim=None, vres=12, vbound=0.6, hier_schedule=(1024, 256, 64),
                 hier_vd=None, k_sa=16, k_fp=3, use_se=False, use_global=True):
        super().__init__()
        d_out = d_in if d_out is None else d_out
        self.hier_schedule = tuple(hier_schedule)
        self.n_levels = len(self.hier_schedule)
        vres_i = _level_vres(vres, self.n_levels)
        self.in_proj = nn.Linear(d_in, d)
        self.stem = VoxelUNet(d, d=hier_vd[0], levels=1, res=vres_i[0], bound=vbound, d_out=d, cond_dim=cond_dim, use_se=use_se)
        self.downs = nn.ModuleList([SADown(d, k=k_sa, vres=vres_i[i + 1], vbound=vbound, vd=hier_vd[i + 1], cond_dim=cond_dim,
                                          use_se=use_se) for i in range(self.n_levels)])
        self.ups = nn.ModuleList([FPUp(d, k=k_fp) for _ in range(self.n_levels)])
        self.global_proj = nn.Linear(d, d_out) if use_global else None
        self.local_proj = nn.Linear(d, d_out)

    def forward(self, pos, feat, cond=None):
        g, l = self.stem(pos, self.in_proj(feat), cond=cond)
        l = l + g.unsqueeze(1)
        positions, feats = [pos], [l]
        cur_pos, cur_feat = pos, l
        for i, down in enumerate(self.downs):
            cur_pos, cur_feat, g, _ = down(cur_pos, cur_feat, self.hier_schedule[i], cond)
            positions.append(cur_pos)
            feats.append(cur_feat)
        for i in reversed(range(self.n_levels)):
            cur_feat = self.ups[i](positions[i], feats[i], positions[i + 1], cur_feat)
        g = self.global_proj(g) if self.global_proj is not None else None
        return g, self.local_proj(cur_feat), positions, feats


class BodyFuse(nn.Module):
    def __init__(self, d, k=8, tau=0.85):
        super().__init__()
        self.k = k
        self.tau = tau
        self.fuse = mlp([2 * d + 3, d, d])
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, pos, feat, body_level):
        bpos, bfeat = body_level
        r = knn_points(pos.detach().float(), bpos.detach().float(), K=min(self.k, bpos.shape[1]))
        dk, idx = r.dists.clamp(min=0).sqrt(), r.idx
        w = torch.softmax(-dk / self.tau, dim=-1)
        up = (gather_nbrs(bfeat, idx) * w.unsqueeze(-1)).sum(2)
        bp = (gather_nbrs(bpos, idx) * w.unsqueeze(-1)).sum(2)
        return feat + self.fuse(torch.cat([feat, up, pos - bp], -1))


class HierDecoderCore(nn.Module):
    def __init__(self, d, cond_dim, body_kv_dim, heads=4, vres=12, vbound=0.6, hier_schedule=(1024, 256, 64), hier_vd=None,
                 k_sa=16, k_fp=3, use_se=False):
        super().__init__()
        self.hier_schedule = tuple(hier_schedule)
        self.n_levels = len(self.hier_schedule)
        n = self.n_levels + 1
        vres_i = _level_vres(vres, self.n_levels)
        self.stem = VoxelUNet(d, d=hier_vd[0], levels=1, res=vres_i[0], bound=vbound, d_out=d, cond_dim=cond_dim, use_se=use_se)
        self.downs = nn.ModuleList([SADown(d, k=k_sa, vres=vres_i[i + 1], vbound=vbound, vd=hier_vd[i + 1], cond_dim=cond_dim,
                                          use_se=use_se) for i in range(self.n_levels)])
        self.ups = nn.ModuleList([FPUp(d, k=k_fp) for _ in range(self.n_levels)])
        self.cross_down = nn.ModuleList([CrossAttnBlock(d, body_kv_dim, heads) for _ in range(n)])
        self.cross_up = nn.ModuleList([CrossAttnBlock(d, body_kv_dim, heads) for _ in range(self.n_levels)])
        self.film_down = nn.ModuleList([FiLM(d, cond_dim) for _ in range(n)])
        self.ff_down = nn.ModuleList([FFBlock(d) for _ in range(n)])
        self.film_up = nn.ModuleList([FiLM(d, cond_dim) for _ in range(self.n_levels)])
        self.ff_up = nn.ModuleList([FFBlock(d) for _ in range(self.n_levels)])
        self.body_fuse_down = nn.ModuleList([BodyFuse(d) for _ in range(n)])
        self.body_fuse_up = nn.ModuleList([BodyFuse(d) for _ in range(self.n_levels)])

    def forward(self, pos, x, cond, body_kv, body_levels, fuse_pos):
        def blk(i, feat, up=False):
            feat = (self.cross_up if up else self.cross_down)[i](feat, body_kv)
            feat = (self.film_up if up else self.film_down)[i](feat, cond)
            return (self.ff_up if up else self.ff_down)[i](feat)

        g, l = self.stem(pos, x, cond=cond)
        x = blk(0, self.body_fuse_down[0](fuse_pos, x + l + g.unsqueeze(1), body_levels[0]))
        positions, feats = [pos], [x]
        cur_pos, cur_feat = pos, x
        for i, down in enumerate(self.downs):
            cur_pos, cur_feat, _, _ = down(cur_pos, cur_feat, self.hier_schedule[i], cond)
            cur_feat = blk(i + 1, self.body_fuse_down[i + 1](cur_pos, cur_feat, body_levels[i + 1]))
            positions.append(cur_pos)
            feats.append(cur_feat)
        for i in reversed(range(self.n_levels)):
            cur_feat = self.ups[i](positions[i], feats[i], positions[i + 1], cur_feat)
            cur_feat = blk(i, self.body_fuse_up[i](positions[i], cur_feat, body_levels[i]), up=True)
        return cur_feat


class ClothEncoder(nn.Module):
    def __init__(self, d=256, L=4096, dg=128, dl=1, vres=32, vbound=5.0, vlevels=2, vd=64, kl_w_pt=1.0, kl_w_feat=1.0,
                 kl_w_glb=1.0, skip_weight=0.01, log_sigma_offset=6.0, lv_max=0.0):
        super().__init__()
        self.L = L
        self.dl = dl
        self.kl_w_pt = kl_w_pt
        self.kl_w_feat = kl_w_feat
        self.kl_w_glb = kl_w_glb
        self.skip_weight = skip_weight
        self.log_sigma_offset = log_sigma_offset
        self.lv_max = lv_max
        self.global_backbone = VoxelUNet(3, d=vd, levels=vlevels, res=vres, bound=vbound, d_out=d, global_only=True)
        self.global_head = mlp([d, d, 2 * dg])
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)
        nn.init.normal_(self.global_head[-1].weight[:dg], std=0.02)
        nn.init.constant_(self.global_head[-1].bias[dg:], -8.0)
        self.backbone = VoxelUNet(3, d=vd, levels=vlevels, res=vres, bound=vbound, d_out=d, cond_dim=dg, use_se=True)
        for m in self.backbone.modules():
            if isinstance(m, ResBlock3D) and m.cond_proj is not None:
                nn.init.normal_(m.cond_proj.weight, std=0.02)
        self.local_film = FiLM(d, dg)
        self.local_head = mlp([d, d, 2 * (dl + 3)])
        nn.init.zeros_(self.local_head[-1].weight)
        nn.init.zeros_(self.local_head[-1].bias)

    def forward(self, x, sample=None, pt_lv=None):
        sample = self.training if sample is None else sample
        gmu, glv_raw = self.global_head(self.global_backbone(x, x)[0]).chunk(2, -1)
        glv = glv_raw.clamp(-8, self.lv_max)
        zg = gmu + torch.randn_like(gmu) * (0.5 * glv).exp() if sample else gmu
        gl, h = self.backbone(x, x, cond=zg)[:2]
        h = h + gl.unsqueeze(1)
        zmu, zlv = self.local_head(self.local_film(h, zg)).chunk(2, -1)
        zmu = torch.cat([self.skip_weight * zmu[..., :3] + x, zmu[..., 3:]], -1)
        zlv_raw = zlv - 2.0 * self.log_sigma_offset
        if pt_lv is not None:
            zlv_raw = torch.cat([torch.full_like(zlv_raw[..., :3], pt_lv), zlv_raw[..., 3:]], -1)
        zlv = zlv_raw.clamp(-28, self.lv_max)
        z = zmu + torch.randn_like(zmu) * (0.5 * zlv).exp() if sample else zmu
        pt, feat = z.split([3, self.dl], -1)
        kz = 0.5 * (zmu.pow(2) + zlv.exp() - zlv - 1) + torch.relu(zlv_raw - self.lv_max).pow(2)
        kpt, kfeat = kz.split([3, self.dl], -1)
        kg = 0.5 * (gmu.pow(2) + glv.exp() - glv - 1) + torch.relu(glv_raw - self.lv_max).pow(2)
        kl = kpt.mean((1, 2)) * self.kl_w_pt + kg.mean(1) * self.kl_w_glb
        if self.dl:
            kl = kl + kfeat.mean((1, 2)) * self.kl_w_feat
        return zg, pt, feat, kl


class BodyTower(nn.Module):
    def __init__(self, d=256, db=64, dbeta=0, hier_schedule=(2048, 512, 256, 128), hier_vd=None, vres=32, vbound=5.0):
        super().__init__()
        self.backbone = HierEncoder(6, d=d, d_out=d, vres=vres, vbound=vbound, hier_schedule=hier_schedule, hier_vd=hier_vd,
                                    use_se=True)
        self.z_head = mlp([d, db])
        self.pos_proj = nn.Linear(3, d)
        self.dbeta = dbeta
        if dbeta > 0:
            self.beta_proj = mlp([dbeta, db, db])
            nn.init.zeros_(self.beta_proj[-1].weight)
            nn.init.zeros_(self.beta_proj[-1].bias)

    def forward(self, bv, bn, beta=None):
        g, l, bpos, bfeat = self.backbone(bv, torch.cat([bv, bn], dim=-1))
        z_body = self.z_head(g)
        if self.dbeta > 0:
            if beta is None:
                beta = torch.zeros(bv.shape[0], self.dbeta, device=bv.device, dtype=bv.dtype)
            z_body = z_body + self.beta_proj(beta)
        return z_body, l + self.pos_proj(bv), [(p, f + self.pos_proj(p)) for p, f in zip(bpos, bfeat)]


class Decoder(nn.Module):
    def __init__(self, d=256, dl=1, cond_dim=192, heads=4, vres=32, vbound=5.0, vd=64, hier_schedule=(2048, 512, 256, 128),
                 hier_vd=None):
        super().__init__()
        self.in_proj = mlp([3 + dl + 2, d, d])
        self.cond_proj = mlp([cond_dim, cond_dim])
        self.hier = HierDecoderCore(d, cond_dim, d, heads=heads, vres=vres, vbound=vbound, hier_schedule=hier_schedule,
                                    hier_vd=hier_vd, use_se=True)
        self.head = mlp([2 * d + 6, d, 3])
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.keep_head = mlp([2 * d + 6, d, 1])
        nn.init.zeros_(self.keep_head[-1].weight)
        nn.init.constant_(self.keep_head[-1].bias, 4.0)

    def context(self, z_global, pt, feat, z_body, body_kv, pocc, body_levels, fuse_pos):
        if pocc is None:
            pocc = torch.zeros_like(pt[..., :2])
        cond = self.cond_proj(torch.cat([z_global, z_body], dim=-1))
        h = self.in_proj(torch.cat([pt, feat, pocc], -1))
        return cond, self.hier(pt, h, cond, body_kv, body_levels, fuse_pos)

    def out(self, h, head_feat, rel, n):
        x = torch.cat([h, head_feat, rel, n], -1)
        return self.head(x), self.keep_head(x)[..., 0]


class ClothVAE(nn.Module):
    """Garment point-cloud VAE. Latents live in the standardized frame on a canonical body; decoding runs on the canonical
    body and conditions on the target body's landmarks (body_kv), per-level body features and occupancy."""

    def __init__(self, d=256, L=4096, dg=128, dl=1, db=64, body_kv_k=2048, nvert=6890, vres=32, vbound=5.0, vlevels=2, vd=64,
                 dbeta=10, kl_w_pt=1.0, kl_w_feat=1.0, kl_w_glb=1.0, hier_schedule=(2048, 512, 256, 128), skip_weight_enc=0.01,
                 log_sigma_offset=6.0, lv_max=0.0, reanchor_k=8, reanchor_sigma=0.01, cover_rings=1, hier_vd=None, pt_sigma=0.0):
        super().__init__()
        self.pt_sigma = pt_sigma
        assert hier_vd is not None, 'hier_vd is required'
        hier_vd = tuple(hier_vd)
        assert len(hier_vd) == len(hier_schedule) + 1
        self.cloth_enc = ClothEncoder(d=d, L=L, dg=dg, dl=dl, vres=vres, vbound=vbound, vlevels=vlevels, vd=vd, kl_w_pt=kl_w_pt,
                                      kl_w_feat=kl_w_feat, kl_w_glb=kl_w_glb,
                                      skip_weight=skip_weight_enc, log_sigma_offset=log_sigma_offset, lv_max=lv_max)
        self.body_enc = BodyTower(d=d, db=db, dbeta=dbeta, hier_schedule=hier_schedule,
                                  hier_vd=tuple(max(v // 2, 32) for v in hier_vd), vres=vres, vbound=vbound)
        self.decoder = Decoder(d=d, dl=dl, cond_dim=dg + db, vres=vres, vbound=vbound, vd=vd, hier_schedule=hier_schedule,
                               hier_vd=hier_vd)
        self.body_kv_k = body_kv_k
        self.reanchor_k = reanchor_k
        self.reanchor_sigma = reanchor_sigma
        self.cover_rings = cover_rings
        self.register_buffer('data_mu', torch.zeros(3))
        self.register_buffer('data_sd', torch.ones(1))
        self.register_buffer('kv_idx', torch.zeros(body_kv_k, dtype=torch.long))
        self.register_buffer('canon_body', torch.zeros(nvert, 3))
        self.register_buffer('canon_n', torch.zeros(nvert, 3))
        f = torch.from_numpy(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smpl_faces.npy')).astype(np.int64))
        e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        self.register_buffer('adj_edges', torch.cat([e, e.flip(1)]).t().contiguous(), persistent=False)
        self.register_buffer('smpl_faces', f, persistent=False)
        self.cover_proj = mlp([4, d, d])
        nn.init.zeros_(self.cover_proj[-1].weight)
        nn.init.zeros_(self.cover_proj[-1].bias)

    def set_data_norm(self, mu, sd):
        self.data_mu.copy_(mu.reshape(3))
        self.data_sd.copy_(sd.reshape(1))

    def norm_pts(self, p):
        return (p - self.data_mu) / self.data_sd

    def denorm_pts(self, p):
        return p * self.data_sd + self.data_mu

    def set_landmarks(self, body_verts):
        tpl = self.norm_pts(body_verts).mean(0)
        self.kv_idx.copy_(sample_farthest_points(tpl.unsqueeze(0), K=self.body_kv_k, random_start_point=False)[1][0])

    def set_canon(self, body_verts):
        self.canon_body.copy_(self.norm_pts(body_verts).mean(0))
        self.canon_n.copy_(self.vert_normals(self.canon_body.unsqueeze(0))[0])

    def vert_normals(self, v):
        with torch.autocast('cuda', enabled=False):
            v = v.float()
            f = self.smpl_faces
            fn = torch.cross(v[:, f[:, 1]] - v[:, f[:, 0]], v[:, f[:, 2]] - v[:, f[:, 0]], dim=-1)
            n = torch.zeros(v.shape, device=v.device, dtype=v.dtype)
            for c in range(3):
                n.index_add_(1, f[:, c], fn)
            return F.normalize(n, dim=-1)

    def canon(self, B):
        return self.canon_body.unsqueeze(0).expand(B, -1, -1), self.canon_n.unsqueeze(0).expand(B, -1, -1)

    def reanchor(self, p, s, sn, t, tn):
        return reanchor_pts(p, s, sn, t, tn, self.reanchor_k, self.reanchor_sigma / float(self.data_sd.item()))

    def encode(self, x, bv, sample=None):
        """World-space garment on body bv -> (zg, pt, feat), pt/feat standardized on the canonical body."""
        bvn = self.norm_pts(bv)
        xn = self.reanchor(self.norm_pts(x), bvn, self.vert_normals(bvn), *self.canon(bvn.shape[0]))
        pt_lv = 2 * math.log(self.pt_sigma / 1000 / float(self.data_sd.item())) if self.pt_sigma > 0 else None
        zg, pt, feat, self.last_kl = self.cloth_enc(xn, sample, pt_lv)
        return zg, pt, feat

    def encode_body(self, bv, bn, beta=None):
        z_body, body_feat, levels = self.body_enc(self.norm_pts(bv), bn, beta)
        return z_body, gather_idx(body_feat, self.kv_idx), levels

    def index_unmet(self, pt, occ, src):
        """Occupied vertices not among the 2 nearest source-body vertices of any latent point (dilated by cover_rings)."""
        B, V = occ.shape[0], occ.shape[1]
        with torch.autocast('cuda', enabled=False):
            idx = knn_points(pt.detach().float(), src.detach().float(), K=2).idx.reshape(B, -1)
            claimed = torch.zeros(B, V, device=pt.device)
            claimed.scatter_(1, idx, 1.0)
            A = torch.sparse_coo_tensor(self.adj_edges, torch.ones(self.adj_edges.shape[1], device=pt.device), (V, V))
            for _ in range(self.cover_rings):
                claimed = torch.maximum(claimed, (torch.sparse.mm(A, claimed.t()).t() > 0).float())
        return ((occ[..., 0] > 0.5).float() * (1 - claimed)).unsqueeze(-1)

    def kv_cover(self, bvn, pt, occ, src):
        bk = gather_idx(bvn, self.kv_idx).float()
        d = knn_points(bk, pt.detach().float(), K=1).dists.clamp(min=0).sqrt()
        if occ is None:
            v = k = u = torch.zeros_like(d)
        else:
            ok = gather_idx(occ, self.kv_idx).float()
            v = ok.clamp(0, 1)
            k = (ok > -0.5).to(d.dtype)
            u = gather_idx(self.index_unmet(pt, occ, src), self.kv_idx).float()
        return torch.cat([d, v, k, u], -1)

    def gather_body_feat(self, pt, bvn, body_feat):
        w, idx = knn_w(pt, bvn, k=8)
        return gather_wc(body_feat, idx, w)

    def point_occ(self, pt, occ, bvn):
        if occ is None:
            return None
        w, idx = knn_w(pt, bvn)
        return torch.cat([gather_wc(occ.clamp(0, 1), idx, w), gather_wc((occ > -0.5).to(pt.dtype), idx, w)], -1)

    def latent_occ(self, pt, occ, bvn):
        if occ is None:
            return None
        return self.point_occ(pt, occ, self.canon(bvn.shape[0])[0])

    def canon_levels(self, levels, bvn):
        src = self.canon(bvn.shape[0])[0]
        out = []
        for p, f in levels:
            i = knn_points(p.float(), bvn.float(), K=1).idx.expand(-1, -1, 3)
            out.append((src.gather(1, i).to(p.dtype), f))
        return out

    def decode_from(self, z_global, pt, feat, bv, bn, z_body, body_kv, body_levels, occ=None):
        """Per-point keep logits (does the point survive the coverage `occ`) land in self.last_keep."""
        bvn = self.norm_pts(bv)
        src, _ = self.canon(bvn.shape[0])
        gi = knn_points(pt.float(), src.float(), K=1).idx[..., 0].unsqueeze(-1).expand(-1, -1, 3)
        tgt_base, tgt_n = bvn.gather(1, gi), F.normalize(bn.gather(1, gi), dim=-1)
        FB, P, levels = src, pt, self.canon_levels(body_levels, bvn)
        base = FB.gather(1, gi)
        body_kv = body_kv + self.cover_proj(self.kv_cover(FB, P, occ, src).to(body_kv.dtype))
        cond, h = self.decoder.context(z_global, P, feat, z_body, body_kv, self.point_occ(P, occ, FB), levels, base)
        head_feat = self.gather_body_feat(base, FB, levels[0][1])
        off, self.last_keep = self.decoder.out(h, head_feat, tgt_base - base, tgt_n)
        return self.denorm_pts(P + off)

    @torch.no_grad()
    def trim_keep(self, z_global, pt, feat, bv, bn, occ, beta=None):
        """[B, N] bool: latent points that survive the coverage `occ` (keep head)."""
        self.decode(z_global, pt, feat, bv, bn, occ, beta)
        return self.last_keep.float() >= 0

    def decode(self, z_global, pt, feat, bv, bn, occ=None, beta=None):
        z_body, body_kv, levels = self.encode_body(bv, bn, beta)
        return self.decode_from(z_global, pt, feat, bv, bn, z_body, body_kv, levels, occ)

    def forward(self, x, bv, bn, beta=None, occ=None):
        z_body, body_kv, levels = self.encode_body(bv, bn, beta)
        zg, pt, feat = self.encode(x, bv)
        pred = self.decode_from(zg, pt, feat, bv, bn, z_body, body_kv, levels, occ)
        return pred, self.last_kl, pt, feat, self.last_keep

def build_vae(vc):
    return ClothVAE(d=vc['d'], L=vc['L'], dg=vc['dg'], dl=vc['dl'], body_kv_k=vc['body_kv_k'], vres=vc['vres'],
                    vbound=vc['vbound'], vlevels=vc['vlevels'], vd=vc['vd'], dbeta=vc['dbeta'], kl_w_pt=vc['kl_w_pt'],
                    kl_w_feat=vc['kl_w_feat'], kl_w_glb=vc['kl_w_glb'],
                    hier_schedule=tuple(int(x) for x in vc['hier_schedule'].split(',')),
                    skip_weight_enc=vc['skip_weight_enc'], log_sigma_offset=vc['log_sigma_offset'], lv_max=vc['lv_max'],
                    reanchor_k=vc.get('reanchor_k', 8), reanchor_sigma=vc.get('reanchor_sigma', 0.01),
                    cover_rings=vc.get('cover_rings', 1), pt_sigma=vc.get('pt_sigma', 0.0),
                    hier_vd=tuple(int(x) for x in vc['hier_vd'].split(',')))

