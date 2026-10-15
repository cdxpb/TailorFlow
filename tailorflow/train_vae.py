import sys, os, time, json, math, random, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from pytorch3d.ops import knn_points

from tailorflow.data_io import build_mesh_loader, MeshClothDataset, C3D_ROOT, RENDER_PICKS
from tailorflow.model import build_vae
from tailorflow.common import EXPS, save_config, inside_pct, ddp_sync_check
from tailorflow.occ_aug import EditAug, CutDataset
from tailorflow.vis import cloud_panel, save_grid
from tailorflow.metrics import chamfer, emd as emd_cost, fps

p = argparse.ArgumentParser()
p.add_argument('--exp', required=True)
p.add_argument('--cates', default='Dress,Upper,Trousers,Skirt,Jumpsuit')
p.add_argument('--limit', type=int, default=0, help='max train items per category (0 = all)')
p.add_argument('--bs', type=int, default=6, help='per GPU')
p.add_argument('--L', type=int, default=4096)
p.add_argument('--d', type=int, default=128)
p.add_argument('--dg', type=int, default=128)
p.add_argument('--dl', type=int, default=0)
p.add_argument('--dbeta', type=int, default=10)
p.add_argument('--body_kv_k', type=int, default=2048)
p.add_argument('--hier_schedule', default='2048,512,256,128')
p.add_argument('--hier_vd', default='32,32,64,64,64', help='voxel width per level (stem + one per hier level)')
p.add_argument('--vres', type=int, default=32)
p.add_argument('--vbound', type=float, default=5.0)
p.add_argument('--vlevels', type=int, default=2)
p.add_argument('--vd', type=int, default=64)
p.add_argument('--reanchor_k', type=int, default=8)
p.add_argument('--reanchor_sigma', type=float, default=0.01)
p.add_argument('--cover_rings', type=int, default=1)
p.add_argument('--skip_weight_enc', type=float, default=0.01)
p.add_argument('--pt_sigma', type=float, default=1.5, help='fixed posterior std of the latent positions in mm (0 = learned)')
p.add_argument('--log_sigma_offset', type=float, default=6.0)
p.add_argument('--lv_max', type=float, default=0.0)
p.add_argument('--kl_w', type=float, default=0.01)
p.add_argument('--kl_w_pt', type=float, default=30.0)
p.add_argument('--kl_w_feat', type=float, default=0.5)
p.add_argument('--kl_w_glb', type=float, default=0.05)
p.add_argument('--emd_w', type=float, default=1.0)
p.add_argument('--l1_w', type=float, default=0.02, help='ordered L1 of each output point to its own input point (kept points on trim rows)')
p.add_argument('--edge_w', type=float, default=1.0, help='kNN edge-length preservation between latent pt and the decoded cloud (untrimmed rows)')
p.add_argument('--keep_w', type=float, default=5.0, help='BCE of the keep head (does a latent point survive the coverage)')
p.add_argument('--occ_drop', type=float, default=0.3)
p.add_argument('--occ_edit', type=float, default=0.6, help='probability of an EditAug coverage cut (hem, per-arm sleeves, pairs)')
p.add_argument('--trim_frac', type=float, default=0.5, help='cut rows that are trims (full garment in, cut coverage, keep labels); '
               'the rest autoencode the cut garment')
p.add_argument('--grow_frac', type=float, default=0.5, help='non-trim cut rows that are grows (cut garment in, full garment out under '
               'full coverage; CD/EMD only)')
p.add_argument('--grow_w', type=float, default=0.1, help='weight of the CD/EMD of grow rows')
p.add_argument('--aug_ramp_steps', type=int, default=1000)
p.add_argument('--data_norm_items', type=int, default=256)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--lr_warmup', type=int, default=500)
p.add_argument('--grad_clip', type=float, default=1.0)
p.add_argument('--ema_decay', type=float, default=0.999)
p.add_argument('--ema_start_step', type=int, default=1000)
p.add_argument('--amp', type=int, default=1)
p.add_argument('--steps', type=int, default=20000)
p.add_argument('--save_every', type=int, default=500)
p.add_argument('--keep_after', type=int, default=10000, help='from this step on, also keep ckpt_<step>.pt every --keep_every steps')
p.add_argument('--keep_every', type=int, default=1000)
p.add_argument('--diag_every', type=int, default=500, help='per-category val + occ trim/grow diagnostics (0 = off)')
p.add_argument('--bs_val', type=int, default=16)
p.add_argument('--val_n_per_cate', type=int, default=64)
p.add_argument('--occ_val_per_cate', type=int, default=8)
p.add_argument('--num_workers', type=int, default=4)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--ddp_bucket_mb', type=int, default=25)
args = p.parse_args()

world_size = int(os.environ.get('WORLD_SIZE', 1))
local_rank = int(os.environ.get('LOCAL_RANK', 0))
rank = int(os.environ.get('RANK', 0))
distributed = world_size > 1
if distributed:
    dist.init_process_group('nccl')
    torch.cuda.set_device(local_rank)
is_main = rank == 0
device = torch.device(f'cuda:{local_rank}' if distributed else 'cuda')
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

out_dir = os.path.join(EXPS, args.exp)
ckpt_path, best_path = f'{out_dir}/ckpt.pt', f'{out_dir}/ckpt_best.pt'
if is_main:
    os.makedirs(f'{out_dir}/renders', exist_ok=True)
if os.path.exists(ckpt_path) and os.path.exists(f'{out_dir}/config.json'):
    saved = json.load(open(f'{out_dir}/config.json'))
    explicit = {a.lstrip('-').split('=')[0] for a in sys.argv[1:] if a.startswith('--')}
    for k, v in saved.items():
        if hasattr(args, k) and k not in explicit:
            setattr(args, k, v)
elif is_main:
    save_config(args, out_dir, extra={'world_size': world_size})

cates = args.cates.split(',')
loader, sampler = build_mesh_loader(cates, 'train', args.bs, limit=args.limit, num_workers=args.num_workers,
                                    distributed=distributed, world_size=world_size, rank=rank, wrap=lambda d: CutDataset(d, args.occ_edit))
ds = loader.dataset
val_ds = MeshClothDataset(cates, 'val')
if is_main:
    print(f'DATA: {C3D_ROOT} train {len(ds)} | val {len(val_ds)} | world_size {world_size}', flush=True)

model = build_vae(vars(args)).to(device)
torch.manual_seed(args.seed + 1000 * rank)
torch.cuda.manual_seed_all(args.seed + 1000 * rank)
np.random.seed(args.seed + 1000 * rank)
random.seed(args.seed + 1000 * rank)
amp_dtype = torch.bfloat16 if args.amp else None


def to_dev(batch):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def collate(items):
    return {k: (torch.stack([it[k] for it in items]) if torch.is_tensor(v) else [it[k] for it in items]) for k, v in items[0].items()}


fixed_train = to_dev(next(iter(loader)))

with torch.no_grad():
    acc, n = [], 0
    stride = max(1, len(ds) // (args.data_norm_items // 2))
    for i in range(0, len(ds), stride):
        acc.append(ds[i]['tr_points'].reshape(-1, 3))
        n += 1
        if n >= int(args.data_norm_items * 1.5):
            break
    allp = torch.cat(acc, 0)
    mu, sd = allp.mean(0), allp.reshape(-1).std().reshape(1)
model.set_data_norm(mu.to(device), sd.to(device))
model.set_landmarks(MeshClothDataset(cates, 'train', limit=1)[0]['body_verts'].unsqueeze(0).to(device))
cidx = list(range(0, len(ds), max(1, len(ds) // 256)))[:256]
model.set_canon(torch.stack([ds[i]['body_verts'] for i in cidx]).to(device))
if is_main:
    print(f'data_norm from {n} garments: mu {[round(v, 4) for v in mu.tolist()]} sd {sd.item():.4f} | canon body: mean of {len(cidx)}', flush=True)

if distributed:
    model = DDP(model, device_ids=[local_rank], bucket_cap_mb=args.ddp_bucket_mb, gradient_as_bucket_view=True)
raw_model = model.module if distributed else model

opt = Adam(model.parameters(), lr=args.lr)
if is_main:
    print('params(M):', sum(q.numel() for q in raw_model.parameters()) / 1e6, flush=True)

start_step, best_val = 0, float('inf')
if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ck['model'])
    if 'opt' in ck:
        opt.load_state_dict(ck['opt'])
    start_step, best_val = ck['step'], ck.get('best_val', float('inf'))
    live = raw_model.state_dict()
    ema = {k: ck['ema'][k].to(device) if k in ck['ema'] else v.detach().clone() for k, v in live.items()}
    if is_main:
        print('resumed from step', start_step, 'best_val', best_val, flush=True)
else:
    ema = {k: v.detach().clone() for k, v in raw_model.state_dict().items()}


def ema_update(step):
    with torch.no_grad():
        pairs = [(ema[k], v) for k, v in raw_model.state_dict().items()]
        flt = [(e, v) for e, v in pairs if e.dtype.is_floating_point]
        es, vs = [e for e, _ in flt], [v for _, v in flt]
        if step < args.ema_start_step:
            torch._foreach_copy_(es, vs)
        else:
            torch._foreach_mul_(es, args.ema_decay)
            torch._foreach_add_(es, vs, alpha=1 - args.ema_decay)
        for e, v in pairs:
            if not e.dtype.is_floating_point:
                e.copy_(v)


def augment(b, x, occ, ramp):
    """Rows: unknown occupancy (occ_drop); loader-side cuts (CutDataset) become trims (full garment in, cut coverage, keep
    labels) or cut autoencoding (cut garment in and out) or grows (cut garment in, full garment out); the rest are plain.
    -> enc_x, occupancy, target, keep, trimmed, grown."""
    B = x.shape[0]
    drop = torch.rand(B, device=device) < args.occ_drop
    cut = b['cut'].to(device) & ~drop & (torch.rand(B, device=device) < ramp)
    trim = cut & (torch.rand(B, device=device) < args.trim_frac)
    grow = cut & ~trim & (torch.rand(B, device=device) < args.grow_frac)
    rec = cut & ~trim & ~grow
    enc_x, tgt = x.clone(), x.clone()
    if (rec | grow).any():
        crops = fps(b['cut_dense'].to(device)[rec | grow], x.shape[1], random_start=True)[0]
        enc_x[rec | grow] = crops
        tgt[rec] = crops[rec[rec | grow]]
    occ_c = torch.where((cut & ~grow).view(-1, 1, 1), b['cut_m'].to(device).unsqueeze(-1), occ)
    occ_c = torch.where(drop.view(-1, 1, 1), torch.full_like(occ, -1.0), occ_c)
    keep = torch.where(trim.view(-1, 1), b['cut_keep'].to(device), torch.ones_like(b['cut_keep'].to(device)))
    return enc_x, occ_c, tgt, keep, trim, grow


def chamfer_emd(pred, gt):
    dl, dr = chamfer(pred, gt)
    return dl.sum(-1) + dr.sum(-1), emd_cost(pred, gt) * pred.shape[1]


def edge_len_loss(pt, pred, k=8):
    predn = raw_model.norm_pts(pred).float()
    pt = pt.float()
    idx = knn_points(pt, pt, K=k + 1).idx[..., 1:]
    B, L, kk = idx.shape
    flat = idx.reshape(B, L * kk, 1).expand(-1, -1, 3)
    d_pt = (pt.gather(1, flat).reshape(B, L, kk, 3) - pt.unsqueeze(2)).norm(dim=-1)
    d_pr = (predn.gather(1, flat).reshape(B, L, kk, 3) - predn.unsqueeze(2)).norm(dim=-1)
    return (d_pr - d_pt).abs().mean((1, 2))


@torch.no_grad()
def recon(x, bv, bn, beta, occ):
    z_body, kv, lv = raw_model.encode_body(bv, bn, beta)
    zg, pt, feat = raw_model.encode(x, bv, sample=True)
    return raw_model.decode_from(zg, pt, feat, bv, bn, z_body, kv, lv, occ)


@torch.no_grad()
def run_val(vb):
    raw_model.eval()
    pred = recon(vb['tr_points'], vb['body_verts'], vb['body_normals'], vb['beta'], vb['body_occupancy'])
    cd, emd = chamfer_emd(pred.float(), vb['tr_points'].float())
    raw_model.train()
    return cd.mean().item(), emd.mean().item(), inside_pct(pred, vb['body_verts'], vb['body_normals']) / 100


@torch.no_grad()
def run_xfer_val(fb):
    raw_model.eval()
    x, bv, bn, beta = fb['tr_points'], fb['body_verts'], fb['body_normals'], fb['beta']
    own = inside_pct(recon(x, bv, bn, beta, fb['body_occupancy']), bv, bn) / 100
    roll = torch.roll(torch.arange(x.shape[0], device=device), 1)
    zg, pt, feat = raw_model.encode(x, bv, sample=True)
    xfer = raw_model.decode(zg, pt, feat, bv[roll], bn[roll], None, beta[roll])
    raw_model.train()
    nn_sp = lambda p: knn_points(p.float(), p.float(), K=2).dists[..., 1].clamp(min=0).sqrt().mean(1)
    return own, inside_pct(xfer, bv[roll], bn[roll]) / 100, (nn_sp(xfer) / nn_sp(x)).mean().item()


def items_per_cate(dset, n):
    by = {}
    for i, (c, _, stem) in enumerate(dset.items):
        if len(by.setdefault(c, [])) < n:
            by[c].append(i)
    return by


def fixed_cuts(b, aug, seed=0):
    """One coverage cut per item, drawn once with a fixed seed: (cut coverage, keep labels, cut garment resampled)."""
    state = random.getstate()
    random.seed(seed)
    ms, keeps, crops = [], [], []
    for i in range(b['tr_points'].shape[0]):
        x, bv, occ, c, m = b['tr_points'][i], b['body_verts'][i], b['body_occupancy'][i], b['sid'][i], b['mid'][i]
        cut = None
        for _ in range(8):
            cut = cut or aug.cut(x, bv, occ, c, m, aug.random_spec(c, m))
        ms.append(occ if cut is None else cut['m'].float().view(-1, 1))
        keeps.append(torch.ones(x.shape[0], dtype=torch.bool, device=x.device) if cut is None else cut['keep'])
        crops.append(x if cut is None else aug.cut_points(cut, x.device))
    random.setstate(state)
    return torch.stack(ms), torch.stack(keeps), torch.stack(crops)


fixed_val = to_dev(collate([val_ds[i] for i in range(args.bs_val)])) if is_main else None
render_batch = render_tags = val_by_cate = occ_diag = render_edits = None
if is_main:
    tmpl = lambda d: torch.stack([d[i]['body_verts'] for i in range(0, len(d), max(1, len(d) // 300))]).mean(0)
    val_aug, train_aug = EditAug(val_ds.items, tmpl(val_ds), device=device), EditAug(ds.items, tmpl(ds), device=device)
    picks = json.load(open(RENDER_PICKS))
    key = {(c, stem): i for i, (c, _, stem) in enumerate(ds.items)}
    sel = [(key[(c, picks[c][w])], f'{c[:5]}-{w[0]}') for c in cates for w in ('tight', 'loose') if (c, picks.get(c, {}).get(w)) in key]
    sel = sel or [(idx[0], c[:5]) for c, idx in items_per_cate(ds, 1).items()]
    render_batch, render_tags = to_dev(collate([ds[i] for i, _ in sel])), [t for _, t in sel]
    val_by_cate = {c: to_dev(collate([val_ds[i] for i in idx])) for c, idx in items_per_cate(val_ds, args.val_n_per_cate).items()}
    ob = to_dev(collate([val_ds[i] for idx in items_per_cate(val_ds, args.occ_val_per_cate).values() for i in idx]))
    occ_diag = (ob, fixed_cuts(ob, val_aug))
    render_edits = fixed_cuts(render_batch, train_aug, seed=1)


def run_val_percat(chunk=16):
    out = {}
    for c, vb in val_by_cate.items():
        n = vb['tr_points'].shape[0]
        tot = [0.0, 0.0, 0.0]
        for s in range(0, n, chunk):
            m = min(chunk, n - s)
            r = run_val({k: v[s:s + chunk] for k, v in vb.items() if torch.is_tensor(v)})
            tot = [a + b * m for a, b in zip(tot, r)]
        out[c] = tuple(t / n for t in tot)
    return out


@torch.no_grad()
def run_occ_diag():
    """Fixed val cuts: plain recon, trim (keep head: accuracy, recall of cut points, inside% of the kept output) and the cut
    garment autoencoded under its own coverage."""
    ob, (m, keep, crops) = occ_diag
    x, bv, bn, beta, occ = ob['tr_points'], ob['body_verts'], ob['body_normals'], ob['beta'], ob['body_occupancy']
    raw_model.eval()
    out = {}
    cd, emd = chamfer_emd(recon(x, bv, bn, beta, occ).float(), x.float())
    out['normal_cd'], out['normal_emd'] = cd.mean().item(), emd.mean().item()
    y = recon(x, bv, bn, beta, m).float()
    k = raw_model.last_keep.float() >= 0
    out['trim_acc'] = (k == keep).float().mean().item() * 100
    out['trim_recall'] = ((~k & ~keep).sum() / (~keep).sum().clamp(min=1)).item() * 100
    out['trim_inside'] = float(np.mean([inside_pct(y[i:i + 1, k[i]], bv[i:i + 1], bn[i:i + 1]) for i in range(len(y)) if k[i].any()]))
    cd, emd = chamfer_emd(recon(crops, bv, bn, beta, m).float(), crops.float())
    out['cut_cd'], out['cut_emd'] = cd.mean().item(), emd.mean().item()
    raw_model.train()
    return out


@torch.no_grad()
def save_render(step):
    raw_model.eval()
    fb = render_batch
    x, bv, bn, beta, occ = fb['tr_points'], fb['body_verts'], fb['body_normals'], fb['beta'], fb['body_occupancy']
    roll = torch.roll(torch.arange(x.shape[0], device=device), 1)
    zg, pt, feat = raw_model.encode(x, bv, sample=True)
    cb = raw_model.canon(x.shape[0])[0]
    panels = [('gt', x, bv, None), ('latent', pt, cb, None), ('recon', raw_model.decode(zg, pt, feat, bv, bn, occ, beta), bv, occ),
              ('xfer', raw_model.decode(zg, pt, feat, bv[roll], bn[roll], None, beta[roll]), bv[roll], None),
              ('noOcc', raw_model.decode(zg, pt, feat, bv, bn, torch.full_like(occ, -1.0), beta), bv, None)]
    m, keep, crops = render_edits
    trimmed = raw_model.decode(zg, pt, feat, bv, bn, m, beta)
    kept = raw_model.last_keep.float() >= 0
    czg, cpt, cfeat = raw_model.encode(crops, bv, sample=True)
    panels += [('trim_tgt', [x[i][keep[i]] for i in range(len(x))], bv, m), ('trim', [trimmed[i][kept[i]] for i in range(len(x))], bv, m),
               ('cut', crops, bv, m), ('cut_recon', raw_model.decode(czg, cpt, cfeat, bv, bn, m, beta), bv, m)]
    rows = [np.concatenate([cloud_panel(pc[i], bd[i], f'{t}_{nm}', occ=None if oc is None else oc[i])
                            for nm, pc, bd, oc in panels], axis=1) for i, t in enumerate(render_tags)]
    save_grid(rows, f'{out_dir}/renders/step_{step:06d}.png')
    raw_model.train()


def atomic_save(obj, path):
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)


progress_path = f'{out_dir}/progress.csv'
if is_main and not os.path.exists(progress_path):
    with open(progress_path, 'w') as f:
        f.write('step,loss,cd,emd,l1,edge,kl,val_cd,val_emd,val_inside,own_inside,xfer_inside,elapsed_s\n')

step, epoch = start_step, 0
t0 = time.time()
loss_hist = []
last_saved = -1
while step < args.steps:
    if sampler is not None:
        sampler.set_epoch(epoch)
    for batch in loader:
        b = to_dev(batch)
        x, bv, bn, beta, occ = b['tr_points'], b['body_verts'], b['body_normals'], b['beta'], b['body_occupancy']
        ramp = min(1.0, step / args.aug_ramp_steps) if args.aug_ramp_steps > 0 else 1.0
        enc_x, occ_aug, tgt, keep, trim, grow = augment(b, x, occ, ramp)
        with torch.autocast('cuda', dtype=amp_dtype, enabled=bool(args.amp)):
            pred, kl_all, pt_all, _, keep_logit = model(enc_x, bv, bn, beta=beta, occ=occ_aug)
            un = (~trim).float()
            kl = kl_all.mean()
            cd_all, emd_all = chamfer_emd(pred.float(), tgt.float())
            rw = un * torch.where(grow, args.grow_w, 1.0)
            cd, emd = (cd_all * rw).sum() / un.sum().clamp(min=1), (emd_all * rw).sum() / un.sum().clamp(min=1)
            lw = (~grow).float()
            l1 = ((pred.float() - tgt.float()).abs().sum(-1) * keep).sum(-1).mul(lw).sum() / lw.sum().clamp(min=1)
            ew = un * (~grow).float()
            edge = (edge_len_loss(pt_all, pred) * ew).sum() / ew.sum().clamp(min=1) if args.edge_w > 0 else torch.zeros((), device=device)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(keep_logit.float(), keep.float())
            loss = cd + args.emd_w * emd + args.l1_w * l1 + args.edge_w * edge + args.kl_w * kl + args.keep_w * bce

        lr_now = args.lr * (min(1.0, (step + 1) / args.lr_warmup) if args.lr_warmup > 0 else 1.0)
        for g in opt.param_groups:
            g['lr'] = lr_now
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(gnorm):
            if is_main:
                print(f'  NON-FINITE GRAD step {step}: loss {loss.item():.4g} cd {cd.item():.4g} emd {emd.item():.4g} kl {kl.item():.4g}', flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        lv = loss.item()
        spike = False
        if math.isfinite(lv):
            loss_hist = (loss_hist + [lv])[-100:]
            spike = len(loss_hist) >= 20 and lv > 30 * sorted(loss_hist)[len(loss_hist) // 2]
        if spike:
            if is_main:
                print(f'  ema skip step {step}: loss {lv:.1f}', flush=True)
        else:
            ema_update(step)

        if distributed and (step in (start_step, start_step + 1, start_step + 100) or step % args.save_every == 0):
            msg = ddp_sync_check(raw_model.parameters(), x, step)
            if is_main and step <= start_step + 100:
                print('  ' + msg, flush=True)
        if is_main and step % 20 == 0:
            print(f'step {step} gnorm {gnorm.item():.2f} loss {lv:.2f} cd {cd.item():.2f} emd {emd.item():.2f} '
                  f'l1 {l1.item():.2f} edge {edge.item():.4f} keep {bce.item():.4f} kl {kl.item():.4f} ({time.time() - t0:.1f}s)', flush=True)

        if is_main and step % args.save_every == 0:
            backup = {k: v.detach().clone() for k, v in raw_model.state_dict().items()}
            raw_model.load_state_dict(ema)
            save_render(step)
            val_cd, val_emd, val_inside = run_val(fixed_val)
            own_inside, xfer_inside, xfer_spacing = run_xfer_val(fixed_train)
            if args.diag_every > 0 and step % args.diag_every == 0:
                if occ_diag is not None:
                    od = run_occ_diag()
                    print('  occ diag: ' + ' '.join(f'{k} {v:.2f}' for k, v in od.items()), flush=True)
                percat = run_val_percat()
                print('  per-cat val: ' + ' | '.join(f'{c} cd={v[0]:.2f} emd={v[1]:.2f}' for c, v in percat.items()), flush=True)
            val_score = val_cd + val_emd
            atomic_save({'model': backup, 'ema': ema, 'opt': opt.state_dict(), 'step': step, 'best_val': best_val}, ckpt_path)
            last_saved = step
            if val_score < best_val:
                best_val = val_score
                atomic_save({'model': backup, 'ema': ema, 'step': step, 'val_score': val_score}, best_path)
            raw_model.load_state_dict(backup)
            print(f'  [ema] val cd {val_cd:.2f} emd {val_emd:.2f} inside {val_inside * 100:.2f}% | train own {own_inside * 100:.2f}% '
                  f'xfer {xfer_inside * 100:.2f}% xfer_spacing {xfer_spacing:.3f}', flush=True)
            with open(progress_path, 'a') as f:
                f.write(f'{step},{lv:.4f},{cd.item():.4f},{emd.item():.4f},{l1.item():.4f},{edge.item():.5f},{kl.item():.4f},'
                        f'{val_cd:.4f},{val_emd:.4f},{val_inside:.4f},{own_inside:.4f},{xfer_inside:.4f},{time.time() - t0:.1f}\n')
        if is_main and step >= args.keep_after and step % args.keep_every == 0:
            atomic_save({'model': raw_model.state_dict(), 'ema': ema, 'step': step}, f'{out_dir}/ckpt_{step:06d}.pt')

        step += 1
        if step >= args.steps:
            break
    epoch += 1

if is_main:
    if step != last_saved:
        atomic_save({'model': raw_model.state_dict(), 'ema': ema, 'opt': opt.state_dict(), 'step': step, 'best_val': best_val}, ckpt_path)
    atomic_save({'model': raw_model.state_dict(), 'ema': ema, 'step': step}, f'{out_dir}/ckpt_{step:06d}.pt')
    print(f'final save: ckpt_{step:06d}.pt', flush=True)
if distributed:
    dist.barrier()
