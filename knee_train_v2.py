# V2 multi-plane knee classifier — target AUC >= 0.9666
# Improvements: stronger backbones, focal loss, MixUp, EMA, gated + cross-plane attention,
# gradient accumulation, label calibration (isotonic), auxiliary per-plane heads, random erasing.
#
# Usage on Kaggle:
#   ARCH=convnextv2_nano.fcmae_ft_in22k_in1k FOLD=0 python knee_train_v2.py
#   ARCH=efficientnet_b3.ra2_in1k            FOLD=1 python knee_train_v2.py
#   ARCH=resnet34d.ra2_in1k                  FOLD=2 FULL=1 python knee_train_v2.py
#   DINOv3 backbones (self-supervised, LVD-1689M; ungated timm weights):
#   ARCH=vit_small_patch16_dinov3.lvd1689m LR=1e-4 LAYER_DECAY=0.8 FOLD=0 python knee_train_v2.py
#   ARCH=vit_base_patch16_dinov3.lvd1689m  LR=8e-5 LAYER_DECAY=0.8 BS=4 ACCUM=3 FOLD=1 python knee_train_v2.py
#   ARCH=convnext_tiny.dinov3_lvd1689m     LR=1.5e-4 LAYER_DECAY=0.85 FOLD=2 python knee_train_v2.py
import os, glob, time, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
torch.backends.cudnn.benchmark = True

L = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
     'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

def find(name):
    for pat in [f'/kaggle/input/*/{name}', f'/kaggle/input/*/*/{name}',
                f'/kaggle/input/*/*/*/{name}', f'/kaggle/input/*/*/*/*/{name}',
                f'/kaggle/working/{name}']:   # built in the same notebook by knee_prep.py
        g = glob.glob(pat)
        if g: return g[0]
    raise FileNotFoundError(name)

D = '/kaggle/input/competitions/rsna-knee-abnormality-detection'

CFG = dict(
    arch    = os.environ.get('ARCH', 'convnextv2_nano.fcmae_ft_in22k_in1k'),
    epochs  = int(os.environ.get('EPOCHS', '20')),
    bs      = int(os.environ.get('BS', '6')),
    lr      = float(os.environ.get('LR', '2e-4')),
    wd      = 0.05,
    fold    = int(os.environ.get('FOLD', '0')),
    nfold   = int(os.environ.get('NFOLD', '4')),
    seed    = int(os.environ.get('SEED', '42')),
    full    = int(os.environ.get('FULL', '0')),
    mixup   = float(os.environ.get('MIXUP', '0.4')),
    focal   = float(os.environ.get('FOCAL', '1.0')),
    smooth  = float(os.environ.get('SMOOTH', '0.02')),
    ema     = float(os.environ.get('EMA_DECAY', '0.998')),
    accum   = int(os.environ.get('ACCUM', '2')),
    swa_pct = float(os.environ.get('SWA_PCT', '0.75')),
    # layer-wise lr decay for the encoder (1.0 = off); defaults to 0.8 for DINOv3 backbones
    layer_decay = float(os.environ.get('LAYER_DECAY', '0')),
    grad_ckpt   = int(os.environ.get('GRAD_CKPT', '-1')),   # -1 = auto (on for ViTs)
    amp         = os.environ.get('AMP', 'fp16'),            # fp16 (T4/P100) or bf16 (A100/H100)
)
IS_DINOV3 = 'dinov3' in CFG['arch']
IS_VIT = CFG['arch'].startswith(('vit_', 'eva', 'deit', 'beit'))
if CFG['layer_decay'] <= 0: CFG['layer_decay'] = 0.8 if IS_DINOV3 else 1.0
if CFG['grad_ckpt'] < 0:   CFG['grad_ckpt'] = int(IS_VIT)
AMP_DTYPE = torch.bfloat16 if CFG['amp'] == 'bf16' else torch.float16
print(CFG, flush=True)
ARCH_TAG = CFG['arch'].replace('.', '-')
t0 = time.time()

# ── data ──────────────────────────────────────────────────────────────────────
X = np.load(find('train_u8.npy'))
idx = pd.read_csv(find('train_index.csv'))
_, _, NS, SZ, _ = X.shape
print(f'X {X.shape}  NS={NS} SZ={SZ}  {time.time()-t0:.0f}s', flush=True)

tr = pd.read_csv(D + '/train.csv')
Ys = []
for nm in ['llm_labels.csv', 'llm_labels_gen.csv']:
    try:
        l = idx.merge(pd.read_csv(find(nm)), on='StudyInstanceUID', how='left')
        Ys.append(l[L].values.astype(np.float32))
        print('loaded', nm, flush=True)
    except FileNotFoundError:
        pass
Y = np.nanmean(np.stack(Ys), 0)
Y = np.nan_to_num(Y, nan=0.0)

# gold labels
gold = tr.set_index('StudyInstanceUID').loc[idx.StudyInstanceUID, L]
gmask = gold.notna().all(axis=1).values
G = gold.values.astype(np.float32)

# ── label calibration via isotonic regression on gold set ─────────────────────
def calibrate(llm, gv, mask):
    cal = llm.copy()
    for c in range(12):
        m = mask & ~np.isnan(llm[:, c])
        if m.sum() < 30: continue
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        ir.fit(llm[m, c], gv[m, c])
        cal[:, c] = ir.predict(np.clip(llm[:, c], 0, 1))
    return cal

Y = calibrate(Y, G, gmask)
print('calibrated means:', np.round(Y.mean(0), 3), flush=True)
Y[gmask] = G[gmask]  # gold overrides

# label smoothing (skip gold)
eps = CFG['smooth']
Y_s = Y.copy()
if eps > 0:
    Y_s[~gmask] = Y[~gmask] * (1 - eps) + (1 - Y[~gmask]) * eps

# ── folds ─────────────────────────────────────────────────────────────────────
rng = np.random.RandomState(0)
fold_of = rng.randint(0, CFG['nfold'], len(Y))
va  = np.where((fold_of == CFG['fold']) | gmask)[0]
trn = np.arange(len(Y)) if CFG['full'] else np.where((fold_of != CFG['fold']) & ~gmask)[0]
print(f'train {len(trn)}  val {len(va)}  gold {gmask.sum()}', flush=True)

# class weights from gold positive rates
gold_pos = G[gmask].mean(0)
cls_w = torch.from_numpy(np.clip(1.0 / (gold_pos + 0.02), 0.5, 8.0).astype(np.float32)).cuda()
print('gold pos:', np.round(gold_pos, 3), '\ncls_w:  ', cls_w.cpu().numpy().round(2), flush=True)

torch.manual_seed(CFG['seed']); np.random.seed(CFG['seed'])

# ── EMA ───────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay):
        self.shadow = deepcopy(model); self.shadow.eval(); self.decay = decay
    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.data.copy_(b.data)

# ── Model ─────────────────────────────────────────────────────────────────────
class NetV2(nn.Module):
    """2.5-D multi-plane classifier with gated attention, cross-plane self-attention, and aux heads."""
    def __init__(s, arch, ns=20, pre=True):
        super().__init__()
        s.enc = timm.create_model(arch, pretrained=pre, num_classes=0, in_chans=3)
        c = s.enc.num_features; s.c = c; s.ns = ns
        s.plane_emb = nn.Parameter(torch.randn(3, c) * 0.02)
        s.pos_emb   = nn.Parameter(torch.randn(1, 1, ns, c) * 0.02)
        # gated attention (Ilse et al.)
        s.att_v = nn.Sequential(nn.Linear(c, 256), nn.Tanh())
        s.att_u = nn.Sequential(nn.Linear(c, 256), nn.Sigmoid())
        s.att_w = nn.Linear(256, 1)
        # cross-plane self-attention
        s.xp_attn = nn.MultiheadAttention(c, num_heads=min(4, c // 64 or 1), dropout=0.1, batch_first=True)
        s.xp_norm = nn.LayerNorm(c)
        # main classification head
        s.head = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(3 * c, 512), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(512, 12))
        # auxiliary per-plane heads (regularization during training)
        s.aux = nn.ModuleList([nn.Linear(c, 12) for _ in range(3)])

    def forward(s, x, return_aux=False):
      with torch.autocast('cuda', dtype=AMP_DTYPE):
        B, P, N, H, W = x.shape
        x = x.float() / 255.0
        # 2.5-D triplets (prev, cur, next)
        pad = torch.cat([x[:,:,:1], x, x[:,:,-1:]], 2)
        trip = torch.stack([pad[:,:,:-2], pad[:,:,1:-1], pad[:,:,2:]], 3)
        trip = (trip - 0.45) / 0.25
        f = s.enc(trip.reshape(B*P*N, 3, H, W)).reshape(B, P, N, s.c)
        # positional + plane embeddings (interpolate if NS differs)
        pos = s.pos_emb
        if N != s.ns:
            pos = F.interpolate(pos[0].transpose(1, 2), size=N, mode='linear',
                                align_corners=False).transpose(1, 2)[None]
        f = f + pos + s.plane_emb[None,:,None,:]
        # gated attention pooling over slices
        h = s.att_v(f) * s.att_u(f)
        a = torch.softmax(s.att_w(h).squeeze(-1), -1)          # B,P,N
        v = (a.unsqueeze(-1) * f).sum(2)                        # B,P,C
        # cross-plane self-attention
        v = s.xp_norm(v + s.xp_attn(v, v, v, need_weights=False)[0])
        main = s.head(v.reshape(B, P * s.c))
        if not return_aux:
            return main
        aux = torch.stack([s.aux[p](v[:, p]) for p in range(P)], 1)  # B,3,12
        return main, aux

# ── losses ────────────────────────────────────────────────────────────────────
def focal_bce(logits, targets, gamma=1.0, weight=None):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt  = torch.sigmoid(logits) * targets + torch.sigmoid(-logits) * (1 - targets)
    loss = ((1 - pt) ** gamma) * bce
    if weight is not None:
        loss = loss * weight
    return loss.mean()

def mixup(x, y, alpha):
    if alpha <= 0: return x, y
    lam = max(np.random.beta(alpha, alpha), 0.55)
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1-lam) * x[perm], lam * y + (1-lam) * y[perm]

# ── augmentation ──────────────────────────────────────────────────────────────
def augment(x):
    B, P, N, H, W = x.shape; d = x.device
    if torch.rand(1).item() < 0.5: x = x.flip(-1)       # h-flip
    if torch.rand(1).item() < 0.5: x = x.flip(2)        # slice-reverse
    th = (torch.rand(B, device=d) - 0.5) * 0.35
    sc = 1 + (torch.rand(B, device=d) - 0.5) * 0.3
    t  = (torch.rand(B, 2, device=d) - 0.5) * 0.16
    A  = torch.stack([torch.stack([sc*th.cos(), -sc*th.sin(), t[:,0]], 1),
                      torch.stack([sc*th.sin(),  sc*th.cos(), t[:,1]], 1)], 1)
    v = x.reshape(B, P*N, H, W)
    v = F.grid_sample(v, F.affine_grid(A, v.shape, align_corners=False), align_corners=False)
    gam = torch.exp((torch.rand(B, 1, 1, 1, device=d) - 0.5) * 0.6)
    v = 255 * (v.clamp(0, 255) / 255) ** gam
    # random erasing (30 %)
    if torch.rand(1).item() < 0.3:
        for b in range(B):
            pi = torch.randint(0, P, (1,)).item()
            eh = torch.randint(H//8, H//4, (1,)).item()
            ew = torch.randint(W//8, W//4, (1,)).item()
            si = torch.randint(0, H - eh, (1,)).item()
            sj = torch.randint(0, W - ew, (1,)).item()
            v[b, pi*N:(pi+1)*N, si:si+eh, sj:sj+ew] = 0
    return v.reshape(B, P, N, H, W)

# ── build ─────────────────────────────────────────────────────────────────────
model = NetV2(CFG['arch'], ns=NS).cuda()
if CFG['grad_ckpt']: model.enc.set_grad_checkpointing(True)
dp    = nn.DataParallel(model)
ema   = EMA(model, CFG['ema'])

def enc_depth(name, n_layers):
    """Layer id of an encoder param: 0 = stem/patch-embed, n_layers+1 = final norm."""
    parts = name.split('.')
    if parts[0] in ('blocks', 'stages') and len(parts) > 1 and parts[1].isdigit():
        return int(parts[1]) + 1
    if parts[0] in ('norm', 'fc_norm', 'head', 'head_norm', 'norm_pre'):
        return n_layers + 1
    return 0

def param_groups(model, lr, wd, decay):
    enc = model.enc
    n_layers = len(enc.blocks) if hasattr(enc, 'blocks') else len(getattr(enc, 'stages', [])) or 1
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        no_wd = p.ndim <= 1 or 'token' in n or n in ('plane_emb', 'pos_emb')
        if n.startswith('enc.'):
            lid = enc_depth(n[4:], n_layers)
            scale = decay ** (n_layers + 1 - lid)
            key = f'enc{lid}_{int(no_wd)}'
        else:
            scale = 3.0; key = f'head_{int(no_wd)}'   # randomly initialised head learns faster
        g = groups.setdefault(key, {'params': [], 'lr': lr * scale, 'weight_decay': 0.0 if no_wd else wd})
        g['params'].append(p)
    return list(groups.values())

opt   = torch.optim.AdamW(param_groups(model, CFG['lr'], CFG['wd'], CFG['layer_decay']))
total_steps = CFG['epochs'] * (len(trn) // (CFG['bs'] * CFG['accum']))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s+1)/300) * 0.5 * (1 + math.cos(math.pi * min(1, s / max(total_steps, 1)))))
scaler = torch.amp.GradScaler(enabled=AMP_DTYPE == torch.float16)

swa_model = None; swa_n = 0; swa_ep = int(CFG['epochs'] * CFG['swa_pct'])

def batches(ids, bs, shuffle):
    ids = np.random.permutation(ids) if shuffle else ids
    for i in range(0, len(ids) - (bs - 1 if shuffle else 0), bs):
        j = np.sort(ids[i:i+bs])
        yield j, torch.from_numpy(X[j]).cuda(non_blocking=True)

@torch.no_grad()
def predict(ids, net):
    net.eval(); P = []
    for j, xb in batches(ids, 12, False):
        with torch.autocast('cuda', dtype=AMP_DTYPE):
            p  = torch.sigmoid(net(xb).float())
            p += torch.sigmoid(net(xb.flip(-1)).float())     # h-flip
            p += torch.sigmoid(net(xb.flip(2)).float())      # slice-reverse
        P.append((p / 3).cpu().numpy())
    return np.concatenate(P)

def calc_aucs(y, p):
    return np.array([roc_auc_score((y[:,c]>0.5).astype(int), p[:,c])
                     if 0 < (y[:,c]>0.5).sum() < len(y) else np.nan for c in range(12)])

# ── train loop ────────────────────────────────────────────────────────────────
best = -1; step = 0
for ep in range(CFG['epochs']):
    dp.train(); tl = 0; n = 0; te = time.time()
    for j, xb in batches(trn, CFG['bs'], True):
        xb = augment(xb.float())
        yb = torch.from_numpy(Y_s[j]).cuda()
        if CFG['mixup'] > 0 and ep < CFG['epochs'] - 2:
            xb, yb = mixup(xb, yb, CFG['mixup'])
        with torch.autocast('cuda', dtype=AMP_DTYPE):
            logits, aux = dp(xb, return_aux=True)
            lo = focal_bce(logits.float(), yb, CFG['focal'], cls_w)
            # auxiliary loss: each plane predicts all 12 targets
            aux_lo = F.binary_cross_entropy_with_logits(
                aux.float(), yb.unsqueeze(1).expand_as(aux.float()))
            lo = (lo + 0.3 * aux_lo) / CFG['accum']
        scaler.scale(lo).backward()
        if (n + 1) % CFG['accum'] == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            sched.step(); step += 1; ema.update(model)
        tl += lo.item() * CFG['accum']; n += 1
        if n % 100 == 0:
            print(f'  step {n} loss {tl/n:.4f} lr {max(g["lr"] for g in opt.param_groups):.2e} {time.time()-te:.0f}s', flush=True)

    # SWA
    if ep >= swa_ep:
        if swa_model is None:
            swa_model = deepcopy(model); swa_n = 1
        else:
            swa_n += 1
            for sp, p in zip(swa_model.parameters(), model.parameters()):
                sp.data.add_((p.data - sp.data) / swa_n)

    # validate with EMA
    ema_dp = nn.DataParallel(ema.shadow)
    p = predict(va, ema_dp)
    gi = gmask[va]; a_gold = calc_aucs(Y[va[gi]], p[gi])
    a_llm  = calc_aucs(Y[va[~gi]], p[~gi]) if (~gi).sum() > 0 else np.full(12, np.nan)
    score  = np.nanmean(a_gold) if gi.sum() > 50 else np.nanmean(a_llm)
    print(f'ep{ep} loss {tl/n:.4f}  gold {np.nanmean(a_gold):.4f}  llm {np.nanmean(a_llm):.4f}  '
          f'{time.time()-te:.0f}s  total {time.time()-t0:.0f}s', flush=True)
    print('  gold/class:', np.round(a_gold, 3).tolist(), flush=True)
    if CFG['full'] or score > best:
        best = score
        tag = f'{ARCH_TAG}_f{CFG["fold"]}_s{CFG["seed"]}{"_full" if CFG["full"] else ""}'
        torch.save({k: v.half() for k, v in ema.shadow.state_dict().items()},
                   f'/kaggle/working/model_{tag}.pt')

# save SWA model too
if swa_model is not None:
    tag = f'{ARCH_TAG}_f{CFG["fold"]}_s{CFG["seed"]}_swa{"_full" if CFG["full"] else ""}'
    torch.save({k: v.half() for k, v in swa_model.state_dict().items()},
               f'/kaggle/working/model_{tag}.pt')

print(f'best {best:.4f}  done {time.time()-t0:.0f}s')
