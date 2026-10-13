# V2 inference: multi-architecture ensemble with 4-way TTA
# Supports both v1 (Net) and v2 (NetV2) model checkpoints auto-detected from keys.
import os, re, time, glob, numpy as np, pandas as pd, pydicom, cv2, torch, torch.nn as nn, torch.nn.functional as F, timm
from multiprocessing import Pool

D = '/kaggle/input/competitions/rsna-knee-abnormality-detection'
SPLIT = 'test'; NS = 20; S = 224; PLANES = ['Sagittal', 'Coronal', 'Axial']
ts = pd.read_csv(f'{D}/{SPLIT}_series.csv')
st = pd.read_csv(f'{D}/{SPLIT}.csv')
L = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
     'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

AMP_DTYPE = torch.bfloat16 if os.environ.get('AMP', 'fp16') == 'bf16' else torch.float16
ENSEMBLE = os.environ.get('ENSEMBLE', 'mean')   # 'mean' of probabilities or 'rank' average (AUC-friendly)

# ── preprocessing (identical to v1) ───────────────────────────────────────────
def pick(g):
    out = {}
    for p in PLANES:
        c = g[g.Anatomical_Plane == p]
        if len(c) == 0: out[p] = None; continue
        c = c.assign(n=[len(os.listdir(f'{D}/{SPLIT}_series/{a}/{b}'))
                        for a, b in zip(c.StudyInstanceUID, c.SeriesInstanceUID)])
        c = c.sort_values(['Fluid_Sensitive', 'Fat_Suppression', 'n'], ascending=False)
        out[p] = c.SeriesInstanceUID.iloc[0]
    return out

def load_series(path):
    fs = os.listdir(path); hs = []
    for f in fs:
        try:
            d = pydicom.dcmread(f'{path}/{f}', stop_before_pixels=True)
            ipp = np.array(d.ImagePositionPatient, float)
            iop = np.array(d.ImageOrientationPatient, float)
            nrm = np.cross(iop[:3], iop[3:]); key = float(ipp @ nrm)
        except Exception:
            key = float(getattr(d, 'InstanceNumber', 0)) if 'd' in dir() else 0.0
        hs.append((key, f))
    hs.sort(); fs = [f for _, f in hs]; n = len(fs)
    sel = np.unique(np.round(np.linspace(0, n-1, NS)).astype(int)) if n > 0 else []
    imgs = []
    for i in sel:
        try:
            a = pydicom.dcmread(f'{path}/{fs[i]}').pixel_array.astype(np.float32)
            if a.ndim == 3: a = a[...,0] if a.shape[-1] in (3,4) else a[0]
            imgs.append(cv2.resize(a, (S, S), interpolation=cv2.INTER_AREA))
        except Exception: pass
    if not imgs: return np.zeros((NS, S, S), np.uint8)
    v = np.stack(imgs)
    lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.5)
    v = np.clip((v - lo) / max(hi - lo, 1e-3), 0, 1) * 255
    if len(v) < NS: v = v[np.round(np.linspace(0, len(v)-1, NS)).astype(int)]
    return v.astype(np.uint8)

def work(args):
    i, sid, ser = args
    out = np.zeros((3, NS, S, S), np.uint8)
    for k, p in enumerate(PLANES):
        if ser.get(p) is not None:
            try: out[k] = load_series(f'{D}/{SPLIT}_series/{sid}/{ser[p]}')
            except Exception as e: print('ERR', sid, p, e, flush=True)
    return i, out

# ── Model definitions (must match training) ───────────────────────────────────
class NetV2(nn.Module):
    """V2 model with gated attention, cross-plane self-attention, auxiliary heads."""
    def __init__(s, arch, ns=20, pre=True):
        super().__init__()
        s.enc = timm.create_model(arch, pretrained=pre, num_classes=0, in_chans=3)
        c = s.enc.num_features; s.c = c; s.ns = ns
        s.plane_emb = nn.Parameter(torch.zeros(3, c))
        s.pos_emb   = nn.Parameter(torch.zeros(1, 1, ns, c))
        s.att_v = nn.Sequential(nn.Linear(c, 256), nn.Tanh())
        s.att_u = nn.Sequential(nn.Linear(c, 256), nn.Sigmoid())
        s.att_w = nn.Linear(256, 1)
        s.xp_attn = nn.MultiheadAttention(c, num_heads=min(4, c // 64 or 1), dropout=0.1, batch_first=True)
        s.xp_norm = nn.LayerNorm(c)
        s.head = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(3 * c, 512), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(512, 12))
        s.aux = nn.ModuleList([nn.Linear(c, 12) for _ in range(3)])

    def forward(s, x, return_aux=False):
      with torch.autocast('cuda', dtype=AMP_DTYPE):
        B, P, N, H, W = x.shape
        x = x.float() / 255.0
        pad = torch.cat([x[:,:,:1], x, x[:,:,-1:]], 2)
        trip = torch.stack([pad[:,:,:-2], pad[:,:,1:-1], pad[:,:,2:]], 3)
        trip = (trip - 0.45) / 0.25
        f = s.enc(trip.reshape(B*P*N, 3, H, W)).reshape(B, P, N, s.c)
        pos = s.pos_emb
        if N != s.ns:
            pos = F.interpolate(pos[0].transpose(1, 2), size=N, mode='linear',
                                align_corners=False).transpose(1, 2)[None]
        f = f + pos + s.plane_emb[None,:,None,:]
        h = s.att_v(f) * s.att_u(f)
        a = torch.softmax(s.att_w(h).squeeze(-1), -1)
        v = (a.unsqueeze(-1) * f).sum(2)
        v = s.xp_norm(v + s.xp_attn(v, v, v, need_weights=False)[0])
        main = s.head(v.reshape(B, P * s.c))
        if not return_aux:
            return main
        aux = torch.stack([s.aux[p](v[:, p]) for p in range(P)], 1)
        return main, aux

class Net(nn.Module):
    """V1 model (backward compatibility)."""
    def __init__(s, arch, pre=True):
        super().__init__()
        s.enc = timm.create_model(arch, pretrained=pre, num_classes=0, in_chans=3)
        c = s.enc.num_features; s.c = c
        s.plane = nn.Parameter(torch.zeros(3, c))
        s.pos = nn.Parameter(torch.zeros(1, 1, 20, c))
        s.att = nn.Sequential(nn.Linear(c, 256), nn.Tanh(), nn.Linear(256, 1))
        s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(3*c, 512), nn.GELU(),
                               nn.Dropout(0.2), nn.Linear(512, 12))
    def forward(s, x):
      with torch.autocast('cuda', dtype=AMP_DTYPE):
        B, P, N, H, W = x.shape; x = x.float() / 255.
        pad = torch.cat([x[:,:,:1], x, x[:,:,-1:]], 2)
        trip = torch.stack([pad[:,:,:-2], pad[:,:,1:-1], pad[:,:,2:]], 3)
        trip = (trip - 0.45) / 0.25
        f = s.enc(trip.reshape(B*P*N, 3, H, W)).reshape(B, P, N, s.c)
        f = f + s.pos + s.plane[None,:,None,:]
        a = torch.softmax(s.att(f).squeeze(-1), -1)
        v = (a.unsqueeze(-1) * f).sum(2)
        return s.head(v.reshape(B, P*s.c))

ARCH_MAP = {
    'convnextv2_nano': 'convnextv2_nano.fcmae_ft_in22k_in1k',
    'convnext_nano':   'convnext_nano.in12k_ft_in1k',
    'efficientnet_b3': 'efficientnet_b3.ra2_in1k',
    'resnet34d':       'resnet34d.ra2_in1k',
    'resnet18d':       'resnet18d.ra2_in1k',
    'resnet50d':       'resnet50d.ra2_in1k',
    'convnext_small':  'convnext_small.in12k_ft_in1k',
    'convnextv2_tiny': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
    'efficientnetv2_s':'tf_efficientnetv2_s.in21k_ft_in1k',
    # short aliases for DINOv3 backbones; v2 training now writes the full name (dots -> '-')
    'vit_small_patch16_dinov3': 'vit_small_patch16_dinov3.lvd1689m',
    'vit_base_patch16_dinov3':  'vit_base_patch16_dinov3.lvd1689m',
    'vit_large_patch16_dinov3': 'vit_large_patch16_dinov3.lvd1689m',
}

def arch_of(basename):
    """model_<arch>_f<fold>_s<seed>[_swa][_full].pt -> timm arch name."""
    m = re.match(r'model_(.+?)_f\d+_s\d+', basename)
    key = m.group(1) if m else basename[len('model_'):].split('_f')[0]
    return ARCH_MAP.get(key, key.replace('-', '.'))

def rank01(p):
    """Per-column rank transform to (0, 1]."""
    return (np.argsort(np.argsort(p, 0), 0) + 1) / len(p)

if __name__ == '__main__':
    t0 = time.time()
    uids = st.StudyInstanceUID.tolist()
    G = dict(list(ts.groupby('StudyInstanceUID')))
    jobs = [(i, u, pick(G[u])) if u in G else (i, u, {}) for i, u in enumerate(uids)]

    X = np.zeros((len(uids), 3, NS, S, S), np.uint8)
    with Pool(4) as pool:
        for i, a in pool.imap_unordered(work, jobs, chunksize=2):
            X[i] = a
    print(f'prep done {len(uids)} studies  {time.time()-t0:.0f}s', flush=True)

    # discover all model checkpoints
    paths = []
    for pattern in ['/kaggle/input/**/model_*.pt']:
        paths.extend(glob.glob(pattern, recursive=True))
    paths = sorted(set(paths))
    print(f'found {len(paths)} models: {[os.path.basename(p) for p in paths]}', flush=True)

    P = np.zeros((len(uids), 12), np.float64); nm = 0

    for pth in paths:
        bn = os.path.basename(pth)
        arch = arch_of(bn)

        # detect model version from checkpoint keys
        ckpt = torch.load(pth, map_location='cpu', weights_only=False)
        sd = {k: v.float() for k, v in ckpt.items()}
        is_v2 = 'att_v.0.weight' in sd or 'xp_attn.in_proj_weight' in sd

        try:
            if is_v2:
                m = NetV2(arch, ns=sd['pos_emb'].shape[2] if 'pos_emb' in sd else NS, pre=False)
            else:
                m = Net(arch, pre=False)
            missing, unexpected = m.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f'WARN {bn}: missing {len(missing)} {missing[:3]} unexpected {len(unexpected)} {unexpected[:3]}', flush=True)
            if len(missing) > 0.5 * len(m.state_dict()):
                raise RuntimeError('checkpoint does not match architecture')
        except Exception as e:
            print(f'SKIP {bn}: {e}', flush=True); continue

        m = m.cuda().eval()
        dp_m = nn.DataParallel(m) if torch.cuda.device_count() > 1 else m

        Pm = np.zeros((len(uids), 12), np.float64)
        with torch.no_grad():
            for b in range(0, len(uids), 8):
                xb = torch.from_numpy(X[b:b+8]).cuda()
                with torch.autocast('cuda', dtype=AMP_DTYPE):
                    p  = torch.sigmoid(dp_m(xb).float())
                    p += torch.sigmoid(dp_m(xb.flip(-1)).float())           # h-flip
                    p += torch.sigmoid(dp_m(xb.flip(2)).float())            # slice-reverse
                    p += torch.sigmoid(dp_m(xb.flip(-1).flip(2)).float())   # both
                Pm[b:b+8] = (p / 4).cpu().numpy()
        P += rank01(Pm) if ENSEMBLE == 'rank' and len(uids) > 1 else Pm
        nm += 1
        print(f'model {nm}: {bn} ({arch}) v{"2" if is_v2 else "1"}  {time.time()-t0:.0f}s', flush=True)
        del m, dp_m; torch.cuda.empty_cache()

    P /= max(nm, 1)
    P = np.clip(P, 0.001, 0.999)

    sub = pd.DataFrame(P, columns=L)
    sub.insert(0, 'StudyInstanceUID', uids)
    sub.to_csv('/kaggle/working/submission.csv', index=False)
    print(sub.describe())
    print(f'done  {nm} models ensembled  {time.time()-t0:.0f}s')
