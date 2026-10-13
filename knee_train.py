# 2.5D multi-plane knee model: shared 2D CNN over (prev,cur,next) slice triplets, attention pooling over slices,
# fusion across sagittal/coronal/axial, 12 sigmoid outputs trained on report-derived soft labels.
import os, glob, time, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
from sklearn.metrics import roc_auc_score
torch.backends.cudnn.benchmark=True
L=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
def find(name):
    for pat in [f'/kaggle/input/*/{name}',f'/kaggle/input/*/*/{name}',f'/kaggle/input/*/*/*/{name}',f'/kaggle/input/*/*/*/*/{name}']:
        g=glob.glob(pat)
        if g: return g[0]
    raise FileNotFoundError(name)
D='/kaggle/input/competitions/rsna-knee-abnormality-detection'
CFG=dict(arch=os.environ.get('ARCH','resnet34d.ra2_in1k'),epochs=int(os.environ.get('EPOCHS',12)),bs=8,lr=3e-4,wd=0.05,
         fold=int(os.environ.get('FOLD',0)),nfold=8,seed=int(os.environ.get('SEED',42)),size=224,full=int(os.environ.get('FULL',0)))
print(CFG,flush=True)
t0=time.time()
X=np.load(find('train_u8.npy')); idx=pd.read_csv(find('train_index.csv'))  # full read into RAM
print('X',X.shape,round(time.time()-t0),flush=True)
tr=pd.read_csv(D+'/train.csv'); Ys=[]
for nm in ['llm_labels.csv','llm_labels_gen.csv']:
    try:
        l=idx.merge(pd.read_csv(find(nm)),on='StudyInstanceUID',how='left'); Ys.append(l[L].values.astype(np.float32)); print('labels',nm,flush=True)
    except FileNotFoundError: pass
Y=np.nanmean(np.stack(Ys),0); Y=np.nan_to_num(Y,nan=0.0)
gold=tr.set_index('StudyInstanceUID').loc[idx.StudyInstanceUID,L]
gmask=gold.notna().all(axis=1).values; G=gold.values.astype(np.float32)
Y[gmask]=G[gmask]  # trust radiologist labels where available
rng=np.random.RandomState(0); fold_of=rng.randint(0,CFG['nfold'],len(Y))
va=np.where((fold_of==CFG['fold'])|gmask)[0]
trn=np.arange(len(Y)) if CFG['full'] else np.where((fold_of!=CFG['fold'])&~gmask)[0]
print('train',len(trn),'val',len(va),'gold',gmask.sum(),flush=True)
torch.manual_seed(CFG['seed']); np.random.seed(CFG['seed'])
class Net(nn.Module):
    def __init__(s,arch,pre=True):
        super().__init__()
        s.enc=timm.create_model(arch,pretrained=pre,num_classes=0,in_chans=3)
        c=s.enc.num_features; s.c=c
        s.plane=nn.Parameter(torch.zeros(3,c)); s.pos=nn.Parameter(torch.zeros(1,1,20,c))
        s.att=nn.Sequential(nn.Linear(c,256),nn.Tanh(),nn.Linear(256,1))
        s.head=nn.Sequential(nn.Dropout(0.3),nn.Linear(3*c,512),nn.GELU(),nn.Dropout(0.2),nn.Linear(512,12))
    def forward(s,x):  # x: B,3,NS,H,W uint8->float
      with torch.autocast('cuda',dtype=torch.float16):
        B,P,N,H,W=x.shape
        x=x.float()/255.
        pad=torch.cat([x[:,:,:1],x,x[:,:,-1:]],2)
        trip=torch.stack([pad[:,:,:-2],pad[:,:,1:-1],pad[:,:,2:]],3)  # B,P,N,3,H,W
        trip=(trip-0.45)/0.25
        f=s.enc(trip.reshape(B*P*N,3,H,W)).reshape(B,P,N,s.c)
        f=f+s.pos+s.plane[None,:,None,:]
        a=torch.softmax(s.att(f).squeeze(-1),-1)  # B,P,N
        v=(a.unsqueeze(-1)*f).sum(2)  # B,P,C
        return s.head(v.reshape(B,P*s.c))
def augment(x):  # x: B,3,N,H,W float (0..255) on gpu; vectorized random affine/flip/gamma
    B,P,N,H,W=x.shape; dev=x.device
    fl=torch.rand(B,device=dev)<0.5; x=torch.where(fl[:,None,None,None,None],x.flip(-1),x)
    rv=torch.rand(B,device=dev)<0.5; x=torch.where(rv[:,None,None,None,None],x.flip(2),x)
    th=(torch.rand(B,device=dev)-0.5)*0.35; sc=1+(torch.rand(B,device=dev)-0.5)*0.25; t=(torch.rand(B,2,device=dev)-0.5)*0.16
    A=torch.stack([torch.stack([sc*torch.cos(th),-sc*torch.sin(th),t[:,0]],1),torch.stack([sc*torch.sin(th),sc*torch.cos(th),t[:,1]],1)],1)
    v=x.reshape(B,P*N,H,W); g=F.affine_grid(A,v.shape,align_corners=False); v=F.grid_sample(v,g,align_corners=False)
    gam=torch.exp((torch.rand(B,1,1,1,device=dev)-0.5)*0.5); v=255*(v.clamp(0,255)/255)**gam
    return v.reshape(B,P,N,H,W)
model=Net(CFG['arch']).cuda()
dp=nn.DataParallel(model)
opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['wd'])
steps=CFG['epochs']*(len(trn)//CFG['bs']); warm=300
sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s: min(1,(s+1)/warm)*0.5*(1+math.cos(math.pi*min(1,s/steps))))
scaler=torch.amp.GradScaler()
def batches(ids,bs,shuffle):
    ids=np.random.permutation(ids) if shuffle else ids
    for i in range(0,len(ids)-(bs-1 if shuffle else 0),bs):
        j=np.sort(ids[i:i+bs]); yield j, torch.from_numpy(X[j]).cuda(non_blocking=True)
@torch.no_grad()
def predict(ids):
    dp.eval(); P=[]
    for j,xb in batches(ids,16,False):
        with torch.autocast('cuda',dtype=torch.float16):
            p=torch.sigmoid(dp(xb).float())+torch.sigmoid(dp(xb.flip(-1)).float())
        P.append((p/2).cpu().numpy())
    return np.concatenate(P)
def aucs(y,p):
    r=[]
    for c in range(12):
        yy=(y[:,c]>0.5).astype(int)
        r.append(roc_auc_score(yy,p[:,c]) if 0<yy.sum()<len(yy) else np.nan)
    return np.array(r)
best=-1; step=0
for ep in range(CFG['epochs']):
    dp.train(); tl=0; n=0; te=time.time()
    for j,xb in batches(trn,CFG['bs'],True):
        xb=augment(xb.float()); yb=torch.from_numpy(Y[j]).cuda()
        with torch.autocast('cuda',dtype=torch.float16):
            lo=F.binary_cross_entropy_with_logits(dp(xb).float(),yb)
        opt.zero_grad(set_to_none=True); scaler.scale(lo).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(),2.0); scaler.step(opt); scaler.update(); sched.step(); step+=1
        tl+=lo.item(); n+=1
        if n%100==0: print(f'  step {n} loss {tl/n:.4f} {time.time()-te:.0f}s',flush=True)
    p=predict(va); vf=va[~gmask[va]]; a_llm=aucs(Y[vf],p[~gmask[va]]); a_gold=aucs(Y[va[gmask[va]]],p[gmask[va]])
    score=np.nanmean(a_llm)
    print(f'ep{ep} loss {tl/n:.4f} valAUC(llm) {score:.4f} goldAUC {np.nanmean(a_gold):.4f} time {time.time()-te:.0f}s total {time.time()-t0:.0f}s',flush=True)
    print('  per-class llm:',np.round(a_llm,3).tolist(),flush=True)
    if CFG['full'] or score>best:
        best=score; torch.save({k:v.half() for k,v in model.state_dict().items()},f'/kaggle/working/model_{CFG["arch"].split(".")[0]}_f{CFG["fold"]}_s{CFG["seed"]}{"_full" if CFG["full"] else ""}.pt')
print('best',best,'done',time.time()-t0)
