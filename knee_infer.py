# Inference: preprocess hidden test studies exactly as in training, run the trained 2.5D models (+hflip TTA), write submission.csv
# Preprocess: per study pick one series per plane (prefer fluid-sensitive), sort slices spatially,
# sample D slices, robust-normalize to uint8, resize to S x S. Writes a uint8 memmap + index.
import os, time, numpy as np, pandas as pd, pydicom, cv2
from multiprocessing import Pool
D='/kaggle/input/competitions/rsna-knee-abnormality-detection'
SPLIT='test'; NS=20; S=224; PLANES=['Sagittal','Coronal','Axial']
ts=pd.read_csv(f'{D}/{SPLIT}_series.csv'); st=pd.read_csv(f'{D}/{SPLIT}.csv')
def pick(g):
    out={}
    for p in PLANES:
        c=g[g.Anatomical_Plane==p]
        if len(c)==0: out[p]=None; continue
        c=c.assign(n=[len(os.listdir(f'{D}/{SPLIT}_series/{a}/{b}')) for a,b in zip(c.StudyInstanceUID,c.SeriesInstanceUID)])
        c=c.sort_values(['Fluid_Sensitive','Fat_Suppression','n'],ascending=False)
        out[p]=c.SeriesInstanceUID.iloc[0]
    return out
def load_series(path):
    fs=os.listdir(path); hs=[]
    for f in fs:
        try:
            d=pydicom.dcmread(f'{path}/{f}',stop_before_pixels=True)
            ipp=np.array(d.ImagePositionPatient,float); iop=np.array(d.ImageOrientationPatient,float)
            nrm=np.cross(iop[:3],iop[3:]); key=float(ipp@nrm)
        except Exception:
            key=float(getattr(d,'InstanceNumber',0)) if 'd' in dir() else 0.0
        hs.append((key,f))
    hs.sort(); fs=[f for _,f in hs]; n=len(fs)
    idx=np.unique(np.round(np.linspace(0,n-1,NS)).astype(int)) if n>0 else []
    imgs=[]
    for i in idx:
        try:
            a=pydicom.dcmread(f'{path}/{fs[i]}').pixel_array.astype(np.float32)
            if a.ndim==3: a=a[...,0] if a.shape[-1] in (3,4) else a[0]
            imgs.append(cv2.resize(a,(S,S),interpolation=cv2.INTER_AREA))
        except Exception: pass
    if not imgs: return np.zeros((NS,S,S),np.uint8)
    v=np.stack(imgs); lo,hi=np.percentile(v,0.5),np.percentile(v,99.5)
    v=np.clip((v-lo)/max(hi-lo,1e-3),0,1)*255
    if len(v)<NS: v=v[np.round(np.linspace(0,len(v)-1,NS)).astype(int)]
    return v.astype(np.uint8)
def work(args):
    i,sid,ser=args; out=np.zeros((3,NS,S,S),np.uint8)
    for k,p in enumerate(PLANES):
        if ser.get(p) is not None:
            try: out[k]=load_series(f'{D}/{SPLIT}_series/{sid}/{ser[p]}')
            except Exception as e: print('ERR',sid,p,e,flush=True)
    return i,out

import glob, torch, torch.nn as nn, torch.nn.functional as F, timm
L=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
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
if __name__=='__main__':
    t0=time.time(); uids=st.StudyInstanceUID.tolist(); G=dict(list(ts.groupby('StudyInstanceUID')))
    jobs=[(i,u,pick(G[u])) if u in G else (i,u,{}) for i,u in enumerate(uids)]
    X=np.zeros((len(uids),3,NS,S,S),np.uint8)
    with Pool(4) as pool:
        for i,a in pool.imap_unordered(work,jobs,chunksize=2): X[i]=a
    print('prep done',len(uids),round(time.time()-t0),flush=True)
    paths=sorted(glob.glob('/kaggle/input/**/model_*.pt',recursive=False) or glob.glob('/kaggle/input/*/*/model_*.pt') or glob.glob('/kaggle/input/*/*/*/model_*.pt') or glob.glob('/kaggle/input/*/*/*/*/model_*.pt'))
    print(paths,flush=True)
    P=np.zeros((len(uids),12),np.float64); nm=0
    for pth in paths:
        arch=os.path.basename(pth)[6:].split('_f')[0]
        arch={'convnext_nano':'convnext_nano.in12k_ft_in1k','resnet18d':'resnet18d.ra2_in1k','resnet34d':'resnet34d.ra2_in1k'}.get(arch,arch)
        m=Net(arch,pre=False); m.load_state_dict({k:v.float() for k,v in torch.load(pth,map_location='cpu').items()}); m=m.cuda().eval()
        dp=nn.DataParallel(m) if torch.cuda.device_count()>1 else m
        with torch.no_grad():
            for b in range(0,len(uids),8):
                xb=torch.from_numpy(X[b:b+8]).cuda()
                with torch.autocast('cuda',dtype=torch.float16):
                    p=torch.sigmoid(dp(xb).float())+torch.sigmoid(dp(xb.flip(-1)).float())
                P[b:b+8]+=(p/2).cpu().numpy()
        nm+=1; print('model',pth,round(time.time()-t0),flush=True)
    P/=max(nm,1)
    sub=pd.DataFrame(P,columns=L); sub.insert(0,'StudyInstanceUID',uids)
    sub.to_csv('/kaggle/working/submission.csv',index=False); print(sub.head()); print('done',time.time()-t0)
