# Preprocess: per study pick one series per plane (prefer fluid-sensitive), sort slices spatially,
# sample D slices, robust-normalize to uint8, resize to S x S. Writes a uint8 memmap + index.
import os, time, numpy as np, pandas as pd, pydicom, cv2
from multiprocessing import Pool
D='/kaggle/input/competitions/rsna-knee-abnormality-detection'
SPLIT='train'; NS=20; S=224; PLANES=['Sagittal','Coronal','Axial']
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
if __name__=='__main__':
    t0=time.time(); uids=st.StudyInstanceUID.tolist(); G=dict(list(ts.groupby('StudyInstanceUID')))
    jobs=[(i,u,pick(G[u])) for i,u in enumerate(uids)]
    print('picked',time.time()-t0,flush=True)
    X=np.lib.format.open_memmap(f'/kaggle/working/{SPLIT}_u8.npy',mode='w+',dtype=np.uint8,shape=(len(uids),3,NS,S,S))
    with Pool(4) as pool:
        for n,(i,a) in enumerate(pool.imap_unordered(work,jobs,chunksize=2)):
            X[i]=a
            if n%200==0: print(n,len(uids),round(time.time()-t0),flush=True)
    X.flush()
    pd.DataFrame({'StudyInstanceUID':uids}).to_csv(f'/kaggle/working/{SPLIT}_index.csv',index=False)
    m=X[:50].reshape(50,3,-1).mean(-1); print('mean per plane first 50',m.mean(0),'zeros',(m==0).sum(0))
    print('done',time.time()-t0)
