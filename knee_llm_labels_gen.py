# Report -> 12 soft labels with an open multilingual LLM (Qwen2.5-7B-Instruct), validated on the gold-labeled studies.
import os, time, re, numpy as np, pandas as pd, torch
os.environ['HF_HOME']='/tmp/hf'
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
D='/kaggle/input/competitions/rsna-knee-abnormality-detection'
L=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
KEYS=['ACL','MCL','MedialMeniscus','LateralMeniscus','MedialOA','LateralOA','PFOA','Effusion','Synovitis','Bakers','Contusion','Fracture']
tr=pd.read_csv(D+'/train.csv')
gold=tr[L].notna().all(axis=1).values
SYS="""You are a musculoskeletal radiologist. Read a knee MRI report (any language) and decide if each finding is PRESENT (1) or ABSENT (0) in the knee described.
Definitions:
ACL: anterior cruciate ligament tear of any kind (complete, partial, high-grade, interstitial tear, rupture, grade 2-3 sprain, non-visualized/discontinuous fibers, graft tear). Mucoid degeneration or grade 1 sprain alone = 0.
MCL: medial collateral ligament grade 2-3 sprain or partial/complete tear. Grade 1 sprain or periligamentous edema alone = 0.
MedialMeniscus: tear of the medial meniscus (any tear type: horizontal, radial, complex, root, bucket-handle, degenerative tear, maceration, signal reaching the articular surface). Intrasubstance/grade 1-2 degeneration without tear = 0.
LateralMeniscus: same rule for the lateral meniscus.
MedialOA: osteoarthritis of the medial tibiofemoral compartment: osteophytes, joint space narrowing, or cartilage loss/defect/ulcer/fissuring of grade 2 or worse (partial or full thickness) on medial femoral condyle or medial tibial plateau. Superficial grade 1 chondropathy alone = 0.
LateralOA: same rule for the lateral tibiofemoral compartment (lateral femoral condyle, lateral tibial plateau).
PFOA: same rule for the patellofemoral compartment (patella, trochlea), including chondromalacia patellae grade 2+.
Effusion: joint effusion or hemarthrosis/lipohemarthrosis that the report says is present (small, mild, moderate or large).
Synovitis: synovitis, synovial thickening/hypertrophy/proliferation, Hoffa fat pad edema/inflammation (hoffitis).
Bakers: Baker's (popliteal) cyst of any size, including ruptured. Ganglion or parameniscal cysts alone = 0.
Contusion: bone contusion / bone bruise / post-traumatic bone marrow edema. Purely degenerative subchondral edema = 0.
Fracture: any fracture: acute, subchondral insufficiency fracture, SONK, osteochondral fracture/impaction, avulsion, tibial plateau or patellar fracture.
If a finding is not mentioned, it is 0. Answer ONLY with the 12 lines in this exact format:
ACL=0
MCL=0
MedialMeniscus=0
LateralMeniscus=0
MedialOA=0
LateralOA=0
PFOA=0
Effusion=0
Synovitis=0
Bakers=0
Contusion=0
Fracture=0"""
MID='Qwen/Qwen2.5-7B-Instruct'
tok=AutoTokenizer.from_pretrained(MID); tok.padding_side='left'
t0=time.time()
model=AutoModelForCausalLM.from_pretrained(MID,dtype=torch.float16,device_map='auto'); model.eval()
print('model loaded',time.time()-t0,flush=True)
id0=tok.convert_tokens_to_ids('0'); id1=tok.convert_tokens_to_ids('1')
def clip(rep):
    rep=' '.join(str(rep).split())
    return rep if len(rep)<=2600 else rep[:1500]+' ... '+rep[-1100:]
def build(rep):
    m=[{'role':'system','content':SYS},{'role':'user','content':'REPORT:\n'+clip(rep)}]
    return tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
@torch.no_grad()
def run(reps,budget=7000):
    out=np.full((len(reps),12),np.nan,np.float32)
    ids=[tok(build(r),add_special_tokens=False).input_ids for r in reps]
    order=np.argsort([len(x) for x in ids])[::-1]; b=0; nb=0
    while b<len(order):
        mx=len(ids[order[b]]); bs=max(1,min(24,budget//mx)); idx=order[b:b+bs]; b+=bs
        T=max(len(ids[i]) for i in idx)
        inp=torch.full((len(idx),T),tok.pad_token_id,dtype=torch.long); att=torch.zeros((len(idx),T),dtype=torch.long)
        for j,i in enumerate(idx): inp[j,T-len(ids[i]):]=torch.tensor(ids[i]); att[j,T-len(ids[i]):]=1
        g=model.generate(input_ids=inp.to('cuda:0'),attention_mask=att.to('cuda:0'),max_new_tokens=64,do_sample=False,
                         output_logits=True,return_dict_in_generate=True,pad_token_id=tok.pad_token_id)
        seq=g.sequences[:,T:].cpu(); lg=torch.stack([l[:,[id0,id1]].float() for l in g.logits],1).cpu()
        for j,i in enumerate(idx):
            text=''; k=0
            for t in range(seq.shape[1]):
                tid=int(seq[j,t])
                if tid in (id0,id1) and text.rstrip().endswith('=') and k<12:
                    out[i,k]=torch.softmax(lg[j,t],0)[1].item(); k+=1
                text+=tok.decode([tid])
        nb+=1
        if nb%40==0: print(b,len(order),round(time.time()-t0),flush=True)
    return out
gi=np.where(gold)[0]
pg=run(tr.Report.values[gi].tolist())
Y=tr.loc[gold,L].values
aucs=[]
for c in range(12):
    ok=~np.isnan(pg[:,c])
    a=roc_auc_score(Y[ok,c],pg[ok,c]) if len(set(Y[ok,c]))>1 else np.nan; aucs.append(a)
    print(f'{L[c]:18s} auc={a:.3f} acc={((pg[ok,c]>0.5)==Y[ok,c]).mean():.3f} pos_rate_pred={(pg[ok,c]>0.5).mean():.2f} gold={Y[:,c].mean():.2f} nan={(~ok).sum()}',flush=True)
print('GOLD mean AUC',np.nanmean(aucs),flush=True)
P=run(tr.Report.values.tolist())
df=pd.DataFrame(P,columns=L); df.insert(0,'StudyInstanceUID',tr.StudyInstanceUID.values)
df.to_csv('/kaggle/working/llm_labels_gen.csv',index=False)
print('nan rows',df[L].isna().any(axis=1).sum()); print(df[L].mean().round(3)); print('done',time.time()-t0)
