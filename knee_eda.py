import pandas as pd, textwrap
D='/kaggle/input/competitions/rsna-knee-abnormality-detection'
tr=pd.read_csv(D+'/train.csv')
L=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
lab=tr[tr[L].notna().all(axis=1)].reset_index(drop=True)
for i,r in lab.iterrows():
    pos=[c for c in L if r[c]==1]
    print(f"### {i} POS={pos}")
    print(' '.join(str(r.Report).split())[:1800])
