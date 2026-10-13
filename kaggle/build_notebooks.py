# Builds self-contained Kaggle notebooks from the repo scripts (each script is embedded with %%writefile,
# so the notebooks need no git clone). Re-run after editing any knee_*.py:  python kaggle/build_notebooks.py
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'kaggle')

def md(text): return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(True)}
def code(text): return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                        'source': text.strip('\n').splitlines(True)}
def embed(name): return code(f'%%writefile {name}\n' + open(os.path.join(ROOT, name)).read())

def save(fname, cells):
    nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                       'language_info': {'name': 'python'}}}
    with open(os.path.join(OUT, fname), 'w') as f: json.dump(nb, f, indent=1)
    print('wrote', fname)

save('knee_train.ipynb', [
    md('''
# RSNA Knee: train DINOv3 / CNN backbones
**Settings:** Accelerator *GPU T4 x2*, Internet *on* (downloads pretrained timm weights).
**Inputs:** the competition data, plus a dataset holding `llm_labels.csv` / `llm_labels_gen.csv` and,
if you already have them, `train_u8.npy` + `train_index.csv` (otherwise step 1 builds them, which takes a few hours).

Edit `RUNS` below, then *Save Version → Save & Run All*. Afterwards, create a dataset from this notebook's output
(`model_*.pt`) and attach it to `knee_submit.ipynb`.
'''),
    code('!pip install -q -U "timm>=1.0.20"\nimport timm; print(timm.__version__)'),
    md('## 1. Preprocess (skipped when `train_u8.npy` is already attached)'),
    embed('knee_prep.py'),
    code('''
import glob, subprocess
if not glob.glob('/kaggle/input/**/train_u8.npy', recursive=True):
    subprocess.run(['python', 'knee_prep.py'], check=True)   # writes /kaggle/working/train_u8.npy
else:
    print('using attached train_u8.npy')
'''),
    md('## 2. Train'),
    embed('knee_train_v2.py'),
    code('''
import os, glob, subprocess, time
# Each entry is one run. ~12 h session budget: start with 2 runs, check the time per epoch in the log, then add more
# (other folds / seeds) in later notebook versions.
RUNS = [
    dict(ARCH='vit_small_patch16_dinov3.lvd1689m', LR='1e-4',   LAYER_DECAY='0.8',  FOLD='0', EPOCHS='15'),
    dict(ARCH='convnext_tiny.dinov3_lvd1689m',     LR='1.5e-4', LAYER_DECAY='0.85', FOLD='1', EPOCHS='15'),
    # dict(ARCH='vit_base_patch16_dinov3.lvd1689m', LR='8e-5', LAYER_DECAY='0.8', BS='4', ACCUM='3', FOLD='2', EPOCHS='12'),
    # dict(ARCH='convnextv2_nano.fcmae_ft_in22k_in1k', FOLD='3'),
]
for r in RUNS:
    t = time.time(); print('=' * 30, r, flush=True)
    subprocess.run(['python', 'knee_train_v2.py'], env={**os.environ, **r}, check=True)
    print(f'run took {(time.time() - t) / 60:.0f} min', flush=True)
print(sorted(glob.glob('/kaggle/working/model_*.pt')))
'''),
    md('''
If step 1 built `train_u8.npy` here, it is part of this notebook's output. Save it as its own dataset for later
versions, so they can skip preprocessing.
'''),
])

save('knee_submit.ipynb', [
    md('''
# RSNA Knee: submission
**Settings:** Accelerator *GPU*, Internet **off**.
**Inputs:** the competition data, plus the dataset(s) holding the `model_*.pt` checkpoints from `knee_train.ipynb`.
Every attached checkpoint is ensembled with 4-way TTA. Set `ENSEMBLE` to `rank` to average per-label ranks.

Run All, check that `submission.csv` looks right, then **Submit to Competition**.
'''),
    embed('knee_infer_v2.py'),
    code('''
import os, subprocess
subprocess.run(['python', 'knee_infer_v2.py'], env={**os.environ, 'ENSEMBLE': 'mean'}, check=True)
'''),
    code('''
import pandas as pd
sub = pd.read_csv('/kaggle/working/submission.csv')
print(sub.shape); sub.head()
'''),
])
