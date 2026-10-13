# rsnadetection

Kaggle [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection):
12-label multi-plane knee MRI classifier (2.5-D encoder + slice attention + cross-plane attention), trained on
report-derived soft labels and the radiologist gold subset, scored by mean per-label AUC.

## Pipeline

| Step | Script | Kaggle notebook settings | Output |
|---|---|---|---|
| 1. Report → soft labels | `knee_llm_labels.py`, `knee_llm_labels_gen.py` | GPU, internet on | `llm_labels.csv`, `llm_labels_gen.csv` |
| 2. DICOM → uint8 volumes | `knee_prep.py` | CPU | `train_u8.npy`, `train_index.csv` |
| 3. Train backbones | `knee_train_v2.py` | GPU T4×2, internet on (pretrained weights) | `model_<arch>_f<fold>_s<seed>[_swa].pt` |
| 4. Submit | `knee_infer_v2.py` | GPU, **internet off** | `submission.csv` |

Save the outputs of steps 1–3 as Kaggle datasets and attach them to the next notebook. The scripts find inputs
anywhere under `/kaggle/input/`.

## Ready-to-run Kaggle notebooks

`kaggle/knee_train.ipynb` and `kaggle/knee_submit.ipynb` embed the scripts, so you only upload and *Run All*:

1. **Train:** go to Kaggle → *Create → New Notebook → File → Import Notebook* and pick `kaggle/knee_train.ipynb`.
   Attach the competition data and your LLM-label dataset (plus `train_u8.npy` if you have it). Set GPU T4×2 with
   internet on, edit `RUNS`, then *Save & Run All*.
2. **Save the models:** from the finished version's *Output* tab, create a dataset (for example `knee-models`).
3. **Submit:** import `kaggle/knee_submit.ipynb`, attach the competition data and `knee-models`, set internet off,
   *Save & Run All*, then *Submit to Competition*.

After editing any `knee_*.py`, regenerate the notebooks with `python kaggle/build_notebooks.py`.

## DINOv3 backbones

`knee_train_v2.py` accepts any timm backbone through `ARCH`, including the self-supervised DINOv3 weights, which timm
serves ungated:

```bash
# ViT-S/16: fast, strong slice features (gradient checkpointing turns on automatically for ViTs)
ARCH=vit_small_patch16_dinov3.lvd1689m LR=1e-4 LAYER_DECAY=0.8 FOLD=0 python knee_train_v2.py
# ViT-B/16: heavier, use a smaller batch with more accumulation
ARCH=vit_base_patch16_dinov3.lvd1689m  LR=8e-5 LAYER_DECAY=0.8 BS=4 ACCUM=3 FOLD=1 python knee_train_v2.py
# ConvNeXt-T distilled from DINOv3
ARCH=convnext_tiny.dinov3_lvd1689m     LR=1.5e-4 LAYER_DECAY=0.85 FOLD=2 python knee_train_v2.py
```

Training options (environment variables):

- `LAYER_DECAY`: layer-wise LR decay for the encoder. It defaults to 0.8 for DINOv3 backbones and 1.0 (off) for
  everything else.
- `GRAD_CKPT`: `1` or `0`. It defaults to on for ViTs.
- `AMP`: `fp16` (default; T4/P100) or `bf16` (A100/H100).
- `EPOCHS`, `BS`, `ACCUM`, `LR`, `FOLD`, `NFOLD`, `SEED`, `FULL`, `MIXUP`, `FOCAL`, `EMA_DECAY`, `SWA_PCT`.

Checkpoints store the full timm name (dots become `-`, for example
`model_vit_small_patch16_dinov3-lvd1689m_f0_s42.pt`), so `knee_infer_v2.py` rebuilds the right architecture
offline without a lookup table.

## Submission

1. Attach the competition data and a dataset that holds every `model_*.pt` you want in the ensemble.
2. Run `knee_infer_v2.py` with internet off. It loads every checkpoint it finds, applies 4-way TTA (h-flip ×
   slice-reverse), averages, and writes `/kaggle/working/submission.csv`.
3. `ENSEMBLE=rank` averages per-label ranks instead of probabilities. This only affects AUC when the members are
   calibrated differently, so it is worth trying when you mix CNN and DINOv3 ViT models.

To push past the current CV, ensemble diverse members: different backbones (DINOv3 ViT-S/B, ConvNeXt-T DINOv3,
ConvNeXtV2-nano, EfficientNet-B3), folds and seeds. Keep only the members that improve the gold-set AUC printed during
training.
