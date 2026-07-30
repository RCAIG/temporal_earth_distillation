# Temporal Earth Distillation (TED)

Self-supervised pretraining for **HLS** time series (**TED**, **MSM**, **NTP**) and paper-aligned downstream land-cover classification evaluation.

## Requirements

Python **≥ 3.10**. Install:

```bash
cd temporal_earth_distillation
pip install -r requirements.txt

# optional: more accurate UTM for spatial 1280 m splits
pip install -r requirements.txt -r requirements-optional.txt
# or: pip install '.[spatial]'
```

| Package | Role |
|---------|------|
| `torch` | training / eval |
| `numpy`, `pandas`, `scipy`, `scikit-learn` | arrays, metrics, probes |
| `h5py`, `netCDF4` | HLS / classification NetCDF (HDF5-backed) |
| `einops` | transformer blocks |
| `matplotlib` | training curves |
| `pyproj` (optional) | UTM projection for spatial splits (approx fallback exists) |

GPU training expects CUDA-enabled PyTorch matching your driver.

## Layout

```text
temporal_earth_distillation/
├── train.py                 # SSL pretraining CLI (TED / MSM / NTP / Imputator)
├── eval_downstream.py       # downstream classification CLI
├── models/                  # ted.py, msm.py, ntp.py, imputator.py
├── modules/                 # Backbone + DINO / Proto heads
├── layers/
├── data/
│   ├── hls_dataset.py       # pretraining loader (HLS only)
│   ├── factory.py
│   ├── downstream_datasets.py
│   └── downstream_factory.py
├── engine/                  # SSLTrainer
├── eval/                    # linear probe, kNN, spatial splits, zoo loader
├── utils/
├── scripts/
│   ├── train_ted_12b768.sh
│   ├── train_msm_*.sh
│   ├── train_ntp_*.sh
│   ├── eval_downstream.sh
│   └── ensure_pretrained_weights.sh
├── pretrained/              # HF-style zoo (config.json + weights)
│   ├── ted-hls-12b768/
│   ├── msm-hls-12b768/
│   └── ntp-hls-12b768/
└── dataset/downstream/classification/
```

## Data

### Pretraining (not bundled)

Place under `--root_path` / `ROOT_PATH` (default `./dataset/`):

- `train.nc` or `train.h5` (required)
- `val.nc` / `val.h5` (optional)

See `dataset/README.md`.

### Downstream (bundled)

Five classification tasks under `dataset/downstream/classification/`:

- `hls_composite_nc/*_hls_classification_processed.nc`
- `*_hls_classification.npz` (lon/lat helpers for spatial splits)

## Train

```bash
pip install -r requirements.txt

ROOT_PATH=/path/to/dir_with_train.nc \
DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
bash scripts/train_ted_12b768.sh

ROOT_PATH=/path/to/dir_with_train.nc bash scripts/train_msm_12b768.sh
ROOT_PATH=/path/to/dir_with_train.nc bash scripts/train_ntp_12b768.sh
```

Single-GPU smoke:

```bash
ROOT_PATH=/path/to/dir_with_train.nc \
DEVICES=0 NPROC_PER_NODE=1 USE_MULTI_GPU=0 \
MAX_TRAIN_STEPS=5 BATCH_SIZE=8 NUM_WORKERS=0 \
bash scripts/train_msm_12b768.sh
```

## Downstream evaluation

Default protocol (paper tables):

- TerraFM-style **2:1:1** split
- **spatial 1280 m** grid + proximity filter
- **cap300** (`max_train_per_class=300`; `0` = nocap)
- **linear** probe (LR grid on val BA) or **kNN**
- MSM pooling: `patch_avg`
- Primary metric: test **balanced accuracy (ba)**

Released anchors live under `pretrained/` (aliases `ted` / `msm` / `ntp`). Weights are gitignored; on this machine they ship as `pretrained/*/pytorch_model.bin`. If missing:

```bash
bash scripts/ensure_pretrained_weights.sh
# optional: PRETRAINED_BUNDLE=/path/to/best_epoch_bundle
```

```bash
CHECKPOINT=ted EVAL_MODE=linear DATASETS=all bash scripts/eval_downstream.sh
CHECKPOINT=msm bash scripts/eval_downstream.sh
CHECKPOINT=ntp bash scripts/eval_downstream.sh

python eval_downstream.py \
  --checkpoint ted-hls-12b768 \
  --eval_mode linear \
  --datasets all \
  --downstream_data_root ./dataset/downstream/classification \
  --spatial_block_m 1280 \
  --max_train_per_class 300 \
  --output_csv results/downstream_ted_linear_cap300.csv
```

Python load:

```python
from eval.load_pretrained import from_pretrained
model, config, family = from_pretrained("ted", device="cuda:0")
```

Details: `pretrained/README.md`.

## Models

| `--model` | Module | Notes |
|-----------|--------|-------|
| `TED` | `models/ted.py` | Temporal Earth Distillation |
| `MSM` | `models/msm.py` | Masked Spectral Modeling |
| `NTP` | `models/ntp.py` | Next-Token Prediction |
| `Imputator` | `models/imputator.py` | optional reconstruction helper for TED |

Legacy name aliases (`TED_modular`, `Patch_Masked`, `Patch_NTP_TED`, `Transformer`) still resolve at train time.

Released zoo recipes: TED **cXattnB** (job 9616), MSM **reg4** (10909), NTP (10685) — see matching `scripts/train_*_12b768.sh`.

## Notes

- `train.py` is train-only (`if __name__ == "__main__"`). This package loads **HLS only** (no ETT / M4 / forecast dataset loaders).
- TED-specific flags default **off**; paper TED recipes set them in `scripts/train_ted_*.sh`.
- License: MIT (`LICENSE`). Citation: `CITATION.cff`.
