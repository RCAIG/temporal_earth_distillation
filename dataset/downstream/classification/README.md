# Downstream classification datasets

Final five-dataset assets used by `eval_downstream.py`.

## Layout

```text
classification/
├── hls_composite_nc/
│   ├── lcmap_hls_classification_processed.nc
│   ├── glancetraining_hls_classification_processed.nc
│   ├── globaltree_hls_classification_processed.nc
│   ├── cdl_hls_classification_processed.nc
│   └── cropharvest_hls_classification_processed.nc
├── *_hls_classification.npz   # lon/lat helpers for spatial splits
├── manifest.json
└── README.md
```

## Datasets

| Dataset | Processed NC |
|---------|--------------|
| LCMAP | `hls_composite_nc/lcmap_hls_classification_processed.nc` |
| GlanceTraining | `hls_composite_nc/glancetraining_hls_classification_processed.nc` |
| GlobalTree | `hls_composite_nc/globaltree_hls_classification_processed.nc` |
| CDL | `hls_composite_nc/cdl_hls_classification_processed.nc` |
| CropHarvest | `hls_composite_nc/cropharvest_hls_classification_processed.nc` |

## Usage

```bash
python eval_downstream.py \
  --downstream_data_root ./dataset/downstream/classification \
  --checkpoint ted \
  --eval_mode linear \
  --datasets all
```
