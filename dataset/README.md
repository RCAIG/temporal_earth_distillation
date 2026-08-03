# Datasets

## Pre-training HLS Data

Pre-training files are not bundled. Point `--root_path` / `ROOT_PATH` at a directory containing:

- `train.nc` or `train.h5` (required)
- `val.nc` or `val.h5` (optional)

Files may be classic NetCDF4 or HDF5-backed `.nc` files.

## Downstream Evaluation Data

Paper-aligned downstream artifacts are bundled under `dataset/downstream/`. See `../docs/downstream_datasets.md` for the paper dataset table and example visualizations.

These artifacts accompany an unpublished manuscript and are provided for review and reproducibility. Source datasets retain their own licenses and citation requirements; see `../NOTICE.md`.

### Classification

Location: `dataset/downstream/classification/`

- HLS time-series composites: `hls_composite_nc/*_hls_classification_processed.nc`
- AlphaEarth comparison features: `*_alphaearth_classification.npz`
- Datasets: LCMAP, GlanCE, GlobalTree, CDL and CropHarvest

```bash
python eval_downstream.py --downstream_data_root ./dataset/downstream/classification ...
```

### Change Detection

Location: `dataset/downstream/change_detection/`

- Event metadata and labels: `*_hls_change_detection.npz`
- HLS time-series composites: `hls_composite_nc/*_hls_change_detection_processed.nc`
- Datasets: Hansen, Wildfire, LCMAP-C and LandTD