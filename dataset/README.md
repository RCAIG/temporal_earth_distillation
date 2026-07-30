# Datasets

## Pretraining (HLS SSL)

Not bundled. Point `--root_path` / `ROOT_PATH` at a directory containing:

- `train.nc` or `train.h5` (required)
- `val.nc` / `val.h5` (optional)

Files may be classic NetCDF4 or HDF5-backed `.nc` (opened via `h5py`).

## Downstream classification

Bundled under `downstream/classification/`:

- `hls_composite_nc/*_hls_classification_processed.nc` — probe inputs
- `*_hls_classification.npz` — lon/lat helpers for spatial splits
- `manifest.json`

```bash
python eval_downstream.py --downstream_data_root ./dataset/downstream/classification ...
```
