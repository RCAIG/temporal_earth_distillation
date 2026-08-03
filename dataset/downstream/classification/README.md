# Classification Datasets

This folder contains the five classification datasets used for downstream evaluation in the TED paper.

## Files

| Dataset | HLS time-series composite | AlphaEarth features |
|---------|---------------------------|---------------------|
| LCMAP | `hls_composite_nc/lcmap_hls_classification_processed.nc` | `lcmap_alphaearth_classification.npz` |
| GlanCE | `hls_composite_nc/glance_hls_classification_processed.nc` | `glance_alphaearth_classification.npz` |
| GlobalTree | `hls_composite_nc/globaltree_hls_classification_processed.nc` | `globaltree_alphaearth_classification.npz` |
| CDL | `hls_composite_nc/cdl_hls_classification_processed.nc` | `cdl_alphaearth_classification.npz` |
| CropHarvest | `hls_composite_nc/cropharvest_hls_classification_processed.nc` | `cropharvest_alphaearth_classification.npz` |

## Notes

- HLS composites are the pixel-based satellite image time series used by TED/MSM/NTP encoders.
- AlphaEarth NPZ files store the spatial-multimodal baseline features used for comparison and fusion analyses.
- `manifest.json` records the current paper-aligned paths.