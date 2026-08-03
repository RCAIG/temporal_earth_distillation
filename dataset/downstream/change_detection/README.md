# Change-Detection Datasets

This folder contains the downstream event-localization datasets used to evaluate frozen TED trajectories and patch tokens.

## Files

| Dataset | Event NPZ | HLS composite |
|---------|-----------|---------------|
| Hansen | `hansen_hls_change_detection.npz` | `hls_composite_nc/hansen_hls_change_detection_processed.nc` |
| Wildfire | `wildfire_hls_change_detection.npz` | `hls_composite_nc/wildfire_hls_change_detection_processed.nc` |
| LCMAP-C | `lcmap_c_hls_change_detection.npz` | `hls_composite_nc/lcmap_c_hls_change_detection_processed.nc` |
| LandTD | `landtd_hls_change_detection.npz` | `hls_composite_nc/landtd_hls_change_detection_processed.nc` |

## Notes

- `LandTD` denotes the manually reviewed land-transition diagnostic dataset used in the paper.
- `manifest.json` records the current paper-aligned paths and target fields. Example visualizations are shown in `../../../docs/downstream_datasets.md`.