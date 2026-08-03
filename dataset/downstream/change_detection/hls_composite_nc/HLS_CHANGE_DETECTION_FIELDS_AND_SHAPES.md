# Downstream Change-Detection Fields and Shapes

This folder documents the paper-aligned event-localization artifacts bundled in this repository.

## Event NPZ Files

| Dataset | File | Main fields |
|---------|------|-------------|
| Hansen | `dataset/downstream/change_detection/hansen_hls_change_detection.npz` | `data` (2,191, 7, 1,135), `time`, `siteIds`, `bands`, `loss_year_actual`, `lon`, `lat`, change-date fields |
| Wildfire | `dataset/downstream/change_detection/wildfire_hls_change_detection.npz` | `data` (1,824, 7, 840), `time`, `plot_ids`, `bands`, `class_ids`, `country_codes`, `lon`, `lat`, change-date fields |
| LCMAP-C | `dataset/downstream/change_detection/lcmap_c_hls_change_detection.npz` | `data` (110, 2,189, 7), `yearly_status`, `status_years`, `plot_ids`, `lon`, `lat`, change-date fields |
| LandTD | `dataset/downstream/change_detection/landtd_hls_change_detection.npz` | `data` (1,879, 732, 7), `timemark`, `lon`, `lat`, `transition_type`, `change_step`, `pivot_year` |

## HLS Composite NC Files

| Dataset | File | Main shape |
|---------|------|------------|
| Hansen | `dataset/downstream/change_detection/hls_composite_nc/hansen_hls_change_detection_processed.nc` | time=732, bands=7, samples=1,135 |
| Wildfire | `dataset/downstream/change_detection/hls_composite_nc/wildfire_hls_change_detection_processed.nc` | time=610, bands=7, samples=840 |
| LCMAP-C | `dataset/downstream/change_detection/hls_composite_nc/lcmap_c_hls_change_detection_processed.nc` | time=732, bands=7, samples=110 |
| LandTD | `dataset/downstream/change_detection/hls_composite_nc/landtd_hls_change_detection_processed.nc` | time=732, bands=7, samples=1,879 |

The common event fields are `num_change_points`, `change_date_1`, `change_date_2`, `change_year_1`, `change_year_2`, `lon` and `lat` when available. LandTD additionally provides `change_step` and transition metadata.