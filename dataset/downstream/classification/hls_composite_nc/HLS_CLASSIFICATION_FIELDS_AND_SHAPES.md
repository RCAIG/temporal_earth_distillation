# Downstream Classification Fields and Shapes

This folder documents the paper-aligned downstream classification artifacts bundled in this repository.

## HLS Composite NC Files

| Dataset | File | Main shape |
|---------|------|------------|
| LCMAP | `dataset/downstream/classification/hls_composite_nc/lcmap_hls_classification_processed.nc` | time=732, bands=7, samples=17,780 |
| GlanCE | `dataset/downstream/classification/hls_composite_nc/glance_hls_classification_processed.nc` | time=732, bands=7, samples=10,100 |
| GlobalTree | `dataset/downstream/classification/hls_composite_nc/globaltree_hls_classification_processed.nc` | time=366, bands=7, samples=4,000 |
| CDL | `dataset/downstream/classification/hls_composite_nc/cdl_hls_classification_processed.nc` | time=122, bands=7, samples=11,000 |
| CropHarvest | `dataset/downstream/classification/hls_composite_nc/cropharvest_hls_classification_processed.nc` | time=488, bands=7, samples=15,802 |

All HLS composite files use seven channels and are consumed by the TED/MSM/NTP downstream data loaders.

## AlphaEarth Classification NPZ Files

| Dataset | File | Main fields |
|---------|------|-------------|
| LCMAP | `dataset/downstream/classification/lcmap_alphaearth_classification.npz` | `data` (17,780, 5, 64), `labels`, `plot_ids`, `lon`, `lat`, `time`, `categories` |
| GlanCE | `dataset/downstream/classification/glance_alphaearth_classification.npz` | `data` (3, 64, 10,100), `time`, `siteIds`, `class_ids`, `lon`, `lat` |
| GlobalTree | `dataset/downstream/classification/globaltree_alphaearth_classification.npz` | `data` (4,000, 2, 64), `labels`, `plot_ids`, `lon`, `lat`, `years`, `year_window` |
| CDL | `dataset/downstream/classification/cdl_alphaearth_classification.npz` | `data` (11,000, 64), `labels`, `plot_ids`, `lon`, `lat`, `featureNames`, `year` |
| CropHarvest | `dataset/downstream/classification/cropharvest_alphaearth_classification.npz` | `data` (15,802, 64), `labels`, `coarse_labels`, `plot_ids`, `lon`, `lat`, `sample_year`, `feature_names` |

The AlphaEarth NPZ files provide comparison features and lon/lat arrays for spatial splitting when needed.