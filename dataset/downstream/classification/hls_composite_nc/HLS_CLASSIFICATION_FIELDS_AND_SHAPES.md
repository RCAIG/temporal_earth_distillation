# Downstream Classification HLS Fields and Shapes

HLS data only, including raw `npz` files and composite `nc` outputs.

## lcmap_hls_classification.npz
- Path: `/intelnvme01/ziyun/downstream/classification/lcmap_hls_classification.npz`
- Field count: 5
- Fields:
  - `data`: shape=(17780, 2189, 7), dtype=float64
  - `labels`: shape=(17780), dtype=int64
  - `plot_ids`: shape=(17780), dtype=<U5
  - `time`: shape=(2189), dtype=<U10
  - `bands`: shape=(7), dtype=<U5
- Composite NC: `/intelnvme01/ziyun/downstream/classification/hls_composite_nc/lcmap_hls_classification_processed.nc`
- NC dims: time=732, bands=7, samples=17780
- NC variables:
  - `data`: shape=(732, 7, 17780), dtype=float64
  - `time`: shape=(732), dtype=<class 'str'>
  - `labels`: shape=(17780), dtype=int64
  - `plot_ids`: shape=(17780), dtype=<class 'str'>
  - `bands`: shape=(7), dtype=<class 'str'>

## glancetraining_hls_classification.npz
- Path: `/intelnvme01/ziyun/downstream/classification/glancetraining_hls_classification.npz`
- Field count: 4
- Fields:
  - `data`: shape=(2188, 7, 10100), dtype=float64
  - `time`: shape=(2188), dtype=<U10
  - `siteIds`: shape=(10100), dtype=<U30
  - `class_ids`: shape=(10100), dtype=int64
- Composite NC: `/intelnvme01/ziyun/downstream/classification/hls_composite_nc/glancetraining_hls_classification_processed.nc`
- NC dims: time=732, bands=7, samples=10100
- NC variables:
  - `data`: shape=(732, 7, 10100), dtype=float64
  - `time`: shape=(732), dtype=<class 'str'>
  - `siteIds`: shape=(10100), dtype=<class 'str'>
  - `class_ids`: shape=(10100), dtype=int64

## cdl_hls_classification.npz
- Path: `/intelnvme01/ziyun/downstream/classification/cdl_hls_classification.npz`
- Field count: 7
- Fields:
  - `data`: shape=(11000, 363, 7), dtype=float64
  - `labels`: shape=(11000), dtype=int32
  - `plot_ids`: shape=(11000), dtype=<U5
  - `lon`: shape=(11000), dtype=float64
  - `lat`: shape=(11000), dtype=float64
  - `time`: shape=(363), dtype=<U10
  - `bands`: shape=(7), dtype=<U5
- Composite NC: `/intelnvme01/ziyun/downstream/classification/hls_composite_nc/cdl_hls_classification_processed.nc`
- NC dims: time=122, bands=7, samples=11000
- NC variables:
  - `data`: shape=(122, 7, 11000), dtype=float64
  - `time`: shape=(122), dtype=<class 'str'>
  - `labels`: shape=(11000), dtype=int32
  - `plot_ids`: shape=(11000), dtype=<class 'str'>
  - `lon`: shape=(11000), dtype=float64
  - `lat`: shape=(11000), dtype=float64
  - `bands`: shape=(7), dtype=<class 'str'>

## cropharvest_hls_classification.npz
- Path: `/intelnvme01/ziyun/downstream/classification/cropharvest_hls_classification.npz`
- Field count: 8
- Fields:
  - `data`: shape=(15802, 1281, 7), dtype=float64
  - `labels`: shape=(15802), dtype=int32
  - `coarse_labels`: shape=(15802), dtype=<U24
  - `plot_ids`: shape=(15802), dtype=<U5
  - `lon`: shape=(15802), dtype=float64
  - `lat`: shape=(15802), dtype=float64
  - `time`: shape=(1281), dtype=<U10
  - `bands`: shape=(7), dtype=<U5
- Composite NC: `/intelnvme01/ziyun/downstream/classification/hls_composite_nc/cropharvest_hls_classification_processed.nc`
- NC dims: time=488, bands=7, samples=15802
- NC variables:
  - `data`: shape=(488, 7, 15802), dtype=float64
  - `time`: shape=(488), dtype=<class 'str'>
  - `labels`: shape=(15802), dtype=int32
  - `coarse_labels`: shape=(15802), dtype=<class 'str'>
  - `plot_ids`: shape=(15802), dtype=<class 'str'>
  - `lon`: shape=(15802), dtype=float64
  - `lat`: shape=(15802), dtype=float64
  - `bands`: shape=(7), dtype=<class 'str'>

## globaltree_hls_classification.npz
- Path: `/intelnvme01/ziyun/downstream/classification/globaltree_hls_classification.npz`
- Field count: 7
- Fields:
  - `data`: shape=(4000, 1093, 7), dtype=float64
  - `labels`: shape=(4000), dtype=int32
  - `plot_ids`: shape=(4000), dtype=<U4
  - `lon`: shape=(4000), dtype=float64
  - `lat`: shape=(4000), dtype=float64
  - `time`: shape=(1093), dtype=<U10
  - `year_window`: shape=(4000, 2), dtype=int32 — per-sample valid HLS year window `(start_year, end_year)`; `end_year=-1` means only `start_year` is valid
- Composite NC: `/intelnvme01/ziyun/downstream/classification/hls_composite_nc/globaltree_hls_classification_processed.nc`
- NC dims: time=366, bands=7, samples=4000, year_window_dim=2
- NC variables:
  - `data`: shape=(366, 7, 4000), dtype=float64
  - `time`: shape=(366), dtype=<class 'str'>
  - `labels`: shape=(4000), dtype=int32
  - `plot_ids`: shape=(4000), dtype=<class 'str'>
  - `lon`: shape=(4000), dtype=float64
  - `lat`: shape=(4000), dtype=float64
  - `year_window`: shape=(4000, 2), dtype=int32 — same as NPZ: which years have valid HLS per sample
- `year_window` value distribution (4000 samples):
  - `(2022, 2023)`: 1597
  - `(2023, 2024)`: 1541
  - `(2022, -1)`: 296
  - `(2023, -1)`: 310
  - `(2024, -1)`: 256
