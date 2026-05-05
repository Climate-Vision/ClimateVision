# Data Pipeline

Sentinel-2 ingestion, band mapping, and preprocessing for ClimateVision.

## Modules

| File | Purpose |
|------|---------|
| `gee_downloader.py` | Download real Sentinel-2 tiles from Google Earth Engine for a given bbox + date range. Falls back to a labelled synthetic tile (`is_synthetic: true`) when GEE credentials are missing. |
| `band_mapping.py` | Single source of truth for which spectral bands each analysis type requires. Reads from `config.yaml`. |
| `preprocessing.py` | Cloud masking (SCL band), normalisation, resampling 20m bands to 10m, tiling to 256×256. |
| `transforms.py` | Augmentation pipeline (flips, rotations, spectral jitter) for training DataLoaders. |
| `sampling.py` | Tile sampling strategies (random, balanced, stratified by region). |
| `quality.py` | Per-tile QA (cloud %, NaN ratio, band coverage). |
| `validation.py` | Schema validation for incoming requests and downloaded tiles. |

## Analysis-Type Band Contract

Every analysis type has its own band list in `config.yaml`. The pipeline must use `get_bands_for_analysis(analysis_type)` — never hardcode band lists.

| Analysis | Bands | Channels |
|----------|-------|----------|
| `deforestation` | B04, B03, B02, B08 | 4 |
| `ice_melting` | B02, B03, B04, B11 | 4 |
| `flooding` | B03, B08, B11 | 3 |

## Cloud Masking

`apply_scl_cloud_mask(image, scl_band)` zeroes out pixels classified as cloud, shadow, snow/ice, or no-data using the Sentinel-2 Scene Classification Layer (SCL). This must run **before** the model forward pass.

Valid SCL classes kept: 4 (vegetation), 5 (bare soil), 6 (water), 7 (low cloud), 10 (thin cirrus).
Masked out: 0 (no-data), 1 (saturated), 2 (dark), 3 (shadow), 8/9 (medium/high cloud), 11 (snow/ice).

## Synthetic Fallback

If GEE auth fails, the downloader returns a deterministic synthetic tile seeded by the bbox so the same region always yields the same fallback. The metadata always includes `is_synthetic: true` so the API can warn the caller.

## Environment

```
GEE_PROJECT_ID=your-project-id
GEE_SERVICE_ACCOUNT=svc@project.iam.gserviceaccount.com
GEE_SERVICE_ACCOUNT_KEY=secrets/gee-key.json
```

Run `python scripts/setup_gee.py` to verify credentials.
