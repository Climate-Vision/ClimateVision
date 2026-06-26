# ClimateVision — Real-Model Training Loop (Colab)

This is the end-to-end loop that turns ClimateVision from a demo (untrained /
synthetic models) into a platform with production models that agencies, NGOs,
and government bodies can rely on. It runs on **Google Colab** (free or Pro;
Pro recommended for the GPU and longer runtimes).

The loop has five stages, repeated per analysis type
(`flooding`, `deforestation`, `ice_melting`):

```
  1. Download data  ->  2. Train  ->  3. Evaluate  ->  4. Export ONNX  ->  5. Deploy weights
        ^                                                                        |
        |________________________ retrain when metrics drift ___________________|
```

---

## 0. One-time Colab setup

```python
!git clone https://github.com/Climate-Vision/ClimateVision.git
%cd ClimateVision
!pip install -q -r requirements.txt
!pip install -e .
# Confirm GPU
import torch; print("CUDA:", torch.cuda.is_available())
```

Mount Google Drive so checkpoints survive a runtime reset:

```python
from google.colab import drive
drive.mount('/content/drive')
```

GEE auth (for pulling fresh tiles during evaluation) — **never paste your
service-account key into the notebook**. Use interactive auth in Colab:

```python
import ee
ee.Authenticate()          # opens an OAuth flow
ee.Initialize(project='YOUR_GEE_PROJECT_ID')
```

> Note: `scripts/run_training.py` currently hardcodes `ee.Initialize(project='kinos-473422')`
> and trains on `SyntheticForestDataset`. The steps below replace that synthetic
> path with real data. Rotate any key that has been shared in plaintext.

---

## 1. Download a real dataset

```bash
!python scripts/download_datasets.py --dataset flooding --out data/datasets
```

| Analysis type | Dataset | License | Tool needed |
|---------------|---------|---------|-------------|
| `flooding` | Sen1Floods11 (Sentinel-1 SAR) | CC-BY 4.0 | `gsutil` |
| `deforestation` | MultiEarth Amazon (Sentinel-2) | research use | `huggingface_hub` |
| `ice_melting` | AI4Arctic Sea Ice v2 | CC-BY 4.0 | `eotdl` |

Each download writes a `provenance.json` so `governance/datasheet.py` can record
the source and license — keep this; it is part of what makes the platform
credible for government use.

---

## 2. Train

Train the U-Net for the chosen type. Match `in_channels` / `num_classes` to the
entry in `config.yaml` (e.g. flooding = 3 channels, 3 classes):

```bash
!python scripts/run_training.py \
    --analysis-type flooding \
    --data-dir data/datasets/flooding \
    --epochs 50 \
    --batch-size 8 \
    --out /content/drive/MyDrive/climatevision/models
```

The trainer should save `best_model.pth` containing at least
`{"model_state_dict", "epoch", "val_iou"}` — this is exactly what
`inference/pipeline.py::_find_best_checkpoint` looks for.

> If `run_training.py` does not yet accept `--analysis-type/--data-dir`, use the
> `training/trainer.py` API directly (see `notebooks/04_model_validation.ipynb`)
> with a `Dataset` that reads the downloaded chips instead of synthetic ones.

---

## 3. Evaluate

```bash
!python scripts/evaluate.py \
    --checkpoint /content/drive/MyDrive/climatevision/models/<run>/best_model.pth \
    --data-dir data/datasets/flooding
```

Record IoU / F1 in the model card and gate on them:

```bash
!python scripts/generate_model_card.py --checkpoint <run>/best_model.pth
!python scripts/governance_ci_gate.py     # enforces metric + fairness thresholds
```

Only promote a model that passes the governance gate. This is the line between a
"preview" and something an NGO can act on.

---

## 4. Export to ONNX

```bash
!python scripts/export_model.py --checkpoint <run>/best_model.pth
# -> <run>/model.onnx  (+ model_quantized.onnx, export_info.json)
```

The serving layer now picks this up automatically: `inference/pipeline.py`
loads `models/<analysis_type>_*/model.onnx` via `onnxruntime` when no `.pth` is
present, or you can set `onnx_weights:` per type in `config.yaml`.

---

## 5. Deploy the weights

Weights are **not** committed (`.gitignore` excludes `models/*.pth`). Ship them
one of two ways:

- **Bake into the image** — copy the trained `best_model.pth` (or `model.onnx`)
  into `models/` before `docker build`; the Dockerfile already does `COPY models/`.
- **Mount on the Fly volume** — `fly.toml` mounts `climatevision_data` at
  `/app/outputs`; place weights there and point `config.yaml` `weights:` at them.

Then redeploy (`./scripts/deploy.sh` for Fly, or push for the Render blueprint).

---

## Retraining cadence

Re-run stages 1-5 when: new labeled data is released, the governance gate flags
metric/fairness drift, or the scheduled freshness check (see the Cowork
"ClimateVision model freshness" task) reports models older than the threshold.
Keep each run's `provenance.json`, model card, and metrics — that audit trail is
the asset, not just the weights.
