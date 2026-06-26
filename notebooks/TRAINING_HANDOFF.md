# ClimateVision — Training Handoff (run this now)

**Goal:** produce real trained models for ClimateVision so the live site
(climatevision.green) stops serving demo/untrained output. Do **flooding first**
(best dataset fit), then deforestation, then ice.

**You need:** Google Colab (Pro recommended — GPU + longer runtime) and a Google
account with Earth Engine access. You do **not** need any private key file — use
interactive auth (step 2). Est. time: ~30 min setup + 1–3 h GPU training per model.

When done, you hand back **one file per model** (`best_model.pth` and/or
`model.onnx`) — see step 7.

---

## 1. Set up the environment (Colab cell)

```python
!git clone https://github.com/Climate-Vision/ClimateVision.git
%cd ClimateVision
!pip install -q -r requirements.txt
!pip install -q -e .
import torch; print("CUDA available:", torch.cuda.is_available())   # must be True

from google.colab import drive          # so checkpoints survive a runtime reset
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/climatevision/models
```

## 2. Earth Engine auth — interactive, NO key file

```python
import ee
ee.Authenticate()                       # opens an OAuth link, paste the token
ee.Initialize(project='YOUR_GEE_PROJECT_ID')
```

> GEE is only needed if you pull fresh tiles for spot-checking. Training itself
> uses the downloaded dataset, so you can skip this if you just want to train.

## 3. Download a dataset

```bash
# flooding (Sen1Floods11) needs gsutil:
!pip install -q gsutil
!python scripts/download_datasets.py --dataset flooding --out data/datasets
```

| Type | Dataset | Tool | License |
|------|---------|------|---------|
| flooding | Sen1Floods11 | `gsutil` | CC-BY 4.0 |
| deforestation | MultiEarth Amazon | `huggingface_hub` | research use |
| ice_melting | AI4Arctic v2 | `eotdl` | CC-BY 4.0 |

## 4. Convert to the training layout  ⚠️ the one manual step

The trainer reads stem-matched GeoTIFF pairs in this structure:

```
data/datasets/<type>/
   train/  images/ *.tif   masks/ *.tif
   val/    images/ *.tif   masks/ *.tif
   test/   images/ *.tif   masks/ *.tif
```

Each dataset ships in its own format, so write a short conversion that:
1. reads each scene + its label,
2. writes the image bands ClimateVision expects for that type
   (flooding = B03,B08,B11 → 3 bands; deforestation = B04,B03,B02,B08 → 4 bands),
3. writes the mask as a single-band `uint8` with class indices matching
   `config.yaml` (flooding: 0=dry_land, 1=permanent_water, 2=flooded),
4. uses the **same filename stem** for image and mask,
5. splits ~80/10/10 into train/val/test.

Notes per dataset:
- **Sen1Floods11**: GeoTIFF chips already; map its water/flood labels to the
  3-class scheme above. SAR bands — run them through
  `src/climatevision/data/sar_preprocessing.py` (speckle filter + dB scaling).
- **MultiEarth**: Sentinel-2 chips with binary deforestation masks (0/1) — fits
  the 4-band / 2-class deforestation config directly.
- **AI4Arctic**: netCDF — extract the SAR layer + ice chart, tile to 256×256.

## 5. Train

```bash
!python scripts/train_real.py \
    --analysis-type flooding \
    --data-dir data/datasets/flooding \
    --epochs 50 --batch-size 8 \
    --out /content/drive/MyDrive/climatevision/models
```

`train_real.py` reads channels/classes from `config.yaml` automatically, uses the
existing `Trainer` (AdamW, warmup→cosine LR, EMA, mixed precision, early stopping)
and saves `best_model.pth` (with `model_state_dict` + `val_iou`) to the run dir.

## 6. Evaluate and gate

```bash
!python scripts/evaluate.py --checkpoint <run>/best_model.pth --data-dir data/datasets/flooding
!python scripts/generate_model_card.py --checkpoint <run>/best_model.pth
!python scripts/governance_ci_gate.py        # must pass before promoting a model
```

Only promote a model that clears the governance gate (metrics + fairness). That
is the line between "preview" and something an agency can act on.

## 7. Export ONNX + hand back

```bash
!python scripts/export_model.py --checkpoint <run>/best_model.pth
# produces <run>/model.onnx (+ model_quantized.onnx, export_info.json)
```

Hand back to the project owner, per analysis type, either:
- `best_model.pth`  (PyTorch), or
- `model.onnx`      (the API serves this automatically via onnxruntime).

Deployment (owner does this): drop the file into `models/<type>_<date>/` and
rebuild the Docker image (the Dockerfile already `COPY models/`), **or** place it
on the Fly volume at `/app/outputs` and point `config.yaml` `weights:` at it.
The serving code (`inference/pipeline.py`) finds `.pth` first, else `model.onnx`.

---

## Quick reference — how serving picks up your model

`inference/pipeline.py` resolution order per type:
1. `config.yaml` `weights:` path (PyTorch `.pth`)
2. `models/best_model.pth`
3. newest `models/*/best_model.pth`
4. `config.yaml` `onnx_weights:` or newest `models/<type>_*/model.onnx` (ONNX)
5. otherwise → untrained demo weights (what's live now)

So just getting a trained file into `models/<type>_*/` is enough to go live with
a real model.
