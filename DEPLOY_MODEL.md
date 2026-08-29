# Deploying a Trained Model to ClimateVision

End-to-end checklist to take a model from training (Colab) to live on
**climatevision.green**. Do **flooding** first; deforestation and ice follow the
same steps with their own datasets.

---

## A. Train (Google Colab, GPU runtime)

Use `notebooks/flood_training_colab.ipynb` (Run all), or run the steps manually:

1. **Setup** — clone the repo, `pip install -r requirements.txt`, `pip install -e .`, confirm `torch.cuda.is_available()`.
2. **Download data** — `gcloud storage cp --recursive gs://sen1floods11/v1.1/data/flood_events/HandLabeled/{S2Hand,LabelHand}` and `.../splits` into `data/sen1floods11/`.
3. **Convert** —
   ```bash
   python scripts/prepare_sen1floods11.py \
     --s2-dir data/sen1floods11/S2Hand \
     --label-dir data/sen1floods11/LabelHand \
     --splits-dir data/sen1floods11/splits/flood_handlabeled \
     --out-dir data/datasets/flooding
   ```
4. **Train** —
   ```bash
   python scripts/train_real.py --analysis-type flooding \
     --data-dir data/datasets/flooding \
     --epochs 50 --batch-size 8 --image-size 256 --out models
   ```
   Watch `val_iou` rise. Output: `models/flooding_<date>/best_model.pth`.

## B. Validate before promoting

5. **Evaluate** — `python scripts/evaluate.py --checkpoint <run>/best_model.pth --data-dir data/datasets/flooding`
6. **Governance gate** — `python scripts/governance_ci_gate.py`. **Only promote a model that passes.** This is the line between a "preview" and something an agency can act on.
7. **Model card** — `python scripts/generate_model_card.py --checkpoint <run>/best_model.pth` (records metrics + provenance for the audit trail).

## C. Export

8. **ONNX** — `python scripts/export_model.py --checkpoint <run>/best_model.pth`
   produces `<run>/model.onnx` (+ quantized + `export_info.json`).
   The API auto-serves this: `inference/pipeline.py` loads a `.pth` if present,
   else `models/<type>_*/model.onnx` via onnxruntime.

## D. Get the model into the repo / image

Weights are **not** kept in git history by default, but the ONNX run dirs are
small (a few MB) and **are** committed so Render (which builds from GitHub) can
ship them.

9. Download `best_model.pth` and `model.onnx` from Colab.
10. Place them on your laptop under `models/flooding_<date>/`.
11. Commit + push:
    ```bash
    git add models/flooding_<date>/
    git commit -m "feat(models): trained Sen1Floods11 flood model (val_iou=<X>)"
    git push origin main
    ```

> Alternative (large weights): instead of committing, put the files on the Fly
> volume at `/app/outputs` and point `config.yaml` `weights:` at them.

## E. Deploy

12. The push triggers a Render rebuild (or click **Sync** on the blueprint).
13. Confirm secrets are set once: `render env list climatevision-green` —
    `GEE_SERVICE_ACCOUNT_KEY_JSON`, `GEE_PROJECT_ID`, `CLIMATEVISION_ALLOW_DEV_KEY=0`,
    `CLIMATEVISION_CORS_ORIGINS` including `https://climatevision.green`.

## F. Verify it's real

14. Health + model check:
    ```bash
    curl -s https://climatevision.green/api/health | jq
    curl -s https://climatevision.green/api/health/models | jq
    ```
15. Confirm auth is locked down (cv_dev must be rejected):
    ```bash
    curl -s -H "X-API-Key: cv_dev" https://climatevision.green/api/runs   # expect 401
    ```
16. When `/api/health/models` reports a real loaded model (not demo/untrained),
    **remove the "technical preview" label** from the UI/API — you are now
    serving genuine predictions.

---

## Per-type status

| Analysis type | Dataset | Status |
|---------------|---------|--------|
| flooding | Sen1Floods11 | first target — follow this doc |
| deforestation | MultiEarth Amazon | same steps, `--analysis-type deforestation` |
| ice_melting | AI4Arctic v2 | same steps, `--analysis-type ice_melting` |

Keep each model's `provenance.json`, model card, and metrics — that audit trail
is what makes the platform credible for NGO and government use.
