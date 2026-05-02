# Governance

Model fairness, regional bias auditing, and prediction explainability.

## Why this exists

ClimateVision serves NGOs across the globe. A model that performs well on the Amazon but poorly on the Congo Basin would systematically under-protect African forests. The governance module exists to detect and report these disparities before they reach production.

## Modules

| File | Purpose |
|------|---------|
| `bias_audit.py` | Region-stratified fairness metrics (demographic parity, equalised odds, predictive parity). |
| `explainability.py` | SHAP-based per-pixel attribution for segmentation predictions. |

## Supported Regions

Bias audit currently stratifies across:

| Region | Coverage |
|--------|----------|
| `amazon` | South American tropical rainforest |
| `congo` | Central African tropical rainforest |
| `southeast_asia` | Indonesia, Malaysia and surrounding tropics |
| `boreal` | Northern coniferous forests (Canada, Russia, Scandinavia) |

Each region has a canonical bbox so per-region samples can be drawn during evaluation.

## Fairness Metrics

- **Demographic parity** — equal positive prediction rate across regions
- **Equalised odds** — equal true-positive and false-positive rates across regions
- **Predictive parity** — equal precision across regions

A region passes the audit when its metric is within `±5%` of the global mean. Failures are written to `outputs/bias_reports/` as timestamped JSON.

## Explainability

`explain_prediction(model, image)` returns SHAP attribution maps highlighting which pixels and which spectral bands contributed most to the segmentation decision. Used by the dashboard to render an "evidence overlay" for any alert.
