# ClimateVision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/ML-PyTorch-ee4c2c.svg)](https://pytorch.org)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## What is ClimateVision?

ClimateVision is an open-source machine learning platform that detects environmental change from satellite imagery. It uses deep learning (U-Net, Siamese networks) to monitor **deforestation**, **arctic ice melting**, and **flooding** — giving conservation NGOs and researchers automated alerts without manual analysis. Built on Sentinel-2 and Landsat data via Google Earth Engine, it runs as a REST API with a React dashboard for real-time monitoring.

---

## Installation

```bash
git clone https://github.com/Climate-Vision/ClimateVision.git
cd ClimateVision
pip install -r requirements.txt
```

---

## Quickstart

**Start the API server:**

```bash
uvicorn climatevision.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Run a deforestation analysis:**

```bash
curl -X POST http://localhost:8000/api/predict/json \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": [-60.0, -15.0, -45.0, -5.0],
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "analysis_type": "deforestation"
  }'
```

**Launch the dashboard:**

```bash
cd frontend && npm install && npm run dev
# Visit http://localhost:5173
```

**Explore the API docs:** `http://localhost:8000/docs`

---

## Features

- **Multi-type climate analysis** — deforestation, arctic ice melting, flood detection (drought and wildfire detection planned)
- **Deep learning inference** — U-Net semantic segmentation and Siamese network change detection on Sentinel-2 imagery
- **Automated data pipeline** — Google Earth Engine integration with cloud masking, normalization, and 256×256 tiling
- **NGO management** — register organisations, subscribe to regions, receive threshold-based alerts via email or webhook
- **REST API** — FastAPI backend with paginated run history, stats endpoint, and full OpenAPI docs
- **React dashboard** — interactive map with bbox region selector, Recharts analytics, confidence gauges, and run history
- **MLflow experiment tracking** — log training runs, hyperparameters, and model checkpoints
- **ONNX export** — optimised model export for fast production inference

---

## Documentation

Full documentation: [github.com/Climate-Vision/ClimateVision/wiki](https://github.com/Climate-Vision/ClimateVision/wiki)

- [Getting Started](GETTING_STARTED.md)
- [API Reference](docs/API_REFERENCE.md) — `http://localhost:8000/docs` when running locally
- [Project Structure](PROJECT_STRUCTURE.md)
- [Training Guide](notebooks/01_getting_started.md)
- [Colab Notebook](notebooks/train_on_colab.ipynb)

---

## Models & Analysis Types

| Analysis Type | Status | Classes | Satellite Bands |
|--------------|--------|---------|----------------|
| Deforestation | Active | forest, non-forest | B02, B03, B04, B08 |
| Arctic Ice Melting | Active | sea-ice, open-water, land | B02, B03, B04, B11 |
| Flood Detection | Active | water, flooded, dry-land | B03, B08, B11 |
| Drought Monitoring | Planned | normal, stressed, severe | B04, B08, B11, B12 |
| Wildfire Detection | Planned | unburned, burned, active-fire | B04, B08, B11, B12 |

**Performance benchmarks** (baseline U-Net on held-out test sets):

| Metric | Value |
|--------|-------|
| Forest segmentation IoU | in progress |
| Change detection F1 | in progress |
| Inference time (256×256 tile) | ~45ms on CPU |
| API response time | <100ms |

*Benchmarks will be updated as the team completes training runs. See [MLflow tracking](logs/) for experiment history.*

---

## Contributing

We welcome contributions — bug reports, new analysis types, model improvements, documentation, and translations.

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
pip install -r requirements.txt
pytest tests/
# Submit your PR against the develop branch
```

Read the [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before getting started.

Good first issues are labelled [`good first issue`](https://github.com/Climate-Vision/ClimateVision/issues?q=label%3A%22good+first+issue%22) on GitHub.

---

## License & Citation

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

If you use ClimateVision in your research, please cite:

```bibtex
@software{climatevision2026,
  author  = {ClimateVision Contributors},
  title   = {ClimateVision: Open-Source AI Platform for Environmental Monitoring},
  year    = {2026},
  url     = {https://github.com/Climate-Vision/ClimateVision},
  version = {0.2.0}
}
```

---

<p align="center">
  <a href="https://github.com/Climate-Vision/ClimateVision/stargazers">⭐ Star us on GitHub</a>
  &nbsp;·&nbsp;
  <a href="CONTRIBUTING.md">Contribute</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/Climate-Vision/ClimateVision/issues">Report a Bug</a>
</p>
