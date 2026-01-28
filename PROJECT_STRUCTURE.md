# ClimateVision Project Structure

This document explains the organization of the ClimateVision codebase.

## 📁 Directory Structure

```
ClimateVision/
├── src/climatevision/           # Main package source code
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Configuration management
│   ├── models/                 # Deep learning models
│   │   ├── __init__.py
│   │   ├── unet.py            # U-Net segmentation model
│   │   └── siamese.py         # Siamese network for change detection
│   ├── data/                   # Data loading and preprocessing
│   │   └── __init__.py         # [TO BE IMPLEMENTED]
│   ├── inference/              # Inference utilities
│   │   └── __init__.py         # [TO BE IMPLEMENTED]
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── metrics.py         # Evaluation metrics and loss functions
│   │   ├── visualization.py   # Plotting and visualization
│   │   └── geospatial.py      # Geospatial utilities
│   └── api/                    # FastAPI backend
│       └── __init__.py         # [TO BE IMPLEMENTED]
│
├── tests/                      # Test suite
│   └── [TO BE CREATED]
│
├── notebooks/                  # Jupyter notebooks
│   └── 01_quickstart.ipynb    # Getting started tutorial
│
├── scripts/                    # Utility scripts
│   └── [TO BE CREATED]
│
├── docs/                       # Documentation
│   └── [TO BE CREATED]
│
├── models_pretrained/          # Pre-trained model weights
│   └── [Models will be saved here]
│
├── data/                       # Data directory (not in git)
│   ├── raw/                   # Raw satellite imagery
│   ├── processed/             # Processed datasets
│   └── satellite/             # Downloaded satellite data
│
├── config/                     # Configuration files
│   └── [TO BE CREATED]
│
├── frontend/                   # Web dashboard (React)
│   └── [TO BE CREATED]
│
├── .github/workflows/          # CI/CD pipelines
│   └── [TO BE CREATED]
│
├── README.md                   # Project overview
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # MIT License
├── setup.py                    # Package installation
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git ignore rules
```

## 🔧 Module Responsibilities

### `src/climatevision/`

**Main Package**
- `__init__.py`: Package initialization, exports main classes
- `config.py`: Configuration management, project paths, hyperparameters

### `models/`
**Deep Learning Architectures**
- `unet.py`: U-Net and Attention U-Net for semantic segmentation
  - Binary/multi-class forest classification
  - Skip connections for preserving spatial information
- `siamese.py`: Siamese and Early Fusion networks for change detection
  - Temporal comparison of satellite images
  - Change map generation

**Status**: ✅ Implemented

### `utils/`
**Helper Functions**
- `metrics.py`: 
  - Segmentation metrics (IoU, Dice, F1)
  - Change detection metrics (confusion matrix, kappa)
  - Custom loss functions (Dice Loss, Focal Loss)
- `visualization.py`:
  - Satellite imagery visualization
  - Prediction overlays
  - Change detection maps
  - NDVI calculation and display
- `geospatial.py`:
  - Coordinate transformations
  - Area calculations (hectares, carbon loss)
  - Bounding box operations
  - GeoTIFF metadata generation

**Status**: ✅ Implemented

### `data/` [TO BE IMPLEMENTED]
**Data Pipeline**

Priority tasks for Data Science Engineer 2:
- [ ] Satellite data downloaders (Sentinel-2, Landsat)
- [ ] Data preprocessing pipeline
  - Cloud masking
  - Atmospheric correction
  - Normalization
  - Tiling for model input
- [ ] Dataset classes (PyTorch Dataset)
- [ ] Data augmentation
- [ ] Caching and versioning (DVC)

### `inference/` [TO BE IMPLEMENTED]
**Model Inference**

Priority tasks for Data Science Engineer 4:
- [ ] Single image prediction
- [ ] Batch processing pipeline
- [ ] Model loading utilities
- [ ] Post-processing (smoothing, filtering)
- [ ] Alert generation logic
- [ ] Uncertainty quantification

### `api/` [TO BE IMPLEMENTED]
**REST API Backend**

Priority tasks for Data Science Engineer 4:
- [ ] FastAPI application setup
- [ ] Prediction endpoints
- [ ] File upload handling
- [ ] Model serving with ONNX
- [ ] Rate limiting
- [ ] Authentication
- [ ] WebSocket for real-time updates

## 🎯 Implementation Priorities

### Week 1-2: Data Pipeline (Engineer 2)
```python
# High priority files to create:
data/
├── sentinel2.py          # Sentinel-2 data loader
├── landsat.py            # Landsat data loader  
├── dataset.py            # PyTorch Dataset classes
├── preprocess.py         # Preprocessing utilities
└── augmentation.py       # Data augmentation
```

### Week 3-4: Training Infrastructure (Engineer 1 & 3)
```python
# Create training loop and evaluation:
training/
├── trainer.py            # Training loop
├── evaluator.py          # Model evaluation
├── callbacks.py          # Training callbacks
└── checkpointing.py      # Model checkpointing
```

### Week 5-6: Inference Pipeline (Engineer 4)
```python
# Create inference system:
inference/
├── predictor.py          # Single image prediction
├── batch_processor.py    # Batch processing
├── postprocess.py        # Post-processing
└── alert_generator.py    # Alert generation
```

### Week 7-8: API Development (Engineer 4)
```python
# Create REST API:
api/
├── main.py              # FastAPI app
├── routes.py            # API endpoints
├── models.py            # Pydantic models
├── middleware.py        # Authentication, CORS
└── serving.py           # Model serving
```

### Week 9-10: Dashboard (Team Collaboration)
```python
# Create web interface:
frontend/
├── src/
│   ├── components/      # React components
│   ├── pages/          # Dashboard pages
│   ├── api/            # API client
│   └── utils/          # Utilities
└── public/             # Static assets
```

## 📝 Coding Standards

### File Naming
- Python files: `lowercase_with_underscores.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_CASE`

### Import Order
```python
# 1. Standard library
import os
from typing import Tuple

# 2. Third-party
import numpy as np
import torch

# 3. Local
from climatevision.models import UNet
from climatevision.utils import calculate_iou
```

### Type Hints
Always include type hints:
```python
def process_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """Process satellite image."""
    pass
```

## 🧪 Testing Strategy

### Unit Tests
```python
tests/
├── test_models/
│   ├── test_unet.py       # Test U-Net
│   └── test_siamese.py    # Test Siamese network
├── test_utils/
│   ├── test_metrics.py    # Test metrics
│   └── test_geospatial.py # Test geospatial utils
└── test_data/
    └── test_dataset.py    # Test data loaders
```

### Integration Tests
- End-to-end inference pipeline
- API endpoint testing
- Model training pipeline

### Test Coverage Goal
- Minimum 80% code coverage
- All public functions tested
- Critical paths 100% covered

## 📊 Data Flow

```
Satellite Data APIs
        ↓
   Data Loader (data/)
        ↓
  Preprocessing (data/)
        ↓
   Dataset (data/)
        ↓
   DataLoader (PyTorch)
        ↓
    Model (models/)
        ↓
  Prediction (inference/)
        ↓
Post-processing (inference/)
        ↓
   API Response (api/)
        ↓
  Dashboard (frontend/)
```

## 🚀 Next Steps

1. **Set up development environment**
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

2. **Run existing code**
   ```bash
   # Test models
   python -c "from climatevision.models import UNet; print('Models loaded!')"
   
   # Try quickstart notebook
   jupyter notebook notebooks/01_quickstart.ipynb
   ```

3. **Choose your module**
   - Engineer 1: Start with `training/` module
   - Engineer 2: Start with `data/` module
   - Engineer 3: Start with regression models in `models/`
   - Engineer 4: Start with `inference/` module

4. **Create first PR**
   - Implement one small feature
   - Add tests
   - Update documentation
   - Submit for review

## 📚 Resources

- **PyTorch**: https://pytorch.org/docs/
- **Rasterio**: https://rasterio.readthedocs.io/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Sentinel-2**: https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2

---

**Questions?** Open a GitHub Discussion or check CONTRIBUTING.md
