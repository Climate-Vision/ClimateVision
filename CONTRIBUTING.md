# Contributing to ClimateVision

Thank you for your interest in contributing to ClimateVision! This document provides guidelines for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/Climate-Vision/ClimateVision/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, etc.)
   - Screenshots if applicable

### Suggesting Features

1. Check [Discussions](https://github.com/Climate-Vision/ClimateVision/discussions) for similar ideas
2. Create a new discussion or issue describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternative approaches considered
   - Potential impact on users

### Contributing Code

#### First Time Contributors

Look for issues labeled `good first issue` - these are specifically chosen for newcomers.

#### Development Process

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/ClimateVision.git
   cd ClimateVision
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   pip install -r requirements-dev.txt
   ```

4. **Make your changes**
   - Write clean, documented code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation

5. **Run tests**
   ```bash
   # Run all tests
   pytest tests/
   
   # Check code style
   black src/
   flake8 src/
   mypy src/
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```
   
   Use these prefixes:
   - `Add:` New feature or file
   - `Fix:` Bug fix
   - `Update:` Modify existing feature
   - `Refactor:` Code restructuring
   - `Docs:` Documentation changes
   - `Test:` Add or modify tests

7. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a PR on GitHub with:
   - Clear description of changes
   - Related issue numbers
   - Screenshots/examples if applicable

#### Pull Request Guidelines

- Keep PRs focused on a single feature/fix
- Ensure all tests pass
- Update documentation if needed
- Respond to review feedback promptly
- Squash commits if requested

## Code Style

### Python

We follow PEP 8 with these specifics:

```python
# Use 4 spaces for indentation
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    """
    # Code here
    pass

# Type hints for all functions
def process_data(data: np.ndarray) -> Dict[str, float]:
    pass

# Docstrings for all classes and public methods
class ForestDetector:
    """High-level interface for forest detection."""
    
    def __init__(self, model_path: str):
        """Initialize detector with model."""
        pass
```

### File Organization

```
src/climatevision/
├── module/
│   ├── __init__.py     # Public API exports
│   ├── core.py         # Main functionality
│   ├── utils.py        # Helper functions
│   └── constants.py    # Constants
```

### Imports

```python
# Standard library
import os
from pathlib import Path

# Third party
import numpy as np
import torch

# Local
from climatevision.data import loader
from climatevision.models.unet import UNet
```

## Testing

### Writing Tests

```python
import pytest
from climatevision.data.loader import load_sentinel2_image

def test_load_sentinel2_image():
    """Test loading Sentinel-2 imagery."""
    image = load_sentinel2_image(
        coordinates=(-3.4653, -62.2159, -3.0653, -61.8159),
        date_range=("2024-01-01", "2024-01-31"),
        cloud_coverage_max=20
    )
    
    assert image.shape[0] == 4  # 4 bands
    assert image.shape[1:] == (256, 256)  # Default size
    assert len(image.bands) == 4
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_loader.py

# With coverage
pytest --cov=climatevision tests/

# Verbose output
pytest -v
```

## Documentation

### Docstring Format

We use Google-style docstrings:

```python
def predict(self, image: SatelliteImage, threshold: float = 0.5) -> PredictionResult:
    """
    Predict forest areas in satellite image.
    
    Args:
        image: SatelliteImage object containing the data
        threshold: Classification threshold (0-1). Higher values require
            more confidence to classify as forest.
    
    Returns:
        PredictionResult object containing mask, probabilities, and statistics
    
    Raises:
        ValueError: If threshold is not between 0 and 1
    
    Example:
        >>> detector = ForestDetector(model_path="model.pth")
        >>> result = detector.predict(image, threshold=0.6)
        >>> stats = result.get_statistics()
    """
    pass
```

### Adding Documentation

- Update `docs/` folder for major features
- Create Jupyter notebooks for tutorials
- Add examples to docstrings
- Update README if user-facing changes

## Development Areas

We welcome contributions in these areas:

### ML Models (Engineer 1)
- Improve U-Net architecture
- Implement new segmentation models
- Add transfer learning from pretrained models
- Optimize hyperparameters

### Data Pipeline (Engineer 2)
- Sentinel Hub API integration
- Google Earth Engine integration
- Data augmentation techniques
- Distributed processing

### Analytics (Engineer 3)
- Carbon estimation models
- Uncertainty quantification
- Validation pipelines
- Impact reporting

### Infrastructure (Engineer 4)
- FastAPI endpoints
- Model serving optimization
- Docker deployment
- Monitoring and logging

## Community

- **GitHub Discussions**: General questions, ideas, showcases
- **Discord**: Real-time chat and collaboration
- **Monthly Community Calls**: Demo new features and discuss roadmap

## Recognition

Contributors will be:
- Listed in README
- Credited in release notes
- Invited to co-author academic papers using ClimateVision
- Given speaking opportunities at conferences

## Questions?

- Check [Documentation](https://docs.climatevision.org)
- Search [GitHub Discussions](https://github.com/Climate-Vision/ClimateVision/discussions)
- Ask in Discord
- Email: contribute@climatevision.org

Thank you for helping protect the world's forests! 🌍🌲
