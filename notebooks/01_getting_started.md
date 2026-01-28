# ClimateVision - Getting Started Example

This notebook demonstrates the basic workflow for forest detection using ClimateVision.

## Setup

```python
import sys
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
from climatevision.data.loader import load_sentinel2_image, SatelliteImage
from climatevision.models.detector import ForestDetector
from climatevision.models.unet import create_unet
import torch

print("✓ Imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 1. Load Satellite Data

```python
# Define area of interest (Amazon rainforest region)
coordinates = (-3.4653, -62.2159, -3.0653, -61.8159)  # (min_lat, min_lon, max_lat, max_lon)
date_range = ("2024-01-01", "2024-01-31")

# Load Sentinel-2 imagery
image = load_sentinel2_image(
    coordinates=coordinates,
    date_range=date_range,
    cloud_coverage_max=20,
    bands=["Red", "Green", "Blue", "NIR"]
)

print(f"Image shape: {image.shape}")
print(f"Bands: {image.bands}")
print(f"Coordinates: {image.coordinates}")
print(f"Date: {image.date}")
```

## 2. Explore the Data

```python
# Visualize RGB composite
rgb = np.stack([
    image.get_band("Red"),
    image.get_band("Green"),
    image.get_band("Blue")
], axis=-1)

# Normalize for display
rgb_normalized = (rgb - rgb.min()) / (rgb.max() - rgb.min())

plt.figure(figsize=(10, 10))
plt.imshow(rgb_normalized)
plt.title("RGB Composite")
plt.axis('off')
plt.show()
```

```python
# Compute and visualize NDVI
ndvi = image.compute_ndvi()

plt.figure(figsize=(10, 10))
plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(label='NDVI')
plt.title("Normalized Difference Vegetation Index (NDVI)")
plt.axis('off')
plt.show()

print(f"NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
print(f"Mean NDVI: {ndvi.mean():.3f}")
```

## 3. Initialize Forest Detector

```python
# Create detector
# Note: Without trained weights, this will use a randomly initialized model
detector = ForestDetector(
    model_path=None,  # Set to trained model path when available
    device='cuda' if torch.cuda.is_available() else 'cpu',
    use_uncertainty=False
)

print("✓ Detector initialized")
print(f"Device: {detector.device}")
```

## 4. Run Forest Detection

```python
# Run prediction
result = detector.predict(
    image=image,
    threshold=0.5
)

print("✓ Prediction complete!")
```

## 5. Analyze Results

```python
# Get statistics
stats = result.get_statistics()

print("\n=== Prediction Statistics ===")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")
```

## 6. Visualize Results

```python
# Plot all results
result.plot(
    show_confidence=True,
    show_uncertainty=False,
    save_path="../outputs/forest_detection_result.png"
)
```

```python
# Create custom visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 15))

# RGB
axes[0, 0].imshow(rgb_normalized)
axes[0, 0].set_title('RGB Composite')
axes[0, 0].axis('off')

# NDVI
im1 = axes[0, 1].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
axes[0, 1].set_title('NDVI')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

# Forest mask
im2 = axes[1, 0].imshow(result.mask, cmap='RdYlGn', vmin=0, vmax=1)
axes[1, 0].set_title('Forest Mask')
axes[1, 0].axis('off')
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

# Probability
im3 = axes[1, 1].imshow(result.probabilities, cmap='viridis', vmin=0, vmax=1)
axes[1, 1].set_title('Forest Probability')
axes[1, 1].axis('off')
plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

plt.tight_layout()
plt.savefig('../outputs/comprehensive_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 7. Test Model Architecture

```python
# Test model creation and forward pass
model = create_unet(in_channels=4, num_classes=2)

# Create dummy input
dummy_input = torch.randn(1, 4, 256, 256)

# Forward pass
with torch.no_grad():
    output = model(dummy_input)

print(f"\nModel test:")
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 8. Test Uncertainty Estimation

```python
# Create model with dropout for uncertainty
uncertainty_model = create_unet(
    in_channels=4,
    num_classes=2,
    uncertainty=True
)

detector_with_uncertainty = ForestDetector(
    model_path=None,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    use_uncertainty=True
)

# Run prediction with uncertainty
result_uncertain = detector_with_uncertainty.predict(image, threshold=0.5)

# Visualize with uncertainty
result_uncertain.plot(
    show_confidence=True,
    show_uncertainty=True,
    save_path="../outputs/forest_detection_with_uncertainty.png"
)

print("✓ Uncertainty estimation test complete!")
```

## Next Steps

1. **Collect Training Data**: Gather labeled satellite imagery for training
2. **Train Model**: Use `scripts/train.py` to train on your dataset
3. **Evaluate Performance**: Test on held-out data and calculate metrics
4. **Deploy**: Set up API for production use
5. **Monitor**: Track model performance over time

## Notes for Development

- Currently using dummy data from `load_sentinel2_image()` - implement real API integration
- Model weights are random - need to train on labeled data
- Adjust threshold based on your specific use case
- Consider ensemble methods for improved accuracy
