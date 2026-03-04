"""
Visualization utilities for satellite imagery and predictions
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import Optional, Tuple, List
import cv2


def visualize_satellite_image(
    image: np.ndarray,
    bands: List[int] = [3, 2, 1],  # RGB bands (Red, Green, Blue for Sentinel-2)
    title: str = "Satellite Image",
    figsize: Tuple[int, int] = (10, 10),
    percentile_clip: Tuple[float, float] = (2, 98),
) -> plt.Figure:
    """
    Visualize satellite imagery as RGB composite
    
    Args:
        image: Multispectral image (C, H, W) or (H, W, C)
        bands: Indices of bands to use for RGB visualization
        title: Plot title
        figsize: Figure size
        percentile_clip: Percentile values for contrast stretching
    
    Returns:
        matplotlib Figure
    """
    # Ensure image is in (H, W, C) format
    if image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0))
    
    # Select RGB bands
    rgb_image = image[:, :, bands]
    
    # Normalize using percentile clipping for better visualization
    p_low, p_high = percentile_clip
    rgb_normalized = np.zeros_like(rgb_image, dtype=np.float32)
    
    for i in range(3):
        band = rgb_image[:, :, i]
        p2, p98 = np.percentile(band, (p_low, p_high))
        rgb_normalized[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(rgb_normalized)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    return fig


def visualize_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    class_names: List[str] = ["Non-Forest", "Forest"],
    colors: List[str] = ['#8B4513', '#228B22'],  # Brown, Green
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Visualize segmentation prediction overlaid on satellite image
    
    Args:
        image: RGB satellite image (H, W, 3)
        prediction: Predicted segmentation mask (H, W)
        ground_truth: Ground truth mask (H, W), optional
        class_names: Names of classes
        colors: Colors for each class
        alpha: Transparency of overlay
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    n_plots = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title("Satellite Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot prediction
    cmap = ListedColormap(colors)
    axes[1].imshow(image)
    pred_overlay = axes[1].imshow(prediction, cmap=cmap, alpha=alpha, vmin=0, vmax=len(class_names)-1)
    axes[1].set_title("Prediction", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Plot ground truth if available
    if ground_truth is not None:
        axes[2].imshow(image)
        axes[2].imshow(ground_truth, cmap=cmap, alpha=alpha, vmin=0, vmax=len(class_names)-1)
        axes[2].set_title("Ground Truth", fontsize=12, fontweight='bold')
        axes[2].axis('off')
    
    # Create legend
    patches = [mpatches.Patch(color=colors[i], label=class_names[i]) for i in range(len(class_names))]
    fig.legend(handles=patches, loc='lower center', ncol=len(class_names), frameon=False)
    
    plt.tight_layout()
    return fig


def visualize_change_detection(
    image_before: np.ndarray,
    image_after: np.ndarray,
    change_map: np.ndarray,
    title: str = "Change Detection",
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Visualize change detection results
    
    Args:
        image_before: Satellite image before (H, W, 3)
        image_after: Satellite image after (H, W, 3)
        change_map: Binary change map (H, W) - 1 indicates change
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Before
    axes[0].imshow(image_before)
    axes[0].set_title("Before", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # After
    axes[1].imshow(image_after)
    axes[1].set_title("After", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Change map
    axes[2].imshow(image_after)
    change_overlay = axes[2].imshow(
        change_map, 
        cmap=ListedColormap(['none', 'red']), 
        alpha=0.6,
        vmin=0,
        vmax=1
    )
    axes[2].set_title("Detected Changes", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add legend
    patches = [
        mpatches.Patch(color='none', label='No Change'),
        mpatches.Patch(color='red', label='Deforestation')
    ]
    fig.legend(handles=patches, loc='lower center', ncol=2, frameon=False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_training_history(
    history: dict,
    metrics: List[str] = ['loss', 'accuracy', 'iou'],
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Plot training history curves
    
    Args:
        history: Dictionary with training history
                 e.g., {'train_loss': [...], 'val_loss': [...], ...}
        metrics: List of metrics to plot
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            axes[idx].plot(history[train_key], label='Train', linewidth=2)
        if val_key in history:
            axes[idx].plot(history[val_key], label='Validation', linewidth=2)
        
        axes[idx].set_xlabel('Epoch', fontsize=11)
        axes[idx].set_ylabel(metric.capitalize(), fontsize=11)
        axes[idx].set_title(f'{metric.capitalize()} over Epochs', fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_false_color_composite(
    image: np.ndarray,
    nir_band: int = 7,  # B08 for Sentinel-2
    red_band: int = 3,  # B04
    green_band: int = 2,  # B03
    percentile_clip: Tuple[float, float] = (2, 98),
) -> np.ndarray:
    """
    Create false color composite (NIR-Red-Green) for vegetation visualization
    
    Healthy vegetation appears bright red in this composite
    
    Args:
        image: Multispectral image (C, H, W) or (H, W, C)
        nir_band: Index of NIR band
        red_band: Index of Red band
        green_band: Index of Green band
        percentile_clip: Percentile values for contrast stretching
    
    Returns:
        False color RGB image (H, W, 3)
    """
    # Ensure image is in (H, W, C) format
    if image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0))
    
    # Create false color composite
    false_color = np.stack([
        image[:, :, nir_band],
        image[:, :, red_band],
        image[:, :, green_band]
    ], axis=2)
    
    # Normalize using percentile clipping
    p_low, p_high = percentile_clip
    false_color_normalized = np.zeros_like(false_color, dtype=np.float32)
    
    for i in range(3):
        band = false_color[:, :, i]
        p2, p98 = np.percentile(band, (p_low, p_high))
        false_color_normalized[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
    
    return false_color_normalized


def calculate_ndvi(image: np.ndarray, nir_band: int = 7, red_band: int = 3) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index (NDVI)
    
    NDVI = (NIR - Red) / (NIR + Red)
    Values range from -1 to 1, with higher values indicating healthier vegetation
    
    Args:
        image: Multispectral image (C, H, W) or (H, W, C)
        nir_band: Index of NIR band
        red_band: Index of Red band
    
    Returns:
        NDVI map (H, W)
    """
    # Ensure image is in (H, W, C) format
    if image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0))
    
    nir = image[:, :, nir_band].astype(float)
    red = image[:, :, red_band].astype(float)
    
    # Calculate NDVI
    ndvi = (nir - red) / (nir + red + 1e-8)  # Add small epsilon to avoid division by zero
    
    return ndvi


def visualize_ndvi(
    image: np.ndarray,
    nir_band: int = 7,
    red_band: int = 3,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Visualize NDVI alongside RGB image
    
    Args:
        image: Multispectral image
        nir_band: Index of NIR band
        red_band: Index of Red band
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    ndvi = calculate_ndvi(image, nir_band, red_band)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # RGB image
    rgb = visualize_satellite_image(image, figsize=figsize)
    axes[0].imshow(image[:, :, [3, 2, 1]])  # Assume standard band order
    axes[0].set_title("RGB Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # NDVI
    im = axes[1].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1].set_title("NDVI (Vegetation Index)", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig
