"""
SHAP-based explainability for ClimateVision segmentation models.

Provides pixel-level and band-level attribution for U-Net predictions,
helping stakeholders understand WHY the model classified each region.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs" / "explanations"

BAND_NAMES = {
    "deforestation": ["Red", "Green", "Blue", "NIR"],
    "ice_melting": ["Red", "Green", "Blue", "NIR"],
    "flooding": ["Green", "NIR", "SWIR1"],
}


class SHAPExplainer:
    """
    SHAP explainer for U-Net segmentation models.

    Uses DeepExplainer for efficient gradient-based SHAP values on CNNs.
    Falls back to GradientExplainer if DeepExplainer fails.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        background_data: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        if background_data is None:
            n_channels = getattr(model, "n_channels", 4)
            background_data = torch.zeros(1, n_channels, 64, 64)

        self.background = background_data.to(self.device)
        self._explainer = None
        self._explainer_type = None

    def _init_explainer(self, input_tensor: torch.Tensor) -> None:
        """Lazily initialize SHAP explainer on first use."""
        if self._explainer is not None:
            return

        try:
            import shap
            self._explainer = shap.DeepExplainer(self.model, self.background)
            self._explainer_type = "deep"
            logger.info("Initialized SHAP DeepExplainer")
        except Exception as e:
            logger.warning("DeepExplainer failed (%s), trying GradientExplainer", e)
            try:
                import shap
                self._explainer = shap.GradientExplainer(self.model, self.background)
                self._explainer_type = "gradient"
                logger.info("Initialized SHAP GradientExplainer")
            except Exception as e2:
                logger.warning("GradientExplainer failed (%s), using gradient fallback", e2)
                self._explainer_type = "fallback"

    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Generate SHAP explanations for input tensor.

        Args:
            input_tensor: (N, C, H, W) input tensor
            target_class: Class index to explain (default: predicted class)

        Returns:
            Dictionary with SHAP values, band contributions, and metadata
        """
        self._init_explainer(input_tensor)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            predictions = torch.argmax(output, dim=1)
            probabilities = torch.softmax(output, dim=1)

        if target_class is None:
            target_class = int(predictions[0].mode().values.item())

        if self._explainer_type == "fallback":
            shap_values = self._gradient_fallback(input_tensor, target_class)
        else:
            try:
                shap_values = self._explainer.shap_values(input_tensor)
                if isinstance(shap_values, list):
                    shap_values = shap_values[target_class]
                shap_values = np.array(shap_values)
            except Exception as e:
                logger.warning("SHAP computation failed (%s), using gradient fallback", e)
                shap_values = self._gradient_fallback(input_tensor, target_class)

        band_contributions = self._compute_band_contributions(shap_values)
        spatial_importance = self._compute_spatial_importance(shap_values)

        return {
            "shap_values": shap_values,
            "band_contributions": band_contributions,
            "spatial_importance": spatial_importance,
            "target_class": target_class,
            "prediction": int(predictions[0].mode().values.item()),
            "confidence": float(probabilities[0, target_class].mean().item()),
            "explainer_type": self._explainer_type,
        }

    def _gradient_fallback(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """Compute gradient-based attribution as SHAP fallback."""
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)
        target_output = output[:, target_class, :, :].sum()
        target_output.backward()

        gradients = input_tensor.grad.detach().cpu().numpy()
        attributions = gradients * input_tensor.detach().cpu().numpy()

        return attributions

    def _compute_band_contributions(self, shap_values: np.ndarray) -> dict[str, float]:
        """Compute per-band contribution scores."""
        abs_shap = np.abs(shap_values)
        band_importance = abs_shap.mean(axis=(0, 2, 3))
        total = band_importance.sum() + 1e-8

        contributions = {}
        for i, importance in enumerate(band_importance):
            contributions[f"band_{i}"] = float(importance / total)

        return contributions

    def _compute_spatial_importance(self, shap_values: np.ndarray) -> np.ndarray:
        """Compute spatial importance heatmap (H, W)."""
        abs_shap = np.abs(shap_values)
        spatial = abs_shap.mean(axis=(0, 1))
        spatial = (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-8)
        return spatial


def explain_prediction(
    model_path: Union[str, Path],
    image_path: Union[str, Path],
    analysis_type: str = "deforestation",
    target_class: Optional[int] = None,
    save_heatmap: bool = True,
) -> dict[str, Any]:
    """
    Generate SHAP explanation for a prediction.

    Args:
        model_path: Path to model checkpoint
        image_path: Path to input image (GeoTIFF or PNG)
        analysis_type: Type of analysis (deforestation, ice_melting, flooding)
        target_class: Class to explain (default: predicted class)
        save_heatmap: Whether to save heatmap to disk

    Returns:
        Dictionary with explanation results
    """
    from climatevision.inference.pipeline import _load_image_file, _load_model

    model, device = _load_model(analysis_type)
    image = _load_image_file(str(image_path))

    if image.ndim == 3 and image.shape[2] < image.shape[0]:
        image = np.transpose(image, (2, 0, 1))

    n_channels = model.n_channels
    c, h, w = image.shape
    if c < n_channels:
        pad = np.zeros((n_channels - c, h, w), dtype=image.dtype)
        image = np.concatenate([image, pad], axis=0)
    elif c > n_channels:
        image = image[:n_channels]

    tensor = torch.FloatTensor(image.astype(np.float32)).unsqueeze(0)

    explainer = SHAPExplainer(model, device=device)
    result = explainer.explain(tensor, target_class=target_class)

    band_names = BAND_NAMES.get(analysis_type, [f"Band_{i}" for i in range(n_channels)])
    top_bands = []
    for i, (band_key, importance) in enumerate(
        sorted(result["band_contributions"].items(), key=lambda x: x[1], reverse=True)
    ):
        band_idx = int(band_key.split("_")[1])
        band_name = band_names[band_idx] if band_idx < len(band_names) else band_key
        top_bands.append({"band": band_name, "importance": round(importance, 4)})

    result["top_bands"] = top_bands
    result["analysis_type"] = analysis_type

    if save_heatmap:
        heatmap_path = generate_shap_heatmap(
            result["spatial_importance"],
            image_path,
            analysis_type,
        )
        result["heatmap_path"] = str(heatmap_path)

    result.pop("shap_values", None)

    return result


def generate_shap_heatmap(
    spatial_importance: np.ndarray,
    source_image_path: Union[str, Path],
    analysis_type: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Generate and save SHAP heatmap visualization.

    Args:
        spatial_importance: (H, W) importance scores
        source_image_path: Original image path (for naming)
        analysis_type: Analysis type
        output_dir: Output directory (default: outputs/explanations/)

    Returns:
        Path to saved heatmap
    """
    output_dir = output_dir or _OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    source_name = Path(source_image_path).stem
    heatmap_path = output_dir / f"{source_name}_{analysis_type}_shap.npy"

    np.save(heatmap_path, spatial_importance)
    logger.info("Saved SHAP heatmap to %s", heatmap_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        png_path = output_dir / f"{source_name}_{analysis_type}_shap.png"

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(spatial_importance, cmap="hot", interpolation="nearest")
        ax.set_title(f"SHAP Importance - {analysis_type.replace('_', ' ').title()}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, label="Importance")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved SHAP heatmap PNG to %s", png_path)
        return png_path

    except ImportError:
        logger.warning("matplotlib not available, saved .npy only")
        return heatmap_path


def get_band_contributions(
    model_path: Union[str, Path],
    image_path: Union[str, Path],
    analysis_type: str = "deforestation",
) -> dict[str, float]:
    """
    Get band-level contribution scores for a prediction.

    Convenience function that returns only band contributions.

    Args:
        model_path: Path to model checkpoint
        image_path: Path to input image
        analysis_type: Type of analysis

    Returns:
        Dictionary mapping band names to importance scores
    """
    result = explain_prediction(
        model_path=model_path,
        image_path=image_path,
        analysis_type=analysis_type,
        save_heatmap=False,
    )

    band_names = BAND_NAMES.get(analysis_type, [])
    contributions = {}

    for band_info in result.get("top_bands", []):
        contributions[band_info["band"]] = band_info["importance"]

    return contributions
