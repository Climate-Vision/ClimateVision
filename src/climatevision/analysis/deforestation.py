"""
Deforestation Analysis

Detects forest coverage and deforestation using satellite imagery.
Uses U-Net semantic segmentation to classify forest vs non-forest areas.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import logging

from climatevision.analysis.base import (
    BaseAnalysisType,
    Alert,
    Severity,
)

logger = logging.getLogger(__name__)


class DeforestationAnalysis(BaseAnalysisType):
    """
    Deforestation detection analysis.
    
    Uses semantic segmentation to identify forest and non-forest areas
    in satellite imagery. Calculates forest coverage percentage and
    generates alerts when significant deforestation is detected.
    
    Input:
        - Sentinel-2 or Landsat imagery with RGB + NIR bands
    
    Output:
        - Binary mask (0 = non-forest, 1 = forest)
        - Forest coverage percentage
        - NDVI statistics
    """
    
    name = "deforestation"
    display_name = "Deforestation Detection"
    description = "Monitor forest coverage and detect deforestation events using satellite imagery"
    
    # Sentinel-2 bands: B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)
    required_bands = ["B04", "B03", "B02", "B08"]
    
    output_classes = ["non_forest", "forest"]
    
    enabled = True
    
    default_thresholds = {
        "alert_forest_loss": 5.0,  # Alert if >5% forest loss
        "critical_forest_loss": 15.0,  # Critical if >15% loss
        "min_forest_coverage": 20.0,  # Alert if coverage drops below 20%
    }
    
    def preprocess(
        self,
        image: np.ndarray,
        bands: Optional[list[str]] = None,
    ) -> np.ndarray:
        """
        Preprocess image for deforestation model.
        
        Steps:
        1. Normalize pixel values to [0, 1]
        2. Ensure correct channel order (RGB + NIR)
        3. Resize to model input size if needed
        """
        # Validate input
        is_valid, error = self.validate_input(image)
        if not is_valid:
            raise ValueError(error)
        
        # Convert to float32
        if image.dtype != np.float32:
            if image.max() > 1:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
        
        # Ensure correct shape
        if image.ndim == 2:
            # Grayscale - replicate to 4 channels
            image = np.stack([image] * 4, axis=-1)
        elif image.ndim == 3:
            # Handle channel order
            if image.shape[0] <= 4 and image.shape[0] < image.shape[2]:
                # (C, H, W) -> (H, W, C)
                image = np.transpose(image, (1, 2, 0))
            
            # Ensure 4 channels
            if image.shape[-1] < 4:
                # Pad with zeros or replicate last channel
                padding = np.zeros((*image.shape[:2], 4 - image.shape[-1]), dtype=np.float32)
                image = np.concatenate([image, padding], axis=-1)
            elif image.shape[-1] > 4:
                # Take first 4 channels
                image = image[..., :4]
        
        # Normalize to [0, 1] if not already
        if image.max() > 1:
            image = image / image.max()
        
        return image
    
    def run_inference(
        self,
        image: np.ndarray,
        model: Optional[Any] = None,
    ) -> tuple[np.ndarray, float]:
        """
        Run deforestation model inference.
        
        Returns binary mask and confidence score.
        """
        h, w = image.shape[:2]
        
        if model is not None:
            # Use provided model
            import torch
            
            # Prepare input tensor
            input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                if isinstance(output, dict):
                    output = output.get("out", output.get("logits", output))
                probs = torch.softmax(output, dim=1)
                prediction = probs.argmax(dim=1).squeeze().cpu().numpy()
                confidence = probs.max(dim=1).values.mean().item()
        else:
            # Fallback: Use NDVI-based classification
            prediction, confidence = self._ndvi_classification(image)
        
        return prediction, confidence
    
    def _ndvi_classification(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Simple NDVI-based forest classification as fallback.
        
        NDVI = (NIR - Red) / (NIR + Red)
        Forest typically has NDVI > 0.4
        """
        # Assume channel order: R, G, B, NIR
        if image.shape[-1] >= 4:
            red = image[..., 0]
            nir = image[..., 3]
        elif image.shape[-1] == 3:
            # RGB only - use green as proxy for vegetation
            red = image[..., 0]
            nir = image[..., 1]  # Green as proxy
        else:
            # Fallback
            return np.zeros(image.shape[:2], dtype=np.int32), 0.5
        
        # Calculate NDVI
        denominator = nir + red
        ndvi = np.where(denominator > 0, (nir - red) / denominator, 0)
        
        # Classify: NDVI > 0.4 = forest
        prediction = (ndvi > 0.4).astype(np.int32)
        
        # Confidence based on NDVI clarity
        confidence = min(1.0, np.abs(ndvi - 0.4).mean() * 2 + 0.5)
        
        return prediction, float(confidence)
    
    def calculate_metrics(
        self,
        prediction: np.ndarray,
        image_size: tuple[int, int],
        bbox: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        """
        Calculate deforestation metrics.
        
        Returns:
            - forest_pixels: Number of forest pixels
            - non_forest_pixels: Number of non-forest pixels
            - forest_percentage: Percentage of forest coverage
            - total_pixels: Total analyzed pixels
            - area_km2: Estimated area in km² (if bbox provided)
        """
        h, w = image_size
        total_pixels = h * w
        
        # Count pixels
        forest_pixels = int(np.sum(prediction == 1))
        non_forest_pixels = total_pixels - forest_pixels
        
        # Calculate percentage
        forest_percentage = (forest_pixels / total_pixels * 100) if total_pixels > 0 else 0
        
        metrics = {
            "image_size": [h, w],
            "forest_pixels": forest_pixels,
            "non_forest_pixels": non_forest_pixels,
            "forest_percentage": round(forest_percentage, 4),
        }
        
        # Calculate area if bbox provided
        if bbox and len(bbox) == 4:
            min_lon, min_lat, max_lon, max_lat = bbox
            
            # Approximate area calculation
            lat_diff = abs(max_lat - min_lat)
            lon_diff = abs(max_lon - min_lon)
            avg_lat = (min_lat + max_lat) / 2
            
            # 1 degree ≈ 111 km at equator
            lat_km = lat_diff * 111
            lon_km = lon_diff * 111 * np.cos(np.radians(avg_lat))
            total_area_km2 = lat_km * lon_km
            forest_area_km2 = total_area_km2 * (forest_percentage / 100)
            
            metrics["total_area_km2"] = round(total_area_km2, 2)
            metrics["forest_area_km2"] = round(forest_area_km2, 2)
        
        return metrics
    
    def generate_alerts(
        self,
        metrics: dict[str, Any],
        thresholds: Optional[dict[str, float]] = None,
        previous_metrics: Optional[dict[str, Any]] = None,
    ) -> list[Alert]:
        """
        Generate deforestation alerts.
        
        Alerts are generated when:
        - Forest loss exceeds threshold (compared to previous)
        - Forest coverage drops below minimum
        """
        alerts = []
        thresholds = thresholds or self.default_thresholds
        
        forest_percentage = metrics.get("forest_percentage", 0)
        
        # Check minimum coverage
        min_coverage = thresholds.get("min_forest_coverage", 20.0)
        if forest_percentage < min_coverage:
            alerts.append(Alert(
                alert_type="low_forest_coverage",
                severity=Severity.HIGH,
                title="Low Forest Coverage",
                message=f"Forest coverage ({forest_percentage:.1f}%) is below minimum threshold ({min_coverage}%)",
                threshold_exceeded=min_coverage,
                measured_value=forest_percentage,
            ))
        
        # Check for forest loss (if previous metrics available)
        if previous_metrics:
            prev_percentage = previous_metrics.get("forest_percentage", 0)
            if prev_percentage > 0:
                loss_percentage = prev_percentage - forest_percentage
                
                critical_loss = thresholds.get("critical_forest_loss", 15.0)
                alert_loss = thresholds.get("alert_forest_loss", 5.0)
                
                if loss_percentage >= critical_loss:
                    alerts.append(Alert(
                        alert_type="critical_deforestation",
                        severity=Severity.CRITICAL,
                        title="Critical Deforestation Detected",
                        message=f"Severe forest loss detected: {loss_percentage:.1f}% reduction in coverage",
                        threshold_exceeded=critical_loss,
                        measured_value=loss_percentage,
                        details={
                            "previous_coverage": prev_percentage,
                            "current_coverage": forest_percentage,
                        },
                    ))
                elif loss_percentage >= alert_loss:
                    alerts.append(Alert(
                        alert_type="deforestation_detected",
                        severity=Severity.MEDIUM,
                        title="Deforestation Detected",
                        message=f"Forest loss detected: {loss_percentage:.1f}% reduction in coverage",
                        threshold_exceeded=alert_loss,
                        measured_value=loss_percentage,
                        details={
                            "previous_coverage": prev_percentage,
                            "current_coverage": forest_percentage,
                        },
                    ))
        
        return alerts
    
    def calculate_ndvi_stats(self, image: np.ndarray) -> dict[str, float]:
        """
        Calculate NDVI statistics for the image.
        
        NDVI (Normalized Difference Vegetation Index) indicates
        vegetation health: -1 to 1, with higher values indicating
        healthier vegetation.
        """
        if image.shape[-1] >= 4:
            red = image[..., 0]
            nir = image[..., 3]
        else:
            return {"NDVI_min": 0.0, "NDVI_mean": 0.0, "NDVI_max": 0.0}
        
        denominator = nir + red
        ndvi = np.where(denominator > 0, (nir - red) / denominator, 0)
        
        return {
            "NDVI_min": float(np.min(ndvi)),
            "NDVI_mean": float(np.mean(ndvi)),
            "NDVI_max": float(np.max(ndvi)),
        }
