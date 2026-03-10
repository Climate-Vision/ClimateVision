"""
Flood Detection Analysis

Detects and monitors flooding events using satellite imagery.
Uses water indices and change detection to identify flooded areas.
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


class FloodingAnalysis(BaseAnalysisType):
    """
    Flood detection analysis.
    
    Uses satellite imagery to detect flooding events by analyzing
    water presence and comparing to normal conditions. Classifies
    areas into permanent water, flooded, and dry land.
    
    Key Features:
    - Flooded area detection
    - Permanent vs temporary water distinction
    - Affected area estimation
    - Urban/agricultural impact assessment
    
    Input:
        - Sentinel-2 or Landsat imagery
        - Bands: Green, NIR, SWIR for water detection
        - Optional: SAR data (Sentinel-1) for all-weather detection
    
    Output:
        - Multi-class mask (0=dry, 1=permanent water, 2=flooded)
        - Flooded area percentage and km²
        - Impact alerts
    """
    
    name = "flooding"
    display_name = "Flood Detection"
    description = "Detect and monitor flooding events and affected areas"
    
    # Sentinel-2 bands for flood detection
    # B03 (Green), B08 (NIR), B11 (SWIR-1)
    required_bands = ["B03", "B08", "B11"]
    
    output_classes = ["dry_land", "permanent_water", "flooded"]
    
    enabled = True
    
    default_thresholds = {
        "alert_flood_area": 5.0,  # Alert if >5% area flooded
        "critical_flood_area": 20.0,  # Critical if >20% flooded
        "rapid_expansion_rate": 10.0,  # % increase per day
    }
    
    # MNDWI (Modified NDWI) threshold for water detection
    mndwi_water_threshold = 0.0
    
    # Threshold for distinguishing flooded vs permanent water
    flood_detection_threshold = 0.3
    
    def preprocess(
        self,
        image: np.ndarray,
        bands: Optional[list[str]] = None,
    ) -> np.ndarray:
        """
        Preprocess image for flood detection.
        """
        is_valid, error = self.validate_input(image)
        if not is_valid:
            raise ValueError(error)
        
        if image.dtype != np.float32:
            if image.max() > 1:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
        
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            if image.shape[0] <= 3 and image.shape[0] < image.shape[2]:
                image = np.transpose(image, (1, 2, 0))
            
            if image.shape[-1] < 3:
                padding = np.zeros((*image.shape[:2], 3 - image.shape[-1]), dtype=np.float32)
                image = np.concatenate([image, padding], axis=-1)
            elif image.shape[-1] > 3:
                image = image[..., :3]
        
        if image.max() > 1:
            image = image / image.max()
        
        return image
    
    def run_inference(
        self,
        image: np.ndarray,
        model: Optional[Any] = None,
    ) -> tuple[np.ndarray, float]:
        """
        Run flood detection inference.
        
        Uses MNDWI (Modified Normalized Difference Water Index) for
        water detection and additional heuristics for flood classification.
        """
        h, w = image.shape[:2]
        
        if model is not None:
            import torch
            
            input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                if isinstance(output, dict):
                    output = output.get("out", output.get("logits", output))
                probs = torch.softmax(output, dim=1)
                prediction = probs.argmax(dim=1).squeeze().cpu().numpy()
                confidence = probs.max(dim=1).values.mean().item()
        else:
            prediction, confidence = self._water_index_classification(image)
        
        return prediction, confidence
    
    def _water_index_classification(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Classify water bodies using MNDWI and other indices.
        
        MNDWI = (Green - SWIR) / (Green + SWIR)
        Higher values indicate water presence.
        """
        # Channel order: Green (B03), NIR (B08), SWIR (B11)
        green = image[..., 0]
        nir = image[..., 1] if image.shape[-1] >= 2 else image[..., 0]
        swir = image[..., 2] if image.shape[-1] >= 3 else image[..., 1]
        
        # MNDWI = (Green - SWIR) / (Green + SWIR)
        mndwi_denom = green + swir
        mndwi = np.where(mndwi_denom > 0, (green - swir) / mndwi_denom, 0)
        
        # NDWI = (Green - NIR) / (Green + NIR) for additional water detection
        ndwi_denom = green + nir
        ndwi = np.where(ndwi_denom > 0, (green - nir) / ndwi_denom, 0)
        
        # Classification
        # 0 = dry land, 1 = permanent water, 2 = flooded
        prediction = np.zeros(image.shape[:2], dtype=np.int32)
        
        # Water mask (MNDWI > 0 or NDWI > 0.3)
        water_mask = (mndwi > self.mndwi_water_threshold) | (ndwi > 0.3)
        
        # Distinguish permanent water (high MNDWI) from flooded (lower MNDWI)
        permanent_water_mask = water_mask & (mndwi > self.flood_detection_threshold)
        flooded_mask = water_mask & (mndwi <= self.flood_detection_threshold) & (mndwi > -0.2)
        
        prediction[permanent_water_mask] = 1
        prediction[flooded_mask] = 2
        # Dry land remains 0
        
        # Calculate confidence
        water_confidence = np.abs(mndwi[water_mask]).mean() if water_mask.any() else 0.5
        overall_confidence = min(1.0, 0.5 + water_confidence)
        
        return prediction, float(overall_confidence)
    
    def calculate_metrics(
        self,
        prediction: np.ndarray,
        image_size: tuple[int, int],
        bbox: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        """
        Calculate flooding metrics.
        """
        h, w = image_size
        total_pixels = h * w
        
        # Count pixels by class
        dry_pixels = int(np.sum(prediction == 0))
        water_pixels = int(np.sum(prediction == 1))
        flooded_pixels = int(np.sum(prediction == 2))
        
        # Calculate percentages
        flooded_percentage = (flooded_pixels / total_pixels * 100) if total_pixels > 0 else 0
        water_percentage = (water_pixels / total_pixels * 100) if total_pixels > 0 else 0
        
        metrics = {
            "image_size": [h, w],
            "dry_pixels": dry_pixels,
            "water_pixels": water_pixels,
            "flooded_pixels": flooded_pixels,
            "flooded_percentage": round(flooded_percentage, 4),
            "permanent_water_percentage": round(water_percentage, 4),
        }
        
        # Calculate area if bbox provided
        if bbox and len(bbox) == 4:
            min_lon, min_lat, max_lon, max_lat = bbox
            
            lat_diff = abs(max_lat - min_lat)
            lon_diff = abs(max_lon - min_lon)
            avg_lat = (min_lat + max_lat) / 2
            
            lat_km = lat_diff * 111
            lon_km = lon_diff * 111 * np.cos(np.radians(avg_lat))
            total_area_km2 = lat_km * lon_km
            
            if total_pixels > 0:
                flooded_area_km2 = total_area_km2 * (flooded_pixels / total_pixels)
                water_area_km2 = total_area_km2 * (water_pixels / total_pixels)
            else:
                flooded_area_km2 = 0
                water_area_km2 = 0
            
            metrics["total_area_km2"] = round(total_area_km2, 2)
            metrics["flooded_area_km2"] = round(flooded_area_km2, 2)
            metrics["permanent_water_km2"] = round(water_area_km2, 2)
        
        return metrics
    
    def generate_alerts(
        self,
        metrics: dict[str, Any],
        thresholds: Optional[dict[str, float]] = None,
        previous_metrics: Optional[dict[str, Any]] = None,
    ) -> list[Alert]:
        """
        Generate flood alerts.
        """
        alerts = []
        thresholds = thresholds or self.default_thresholds
        
        flooded_percentage = metrics.get("flooded_percentage", 0)
        flooded_area_km2 = metrics.get("flooded_area_km2")
        
        # Check flood severity
        critical_threshold = thresholds.get("critical_flood_area", 20.0)
        alert_threshold = thresholds.get("alert_flood_area", 5.0)
        
        if flooded_percentage >= critical_threshold:
            message = f"Critical flooding: {flooded_percentage:.1f}% of area flooded"
            if flooded_area_km2:
                message += f" ({flooded_area_km2:.1f} km²)"
            
            alerts.append(Alert(
                alert_type="critical_flooding",
                severity=Severity.CRITICAL,
                title="Critical Flooding Detected",
                message=message,
                threshold_exceeded=critical_threshold,
                measured_value=flooded_percentage,
                details={"flooded_area_km2": flooded_area_km2},
            ))
        elif flooded_percentage >= alert_threshold:
            message = f"Flooding detected: {flooded_percentage:.1f}% of area flooded"
            if flooded_area_km2:
                message += f" ({flooded_area_km2:.1f} km²)"
            
            alerts.append(Alert(
                alert_type="flooding_detected",
                severity=Severity.HIGH,
                title="Flooding Detected",
                message=message,
                threshold_exceeded=alert_threshold,
                measured_value=flooded_percentage,
            ))
        
        # Check for rapid expansion
        if previous_metrics:
            prev_flooded = previous_metrics.get("flooded_percentage", 0)
            expansion = flooded_percentage - prev_flooded
            rapid_rate = thresholds.get("rapid_expansion_rate", 10.0)
            
            if expansion >= rapid_rate:
                alerts.append(Alert(
                    alert_type="rapid_flood_expansion",
                    severity=Severity.HIGH,
                    title="Rapid Flood Expansion",
                    message=f"Flooded area increased by {expansion:.1f}%",
                    threshold_exceeded=rapid_rate,
                    measured_value=expansion,
                    details={
                        "previous_flooded": prev_flooded,
                        "current_flooded": flooded_percentage,
                    },
                ))
        
        return alerts
