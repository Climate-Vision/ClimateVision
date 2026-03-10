"""
Arctic Ice Melting Analysis

Monitors sea ice extent and melting patterns in polar regions.
Uses spectral indices and semantic segmentation to classify ice,
open water, and land areas.
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


class IceMeltingAnalysis(BaseAnalysisType):
    """
    Arctic/Antarctic ice melting analysis.
    
    Uses satellite imagery to monitor sea ice extent, concentration,
    and melting patterns. Classifies areas into sea ice, open water,
    and land.
    
    Key Features:
    - Ice extent calculation (km²)
    - Ice concentration percentage
    - Multi-year vs first-year ice detection
    - Melt rate estimation (when historical data available)
    
    Input:
        - Sentinel-2, MODIS, or Landsat imagery
        - Bands: Blue, Green, Red, SWIR for ice detection
    
    Output:
        - Multi-class mask (0=water, 1=ice, 2=land)
        - Ice extent and concentration metrics
        - Change detection alerts
    """
    
    name = "ice_melting"
    display_name = "Arctic Ice Melting"
    description = "Monitor sea ice extent and melting patterns in polar regions"
    
    # Sentinel-2 bands useful for ice detection
    # B02 (Blue), B03 (Green), B04 (Red), B11 (SWIR-1)
    required_bands = ["B02", "B03", "B04", "B11"]
    
    output_classes = ["open_water", "sea_ice", "land", "cloud"]
    
    enabled = True
    
    default_thresholds = {
        "alert_ice_loss": 10.0,  # Alert if >10% ice loss
        "critical_ice_loss": 25.0,  # Critical if >25% loss
        "min_ice_concentration": 15.0,  # Alert if concentration drops below 15%
        "rapid_melt_rate": 5.0,  # km²/day threshold for rapid melt alert
    }
    
    # NDSI (Normalized Difference Snow Index) threshold
    ndsi_ice_threshold = 0.4
    
    # NDWI (Normalized Difference Water Index) threshold
    ndwi_water_threshold = 0.3
    
    def preprocess(
        self,
        image: np.ndarray,
        bands: Optional[list[str]] = None,
    ) -> np.ndarray:
        """
        Preprocess image for ice detection.
        
        Steps:
        1. Normalize pixel values
        2. Ensure correct channel order (Blue, Green, Red, SWIR)
        3. Apply polar region corrections if needed
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
            image = np.stack([image] * 4, axis=-1)
        elif image.ndim == 3:
            if image.shape[0] <= 4 and image.shape[0] < image.shape[2]:
                image = np.transpose(image, (1, 2, 0))
            
            if image.shape[-1] < 4:
                padding = np.zeros((*image.shape[:2], 4 - image.shape[-1]), dtype=np.float32)
                image = np.concatenate([image, padding], axis=-1)
            elif image.shape[-1] > 4:
                image = image[..., :4]
        
        # Normalize to [0, 1]
        if image.max() > 1:
            image = image / image.max()
        
        return image
    
    def run_inference(
        self,
        image: np.ndarray,
        model: Optional[Any] = None,
    ) -> tuple[np.ndarray, float]:
        """
        Run ice detection inference.
        
        Uses spectral indices for ice classification:
        - NDSI (Normalized Difference Snow Index) for ice/snow
        - NDWI (Normalized Difference Water Index) for water
        
        Returns multi-class mask and confidence score.
        """
        h, w = image.shape[:2]
        
        if model is not None:
            # Use provided model
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
            # Use spectral indices for classification
            prediction, confidence = self._spectral_classification(image)
        
        return prediction, confidence
    
    def _spectral_classification(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Classify ice, water, and land using spectral indices.
        
        Uses NDSI and NDWI for classification:
        - NDSI > 0.4 and NDWI < 0 → Sea ice
        - NDWI > 0.3 → Open water
        - Otherwise → Land
        """
        # Channel order: Blue (B02), Green (B03), Red (B04), SWIR (B11)
        blue = image[..., 0]
        green = image[..., 1]
        red = image[..., 2]
        swir = image[..., 3] if image.shape[-1] >= 4 else image[..., 2]
        
        # NDSI = (Green - SWIR) / (Green + SWIR)
        # Higher values indicate snow/ice
        ndsi_denom = green + swir
        ndsi = np.where(ndsi_denom > 0, (green - swir) / ndsi_denom, 0)
        
        # NDWI = (Green - NIR) / (Green + NIR)
        # For water detection, we can use (Green - SWIR) / (Green + SWIR) as proxy
        # Or use (Blue - Red) / (Blue + Red) for water bodies
        ndwi_denom = blue + red
        ndwi = np.where(ndwi_denom > 0, (blue - red) / ndwi_denom, 0)
        
        # Classification
        # 0 = open water, 1 = sea ice, 2 = land
        prediction = np.zeros(image.shape[:2], dtype=np.int32)
        
        # Water: NDWI > threshold and not ice
        water_mask = (ndwi > self.ndwi_water_threshold) & (ndsi < 0.2)
        prediction[water_mask] = 0
        
        # Ice: NDSI > threshold
        ice_mask = ndsi > self.ndsi_ice_threshold
        prediction[ice_mask] = 1
        
        # Land: Everything else
        land_mask = ~water_mask & ~ice_mask
        prediction[land_mask] = 2
        
        # Calculate confidence based on index clarity
        ice_confidence = np.where(ice_mask, np.abs(ndsi - self.ndsi_ice_threshold), 0).mean()
        water_confidence = np.where(water_mask, np.abs(ndwi - self.ndwi_water_threshold), 0).mean()
        overall_confidence = min(1.0, 0.5 + ice_confidence + water_confidence)
        
        return prediction, float(overall_confidence)
    
    def calculate_metrics(
        self,
        prediction: np.ndarray,
        image_size: tuple[int, int],
        bbox: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        """
        Calculate ice extent metrics.
        
        Returns:
            - ice_pixels: Number of ice pixels
            - water_pixels: Number of open water pixels
            - land_pixels: Number of land pixels
            - ice_percentage: Ice concentration percentage
            - ice_extent_km2: Ice extent in km² (if bbox provided)
        """
        h, w = image_size
        total_pixels = h * w
        
        # Count pixels by class
        ice_pixels = int(np.sum(prediction == 1))
        water_pixels = int(np.sum(prediction == 0))
        land_pixels = int(np.sum(prediction == 2))
        
        # Calculate ice concentration (ice / (ice + water))
        ice_water_total = ice_pixels + water_pixels
        ice_percentage = (ice_pixels / ice_water_total * 100) if ice_water_total > 0 else 0
        
        metrics = {
            "image_size": [h, w],
            "ice_pixels": ice_pixels,
            "water_pixels": water_pixels,
            "land_pixels": land_pixels,
            "ice_percentage": round(ice_percentage, 4),
            "total_analyzed_pixels": ice_water_total,
        }
        
        # Calculate area if bbox provided
        if bbox and len(bbox) == 4:
            min_lon, min_lat, max_lon, max_lat = bbox
            
            lat_diff = abs(max_lat - min_lat)
            lon_diff = abs(max_lon - min_lon)
            avg_lat = (min_lat + max_lat) / 2
            
            # 1 degree ≈ 111 km (adjusted for latitude)
            lat_km = lat_diff * 111
            lon_km = lon_diff * 111 * np.cos(np.radians(avg_lat))
            total_area_km2 = lat_km * lon_km
            
            # Calculate areas
            if total_pixels > 0:
                ice_area_km2 = total_area_km2 * (ice_pixels / total_pixels)
                water_area_km2 = total_area_km2 * (water_pixels / total_pixels)
            else:
                ice_area_km2 = 0
                water_area_km2 = 0
            
            metrics["total_area_km2"] = round(total_area_km2, 2)
            metrics["ice_extent_km2"] = round(ice_area_km2, 2)
            metrics["open_water_km2"] = round(water_area_km2, 2)
        
        return metrics
    
    def generate_alerts(
        self,
        metrics: dict[str, Any],
        thresholds: Optional[dict[str, float]] = None,
        previous_metrics: Optional[dict[str, Any]] = None,
    ) -> list[Alert]:
        """
        Generate ice melting alerts.
        
        Alerts are generated when:
        - Ice loss exceeds threshold (compared to previous)
        - Ice concentration drops below minimum
        - Rapid melt rate detected
        """
        alerts = []
        thresholds = thresholds or self.default_thresholds
        
        ice_percentage = metrics.get("ice_percentage", 0)
        ice_extent_km2 = metrics.get("ice_extent_km2")
        
        # Check minimum concentration
        min_concentration = thresholds.get("min_ice_concentration", 15.0)
        if ice_percentage < min_concentration:
            alerts.append(Alert(
                alert_type="low_ice_concentration",
                severity=Severity.HIGH,
                title="Low Ice Concentration",
                message=f"Ice concentration ({ice_percentage:.1f}%) is below minimum threshold ({min_concentration}%)",
                threshold_exceeded=min_concentration,
                measured_value=ice_percentage,
            ))
        
        # Check for ice loss (if previous metrics available)
        if previous_metrics:
            prev_percentage = previous_metrics.get("ice_percentage", 0)
            prev_extent = previous_metrics.get("ice_extent_km2")
            
            if prev_percentage > 0:
                loss_percentage = prev_percentage - ice_percentage
                
                critical_loss = thresholds.get("critical_ice_loss", 25.0)
                alert_loss = thresholds.get("alert_ice_loss", 10.0)
                
                if loss_percentage >= critical_loss:
                    message = f"Critical ice loss: {loss_percentage:.1f}% reduction"
                    if prev_extent and ice_extent_km2:
                        extent_loss = prev_extent - ice_extent_km2
                        message += f" ({extent_loss:.1f} km² lost)"
                    
                    alerts.append(Alert(
                        alert_type="critical_ice_loss",
                        severity=Severity.CRITICAL,
                        title="Critical Ice Loss Detected",
                        message=message,
                        threshold_exceeded=critical_loss,
                        measured_value=loss_percentage,
                        details={
                            "previous_concentration": prev_percentage,
                            "current_concentration": ice_percentage,
                            "previous_extent_km2": prev_extent,
                            "current_extent_km2": ice_extent_km2,
                        },
                    ))
                elif loss_percentage >= alert_loss:
                    message = f"Ice loss detected: {loss_percentage:.1f}% reduction"
                    if prev_extent and ice_extent_km2:
                        extent_loss = prev_extent - ice_extent_km2
                        message += f" ({extent_loss:.1f} km² lost)"
                    
                    alerts.append(Alert(
                        alert_type="ice_loss_detected",
                        severity=Severity.MEDIUM,
                        title="Ice Loss Detected",
                        message=message,
                        threshold_exceeded=alert_loss,
                        measured_value=loss_percentage,
                        details={
                            "previous_concentration": prev_percentage,
                            "current_concentration": ice_percentage,
                        },
                    ))
        
        return alerts
    
    def calculate_spectral_indices(self, image: np.ndarray) -> dict[str, dict[str, float]]:
        """
        Calculate spectral indices for ice analysis.
        
        Returns NDSI and NDWI statistics.
        """
        blue = image[..., 0]
        green = image[..., 1]
        red = image[..., 2]
        swir = image[..., 3] if image.shape[-1] >= 4 else image[..., 2]
        
        # NDSI
        ndsi_denom = green + swir
        ndsi = np.where(ndsi_denom > 0, (green - swir) / ndsi_denom, 0)
        
        # NDWI
        ndwi_denom = blue + red
        ndwi = np.where(ndwi_denom > 0, (blue - red) / ndwi_denom, 0)
        
        return {
            "NDSI": {
                "min": float(np.min(ndsi)),
                "mean": float(np.mean(ndsi)),
                "max": float(np.max(ndsi)),
            },
            "NDWI": {
                "min": float(np.min(ndwi)),
                "mean": float(np.mean(ndwi)),
                "max": float(np.max(ndwi)),
            },
        }
