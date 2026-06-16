"""
SAR-based flood detection analysis (Sentinel-1 VV/VH).

Wraps the physics/statistics-based EnsembleFloodPipeline (LIST + DLR + TUW) and
the permanent-vs-flood classifier behind the standard BaseAnalysisType contract,
so it is discoverable through the registry and runnable through the API.

Unlike the optical FloodingAnalysis (MNDWI), this works in cloud and at night
(SAR is all-weather) and -- given a permanent-water reference or a pre-event
scene -- genuinely separates flood water from permanent water rather than
guessing from index magnitude.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from climatevision.analysis.base import Alert, BaseAnalysisType, Severity
from climatevision.analysis.flooding_ensemble import EnsembleFloodPipeline
from climatevision.data.sar_preprocessing import preprocess_sar

logger = logging.getLogger(__name__)

DRY_LAND = 0
PERMANENT_WATER = 1
FLOODED = 2


class FloodingSARAnalysis(BaseAnalysisType):
    """All-weather SAR flood detection via a 3-algorithm ensemble."""

    name = "flooding_sar"
    display_name = "Flood Detection (SAR)"
    description = "All-weather flood detection from Sentinel-1 VV/VH using a physics-based ensemble"

    # Sentinel-1 dual-pol bands.
    required_bands = ["VV", "VH"]
    output_classes = ["dry_land", "permanent_water", "flooded"]
    enabled = True

    default_thresholds = {
        "alert_flood_area": 5.0,
        "critical_flood_area": 20.0,
    }

    def __init__(
        self,
        permanent_water_ref: Optional[np.ndarray] = None,
        pre_event_vh: Optional[np.ndarray] = None,
    ):
        """
        Args:
            permanent_water_ref: Optional (H, W) binary mask of normally-present
                water (e.g. from JRC GSW occurrence). Authority for the
                permanent/flood split when provided.
            pre_event_vh: Optional (H, W) pre-event VH (dB) for change detection,
                used when no reference is supplied.
        """
        self.permanent_water_ref = permanent_water_ref
        self.pre_event_vh = pre_event_vh
        self._pipeline = EnsembleFloodPipeline()

    def preprocess(self, image: np.ndarray, bands: Optional[list[str]] = None) -> np.ndarray:
        """Speckle-filter and convert VV/VH to dB. Returns (2, H, W)."""
        is_valid, error = self.validate_input(image)
        if not is_valid:
            raise ValueError(error)

        # Normalise to (C, H, W) with C=2 (VV, VH).
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] in (2, 3) and arr.shape[-1] < arr.shape[0]:
            arr = np.transpose(arr, (2, 0, 1))
        if arr.ndim == 3 and arr.shape[0] > 2:
            arr = arr[:2]
        if arr.ndim == 2:
            arr = np.stack([arr, arr], axis=0)

        # S1_GRD is already in dB; only apply speckle filtering here to avoid
        # double log-scaling. (Linear input should set to_db=True upstream.)
        return preprocess_sar(arr, apply_filter=True, to_db=False)

    def run_inference(
        self, image: np.ndarray, model: Optional[Any] = None,
    ) -> tuple[np.ndarray, float]:
        """Run the ensemble on the VH band and classify permanent vs flood.

        Returns (prediction, confidence) where prediction is the 3-class map
        (0=dry, 1=permanent_water, 2=flooded). If neither a permanent-water
        reference nor a pre-event scene is available, permanent water cannot be
        separated and detected water is reported as class 2 (flooded) with a
        lowered confidence to signal the ambiguity.
        """
        vh = image[1] if image.ndim == 3 else image

        out = self._pipeline.detect(
            post_vh=vh,
            pre_vh=self.pre_event_vh,
            permanent_water_ref=self.permanent_water_ref,
        )
        classified = out["classified_mask"]

        water_frac = float(out["ensemble_mask"].mean())
        if classified is not None:
            # Higher confidence when we could actually resolve permanent vs flood.
            confidence = round(min(1.0, 0.7 + 0.3 * water_frac), 4)
            return classified.astype(np.int32), confidence

        # No reference: cannot distinguish -> mark water as flooded, flag via confidence.
        prediction = (out["ensemble_mask"].astype(np.int32)) * FLOODED
        logger.warning(
            "flooding_sar: no permanent-water reference or pre-event scene; "
            "reporting detected water as flooded (permanent/flood unresolved)."
        )
        return prediction, round(min(1.0, 0.5 + 0.2 * water_frac), 4)

    def calculate_metrics(
        self, prediction: np.ndarray, image_size: tuple[int, int], bbox: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        h, w = image_size
        total = h * w
        dry = int(np.sum(prediction == DRY_LAND))
        permanent = int(np.sum(prediction == PERMANENT_WATER))
        flooded = int(np.sum(prediction == FLOODED))

        flooded_pct = (flooded / total * 100) if total else 0.0
        permanent_pct = (permanent / total * 100) if total else 0.0

        metrics: dict[str, Any] = {
            "image_size": [h, w],
            "dry_pixels": dry,
            "permanent_water_pixels": permanent,
            "flooded_pixels": flooded,
            "flooded_percentage": round(flooded_pct, 4),
            "permanent_water_percentage": round(permanent_pct, 4),
            "permanent_flood_distinguished": bool(permanent > 0 or self.permanent_water_ref is not None
                                                  or self.pre_event_vh is not None),
        }

        if bbox and len(bbox) == 4:
            min_lon, min_lat, max_lon, max_lat = bbox
            avg_lat = (min_lat + max_lat) / 2
            lat_km = abs(max_lat - min_lat) * 111
            lon_km = abs(max_lon - min_lon) * 111 * np.cos(np.radians(avg_lat))
            area = lat_km * lon_km
            if total:
                metrics["total_area_km2"] = round(area, 2)
                metrics["flooded_area_km2"] = round(area * flooded / total, 2)
                metrics["permanent_water_km2"] = round(area * permanent / total, 2)
        return metrics

    def generate_alerts(
        self, metrics: dict[str, Any], thresholds: Optional[dict[str, float]] = None,
        previous_metrics: Optional[dict[str, Any]] = None,
    ) -> list[Alert]:
        thresholds = thresholds or self.default_thresholds
        flooded_pct = metrics.get("flooded_percentage", 0.0)
        flooded_km2 = metrics.get("flooded_area_km2")
        critical = thresholds.get("critical_flood_area", 20.0)
        alert_at = thresholds.get("alert_flood_area", 5.0)

        alerts: list[Alert] = []
        if flooded_pct >= critical:
            msg = f"Critical flooding: {flooded_pct:.1f}% of area flooded"
            if flooded_km2:
                msg += f" ({flooded_km2:.1f} km²)"
            alerts.append(Alert(
                alert_type="critical_flooding", severity=Severity.CRITICAL,
                title="Critical Flooding Detected", message=msg,
                threshold_exceeded=critical, measured_value=flooded_pct,
            ))
        elif flooded_pct >= alert_at:
            msg = f"Flooding detected: {flooded_pct:.1f}% of area flooded"
            if flooded_km2:
                msg += f" ({flooded_km2:.1f} km²)"
            alerts.append(Alert(
                alert_type="flooding_detected", severity=Severity.HIGH,
                title="Flooding Detected", message=msg,
                threshold_exceeded=alert_at, measured_value=flooded_pct,
            ))
        return alerts
