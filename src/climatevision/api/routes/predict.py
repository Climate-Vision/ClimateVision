"""
Prediction endpoints: run inference from a bounding box or an uploaded image.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator, model_validator

from climatevision.api.auth import require_api_key
from climatevision.api.run_events import run_event_hub
from climatevision.api.shared import (
    PROJECT_ROOT,
    AnalysisType,
    _compute_carbon_estimation,
    _status_event_for,
    _utc_now_iso,
    _validate_forest_type,
    _validate_region,
)
from climatevision.db import get_connection
from climatevision.inference import run_inference_from_file, run_inference_from_gee
from climatevision.inference.flood_pipeline import run_flood_inference_from_gee

logger = logging.getLogger(__name__)

router = APIRouter()


class PredictRequest(BaseModel):
    kind: str = Field(default="demo")
    analysis_type: AnalysisType = Field(default="deforestation")
    bbox: Optional[list[float]] = None
    start_date: Optional[str] = Field(
        default=None,
        description="Start date in YYYY-MM-DD format. Must be earlier than end_date.",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date in YYYY-MM-DD format. Must be later than start_date.",
    )
    enable_carbon: bool = Field(
        default=False,
        description="If true, include carbon loss estimates for deforestation runs.",
    )
    forest_type: Optional[str] = Field(
        default=None,
        description="Forest type for carbon estimation (e.g. tropical_moist, mangrove).",
    )
    region: Optional[str] = Field(
        default=None,
        description="Region for carbon estimation adjustment (e.g. amazon, congo).",
    )

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError(
                "bbox must have exactly 4 values: [west, south, east, north]"
            )
        west, south, east, north = v
        if not (-180 <= west <= 180 and -180 <= east <= 180):
            raise ValueError("bbox longitude values must be between -180 and 180")
        if not (-90 <= south <= 90 and -90 <= north <= 90):
            raise ValueError("bbox latitude values must be between -90 and 90")
        if west >= east:
            raise ValueError("bbox west longitude must be less than east longitude")
        if south >= north:
            raise ValueError("bbox south latitude must be less than north latitude")
        return v

    @model_validator(mode="after")
    def validate_date_range(self) -> "PredictRequest":
        if self.start_date and self.end_date:
            try:
                start = datetime.strptime(self.start_date, "%Y-%m-%d")
                end = datetime.strptime(self.end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("start_date and end_date must be in YYYY-MM-DD format")
            if start >= end:
                raise ValueError("start_date must be earlier than end_date")
        return self


# ===== Helper Functions =====


def _load_template_result(
    *,
    bbox: Optional[list[float]],
    start_date: Optional[str],
    end_date: Optional[str],
    analysis_type: str = "deforestation",
) -> dict[str, Any]:
    """Load or create a template result for failed inference."""
    outputs_dir = PROJECT_ROOT / "outputs"
    template_path = outputs_dir / "inference_results.json"

    if template_path.exists():
        template: dict[str, Any] = json.loads(template_path.read_text(encoding="utf-8"))
    else:
        # Create analysis-specific template
        if analysis_type == "ice_melting":
            template = {
                "region": {"bbox": bbox or None},
                "inference": {
                    "image_size": [256, 256],
                    "ice_pixels": 0,
                    "water_pixels": 0,
                    "land_pixels": 0,
                    "ice_percentage": 0.0,
                    "mean_confidence": 0.0,
                },
            }
        elif analysis_type == "flooding":
            template = {
                "region": {"bbox": bbox or None},
                "inference": {
                    "image_size": [256, 256],
                    "flooded_pixels": 0,
                    "dry_pixels": 0,
                    "water_pixels": 0,
                    "flooded_percentage": 0.0,
                    "mean_confidence": 0.0,
                },
            }
        else:  # deforestation (default)
            template = {
                "region": {"bbox": bbox or None},
                "ndvi_stats": {"NDVI_min": 0.0, "NDVI_mean": 0.0, "NDVI_max": 0.0},
                "inference": {
                    "image_size": [256, 256],
                    "forest_pixels": 0,
                    "non_forest_pixels": 0,
                    "forest_percentage": 0.0,
                    "mean_confidence": 0.0,
                },
            }

    if bbox is not None:
        template.setdefault("region", {})["bbox"] = bbox
    if start_date and end_date:
        template.setdefault("region", {})["date_range"] = f"{start_date} to {end_date}"

    template["analysis_type"] = analysis_type

    return template


async def _persist_upload(*, run_id: int, file: UploadFile) -> str:
    """Save uploaded file to disk."""
    outputs_dir = PROJECT_ROOT / "outputs"
    uploads_dir = outputs_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / f"run_{run_id}_{file.filename}"
    dest.write_bytes(await file.read())
    return str(dest)


# ===== Prediction Endpoints =====


@router.post("/api/predict")
async def predict_json(
    body: PredictRequest,
    org: dict[str, Any] = Depends(require_api_key),
) -> dict[str, Any]:
    """Run prediction using bounding box and date range."""
    if body.enable_carbon:
        _validate_forest_type(body.forest_type)
        _validate_region(body.region)

    created_at = _utc_now_iso()
    bbox_json = json.dumps(body.bbox) if body.bbox else None

    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO runs (kind, status, analysis_type, bbox, start_date, end_date, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                body.kind,
                "running",
                body.analysis_type,
                bbox_json,
                body.start_date,
                body.end_date,
                created_at,
                created_at,
            ),
        )
        run_id = int(cur.lastrowid)

    # Run inference. SAR flood detection has its own Sentinel-1 + JRC pipeline;
    # all other analysis types use the shared Sentinel-2 inference path.
    try:
        if body.analysis_type == "flooding_sar":
            result_payload = run_flood_inference_from_gee(
                bbox=body.bbox,
                start_date=body.start_date,
                end_date=body.end_date,
            )
        else:
            result_payload = run_inference_from_gee(
                bbox=body.bbox,
                start_date=body.start_date,
                end_date=body.end_date,
                analysis_type=body.analysis_type,
            )
        result_payload["analysis_type"] = body.analysis_type
        status = "completed"
    except Exception as exc:
        logger.exception("Inference failed for run %s", run_id)
        result_payload = _load_template_result(
            bbox=body.bbox,
            start_date=body.start_date,
            end_date=body.end_date,
            analysis_type=body.analysis_type,
        )
        result_payload["error"] = str(exc)
        status = "failed"

    # Carbon estimation (feature-flagged). Degrades gracefully if the
    # analytics module is unavailable or the payload lacks pixel counts.
    if (
        body.enable_carbon
        and status == "completed"
        and body.analysis_type == "deforestation"
    ):
        carbon = _compute_carbon_estimation(
            result_payload,
            forest_type=body.forest_type,
            region=body.region,
        )
        if carbon is not None:
            result_payload["carbon_estimation"] = carbon

    # Persist result
    result_created_at = _utc_now_iso()
    with get_connection() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
            (status, result_created_at, run_id),
        )
        conn.execute(
            """
            INSERT INTO results (run_id, payload_json, mask_path, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, json.dumps(result_payload), None, result_created_at),
        )

    await run_event_hub.publish(
        run_id, _status_event_for(run_id, status, result_payload)
    )

    return {"run_id": run_id, "result": result_payload}


@router.post("/api/predict/upload")
async def predict_upload(
    kind: str = Form(default="upload"),
    org: dict[str, Any] = Depends(require_api_key),
    analysis_type: str = Form(default="deforestation"),
    bbox: str | None = Form(default=None),
    start_date: str | None = Form(default=None),
    end_date: str | None = Form(default=None),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Run prediction on uploaded satellite imagery file."""
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400, detail="start_date must be before end_date"
        )

    created_at = _utc_now_iso()

    parsed_bbox: Optional[list[float]] = None
    if bbox:
        try:
            parsed_bbox = json.loads(bbox)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail="Invalid bbox JSON") from e

    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO runs (kind, status, analysis_type, bbox, start_date, end_date, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kind,
                "running",
                analysis_type,
                json.dumps(parsed_bbox) if parsed_bbox else None,
                start_date,
                end_date,
                created_at,
                created_at,
            ),
        )
        run_id = int(cur.lastrowid)

    dest = await _persist_upload(run_id=run_id, file=file)

    # Run inference
    try:
        result_payload = run_inference_from_file(
            dest,
            bbox=parsed_bbox,
            start_date=start_date,
            end_date=end_date,
            analysis_type=analysis_type,
        )
        result_payload["analysis_type"] = analysis_type
        status = "completed"
    except Exception as exc:
        logger.exception("Inference failed for upload run %s", run_id)
        result_payload = _load_template_result(
            bbox=parsed_bbox,
            start_date=start_date,
            end_date=end_date,
            analysis_type=analysis_type,
        )
        result_payload.setdefault("input", {})["file"] = dest
        result_payload["error"] = str(exc)
        status = "failed"

    # Persist result
    result_created_at = _utc_now_iso()
    with get_connection() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
            (status, result_created_at, run_id),
        )
        conn.execute(
            """
            INSERT INTO results (run_id, payload_json, mask_path, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, json.dumps(result_payload), None, result_created_at),
        )

    await run_event_hub.publish(
        run_id, _status_event_for(run_id, status, result_payload)
    )

    return {"run_id": run_id, "result": result_payload}
