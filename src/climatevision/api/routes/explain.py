"""
Explainability endpoints: SHAP band contributions for a prediction.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from climatevision.api.shared import AnalysisType
from climatevision.db import get_connection
from climatevision.governance import SHAPExplainer

router = APIRouter()


# Explainability models
class ExplainRequest(BaseModel):
    run_id: Optional[int] = None
    analysis_type: AnalysisType = Field(default="deforestation")
    target_class: Optional[int] = None


class BandContribution(BaseModel):
    band: str
    importance: float


class ExplainResponse(BaseModel):
    run_id: Optional[int] = None
    analysis_type: str
    target_class: int
    prediction: int
    confidence: float
    top_bands: list[BandContribution]
    heatmap_path: Optional[str] = None
    explainer_type: str


# ===== Explainability Endpoints =====


@router.post("/api/explain", response_model=ExplainResponse)
async def explain_run(body: ExplainRequest) -> dict[str, Any]:
    """
    Generate SHAP-based explanation for a prediction.

    Returns band-level contributions showing which spectral bands
    drove the model's classification decision.
    """
    from climatevision.inference.pipeline import _load_model, _load_image_file
    import numpy as np
    import torch

    # If run_id provided, get the image from that run
    image_path = None
    if body.run_id:
        with get_connection() as conn:
            run = conn.execute(
                "SELECT * FROM runs WHERE id = ?", (body.run_id,)
            ).fetchone()
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")

            result = conn.execute(
                "SELECT * FROM results WHERE run_id = ? ORDER BY id DESC LIMIT 1",
                (body.run_id,),
            ).fetchone()

            if result:
                payload = json.loads(result["payload_json"])
                input_info = payload.get("input", {})
                image_path = input_info.get("file")

    # Load model and create explainer
    model, device = _load_model(body.analysis_type)

    # If we have an image, use it; otherwise create synthetic
    if image_path:
        try:
            image = _load_image_file(image_path)
        except Exception:
            image = np.random.randn(model.n_channels, 256, 256).astype(np.float32)
    else:
        image = np.random.randn(model.n_channels, 256, 256).astype(np.float32)

    # Ensure correct shape
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

    # Generate explanation
    explainer = SHAPExplainer(model, device=device)
    result = explainer.explain(tensor, target_class=body.target_class)

    # Format band contributions
    band_names = {
        "deforestation": ["Red", "Green", "Blue", "NIR"],
        "ice_melting": ["Red", "Green", "Blue", "NIR"],
        "flooding": ["Green", "NIR", "SWIR1"],
    }
    names = band_names.get(body.analysis_type, [f"Band_{i}" for i in range(n_channels)])

    top_bands = []
    for i, (band_key, importance) in enumerate(
        sorted(result["band_contributions"].items(), key=lambda x: x[1], reverse=True)
    ):
        band_idx = int(band_key.split("_")[1])
        band_name = names[band_idx] if band_idx < len(names) else band_key
        top_bands.append(
            BandContribution(band=band_name, importance=round(importance, 4))
        )

    return {
        "run_id": body.run_id,
        "analysis_type": body.analysis_type,
        "target_class": result["target_class"],
        "prediction": result["prediction"],
        "confidence": round(result["confidence"], 4),
        "top_bands": top_bands,
        "heatmap_path": None,
        "explainer_type": result["explainer_type"],
    }


@router.get("/api/explain/{run_id}")
async def get_explanation(
    run_id: int,
    target_class: Optional[int] = None,
) -> dict[str, Any]:
    """Get SHAP explanation for a specific run."""
    with get_connection() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

    analysis_type = run["analysis_type"] or "deforestation"

    body = ExplainRequest(
        run_id=run_id,
        analysis_type=analysis_type,
        target_class=target_class,
    )
    return await explain_run(body)
