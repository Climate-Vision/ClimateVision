"""
Health endpoints: API status, config validation and model availability.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from climatevision.api.routes.analysis import SUPPORTED_ANALYSIS_TYPES
from climatevision.api.shared import PROJECT_ROOT

router = APIRouter()


@router.get("/api/health")
def health() -> dict[str, Any]:
    """Health check endpoint with API information and config validation."""
    from climatevision.data.band_mapping import get_model_config

    enabled_types = [t for t in SUPPORTED_ANALYSIS_TYPES if t["enabled"]]
    config_issues: list[dict[str, Any]] = []

    for atype in enabled_types:
        name = atype["name"]
        try:
            cfg = get_model_config(name)
            expected_channels = len(atype["bands"])
            expected_classes = len(atype["classes"])
            if cfg.get("in_channels") != expected_channels:
                config_issues.append(
                    {
                        "analysis_type": name,
                        "issue": "in_channels mismatch",
                        "expected": expected_channels,
                        "got": cfg.get("in_channels"),
                    }
                )
            if cfg.get("num_classes") != expected_classes:
                config_issues.append(
                    {
                        "analysis_type": name,
                        "issue": "num_classes mismatch",
                        "expected": expected_classes,
                        "got": cfg.get("num_classes"),
                    }
                )
        except Exception as exc:
            config_issues.append(
                {"analysis_type": name, "issue": "config missing", "error": str(exc)}
            )

    health_status = "ok" if not config_issues else "degraded"

    return {
        "status": health_status,
        "version": "0.2.0",
        "analysis_types": [t["name"] for t in enabled_types],
        "config_valid": len(config_issues) == 0,
        "config_issues": config_issues,
    }


@router.get("/api/health/models")
def health_models() -> dict[str, Any]:
    """Report which analysis types have trained weights available on disk."""
    from climatevision.data.band_mapping import get_model_config

    models_status = []
    for atype in SUPPORTED_ANALYSIS_TYPES:
        name = atype["name"]
        cfg = get_model_config(name)
        weights_path = cfg.get("weights")
        has_weights = False
        if weights_path:
            has_weights = (PROJECT_ROOT / weights_path).exists()
        models_status.append(
            {
                "analysis_type": name,
                "enabled": atype["enabled"],
                "has_trained_weights": has_weights,
                "weights_path": weights_path,
                "note": (
                    "production inference" if has_weights else "synthetic/demo fallback"
                ),
            }
        )

    ready_count = sum(1 for m in models_status if m["has_trained_weights"])
    return {
        "status": "ready" if ready_count else "demo_mode",
        "models": models_status,
        "ready_count": ready_count,
        "total_count": len(models_status),
    }
