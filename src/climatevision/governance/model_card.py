"""
Automated model card generator for ClimateVision releases.

Builds a Google-style "Model Card" (Mitchell et al., 2019) from the
training config and an evaluation metrics blob. Output is rendered as
both Markdown (for the GitHub release notes / model registry) and JSON
(for programmatic consumption by downstream tooling).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "model_cards"

REQUIRED_METRICS = ("iou", "f1", "precision", "recall")


@dataclass
class ModelCard:
    name: str
    version: str
    analysis_type: str
    description: str
    intended_use: str
    out_of_scope_uses: list[str]
    training_data: dict
    evaluation_data: dict
    metrics: dict
    fairness: dict
    limitations: list[str]
    ethical_considerations: list[str]
    contact: str
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "analysis_type": self.analysis_type,
            "description": self.description,
            "intended_use": self.intended_use,
            "out_of_scope_uses": self.out_of_scope_uses,
            "training_data": self.training_data,
            "evaluation_data": self.evaluation_data,
            "metrics": self.metrics,
            "fairness": self.fairness,
            "limitations": self.limitations,
            "ethical_considerations": self.ethical_considerations,
            "contact": self.contact,
            "generated_at": self.generated_at,
        }


_DEFAULT_INTENDED_USE = (
    "Detection of {analysis_type} in satellite imagery for use by "
    "conservation organisations, NGOs, and government agencies. The "
    "model produces per-pixel probability scores intended to be reviewed "
    "alongside ground-truth reference data and analyst judgement."
)

_DEFAULT_OUT_OF_SCOPE = [
    "Real-time legal enforcement decisions without analyst review.",
    "Carbon credit issuance without independent ground-truth validation.",
    "Use on imagery from sensors not represented in the training set.",
]

_DEFAULT_LIMITATIONS = [
    "Performance degrades on cloud cover above the masking threshold used in preprocessing.",
    "Geographic coverage limited to regions present in the training set.",
    "Temporal generalisation to seasons or years outside the training window is unverified.",
]

_DEFAULT_ETHICS = [
    "Model outputs may carry geographic bias; downstream users must run "
    "the bias audit pipeline before distributing results across regions.",
    "Predictions should never be the sole basis for actions affecting "
    "indigenous land rights or local communities.",
]


def _coerce_config(config: Union[dict, str, Path]) -> dict:
    if isinstance(config, dict):
        return config
    path = Path(config)
    text = path.read_text()
    if path.suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("PyYAML is required to load YAML configs") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def _validate_metrics(metrics: dict) -> None:
    missing = [m for m in REQUIRED_METRICS if m not in metrics]
    if missing:
        raise ValueError(f"metrics missing required keys: {missing}")


def build_model_card(
    config: Union[dict, str, Path],
    metrics: dict,
    fairness_report: Optional[dict] = None,
    *,
    name: Optional[str] = None,
    version: Optional[str] = None,
    description: Optional[str] = None,
    contact: str = "ClimateVision Governance <governance@climate-vision.org>",
) -> ModelCard:
    cfg = _coerce_config(config)
    _validate_metrics(metrics)

    analysis_type = cfg.get("analysis_type") or cfg.get("analysis", {}).get("type", "deforestation")
    resolved_name = name or cfg.get("model", {}).get("name") or f"climatevision-{analysis_type}"
    resolved_version = version or cfg.get("model", {}).get("version") or "0.0.0"

    return ModelCard(
        name=resolved_name,
        version=resolved_version,
        analysis_type=analysis_type,
        description=description or f"U-Net segmentation model for {analysis_type}.",
        intended_use=_DEFAULT_INTENDED_USE.format(analysis_type=analysis_type),
        out_of_scope_uses=list(_DEFAULT_OUT_OF_SCOPE),
        training_data=cfg.get("training_data", cfg.get("data", {})),
        evaluation_data=cfg.get("evaluation_data", {}),
        metrics=dict(metrics),
        fairness=fairness_report or {},
        limitations=list(_DEFAULT_LIMITATIONS),
        ethical_considerations=list(_DEFAULT_ETHICS),
        contact=contact,
    )


def render_markdown(card: ModelCard) -> str:
    metrics_rows = "\n".join(
        f"| {k} | {v} |" for k, v in sorted(card.metrics.items())
    )
    fairness_block = (
        "\n".join(f"- **{k}**: {v}" for k, v in card.fairness.items())
        or "_No fairness report attached._"
    )

    sections = [
        f"# Model Card: {card.name} ({card.version})",
        f"_Generated {card.generated_at}_",
        "",
        "## Description",
        card.description,
        "",
        "## Intended Use",
        card.intended_use,
        "",
        "### Out-of-Scope Uses",
        "\n".join(f"- {u}" for u in card.out_of_scope_uses),
        "",
        "## Training Data",
        f"```json\n{json.dumps(card.training_data, indent=2)}\n```",
        "",
        "## Evaluation",
        "| Metric | Value |",
        "| --- | --- |",
        metrics_rows,
        "",
        "## Fairness",
        fairness_block,
        "",
        "## Limitations",
        "\n".join(f"- {l}" for l in card.limitations),
        "",
        "## Ethical Considerations",
        "\n".join(f"- {e}" for e in card.ethical_considerations),
        "",
        "## Contact",
        card.contact,
    ]
    return "\n".join(sections) + "\n"


def write_model_card(
    card: ModelCard,
    output_dir: Optional[Union[str, Path]] = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    base = f"{card.name}_{card.version}"
    md_path = output_dir / f"{base}.md"
    json_path = output_dir / f"{base}.json"

    md_path.write_text(render_markdown(card))
    json_path.write_text(json.dumps(card.to_dict(), indent=2))

    logger.info("Wrote model card to %s and %s", md_path, json_path)
    return {"markdown": md_path, "json": json_path}


def generate(
    config: Union[dict, str, Path],
    metrics: Union[dict, str, Path],
    fairness_report: Optional[Union[dict, str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> dict[str, Path]:
    """End-to-end: load inputs, build the card, render to disk."""
    metrics_dict = _coerce_config(metrics) if not isinstance(metrics, dict) else metrics
    fairness_dict = (
        _coerce_config(fairness_report)
        if fairness_report is not None and not isinstance(fairness_report, dict)
        else fairness_report
    )
    card = build_model_card(config, metrics_dict, fairness_dict, **kwargs)
    return write_model_card(card, output_dir=output_dir)
