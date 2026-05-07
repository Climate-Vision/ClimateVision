"""
Datasheets for the datasets that train ClimateVision models.

Companion to the Mitchell-style model cards in ``governance.model_card``:
where a model card describes the *model*, a datasheet describes the
*dataset* the model was trained on (Gebru et al., 2018, "Datasheets for
Datasets"). The two artifacts answer different questions and both need
to ship with a release.

The module mirrors the model_card public surface (``build``, ``render``,
``write``, ``generate``) so contributors only have to learn one pattern,
and the release CI pipeline can call them in sequence.

Sections covered:

- Motivation
- Composition
- Collection process
- Preprocessing, cleaning, labeling
- Uses (intended and inappropriate)
- Distribution
- Maintenance

Every section is a free-form ``dict`` of question -> answer so the schema
can grow without code changes; ``REQUIRED_QUESTIONS`` enforces the bare
minimum a release datasheet must answer.
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
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "datasheets"

REQUIRED_QUESTIONS = {
    "motivation": ("purpose", "creators"),
    "composition": ("instances", "labels", "splits"),
    "collection_process": ("source", "timeframe"),
    "uses": ("intended_uses", "inappropriate_uses"),
}


@dataclass
class Datasheet:
    """Structured datasheet for a single training dataset."""

    name: str
    version: str
    motivation: dict
    composition: dict
    collection_process: dict
    preprocessing: dict
    uses: dict
    distribution: dict
    maintenance: dict
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "motivation": self.motivation,
            "composition": self.composition,
            "collection_process": self.collection_process,
            "preprocessing": self.preprocessing,
            "uses": self.uses,
            "distribution": self.distribution,
            "maintenance": self.maintenance,
            "generated_at": self.generated_at,
        }


_DEFAULT_INAPPROPRIATE_USES = [
    "Training models for real-time legal enforcement against individual landowners.",
    "Land-rights or sovereignty disputes without on-the-ground verification.",
    "Generative model training where label provenance is required to be human-verified.",
]

_DEFAULT_MAINTENANCE = {
    "owner": "ClimateVision Governance <governance@climate-vision.org>",
    "update_cadence": "Reviewed each minor release; refreshed when source providers change.",
    "deprecation_policy": (
        "Versions are retained for two minor releases after supersession; "
        "models trained on deprecated versions are flagged in their model cards."
    ),
}


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


def _validate(datasheet: "Datasheet") -> None:
    missing: list[str] = []
    for section_name, required_keys in REQUIRED_QUESTIONS.items():
        section = getattr(datasheet, section_name)
        for key in required_keys:
            if key not in section or section[key] in (None, "", []):
                missing.append(f"{section_name}.{key}")
    if missing:
        raise ValueError(f"datasheet missing required answers: {missing}")


def build_datasheet(
    manifest: Union[dict, str, Path],
    *,
    name: Optional[str] = None,
    version: Optional[str] = None,
) -> Datasheet:
    """Build a Datasheet from a structured dataset manifest."""
    m = _coerce_config(manifest)

    resolved_name = name or m.get("name") or "climatevision-dataset"
    resolved_version = version or m.get("version") or "0.0.0"

    uses = dict(m.get("uses", {}))
    uses.setdefault("inappropriate_uses", list(_DEFAULT_INAPPROPRIATE_USES))

    sheet = Datasheet(
        name=resolved_name,
        version=resolved_version,
        motivation=dict(m.get("motivation", {})),
        composition=dict(m.get("composition", {})),
        collection_process=dict(m.get("collection_process", {})),
        preprocessing=dict(m.get("preprocessing", {})),
        uses=uses,
        distribution=dict(m.get("distribution", {})),
        maintenance=dict(m.get("maintenance", _DEFAULT_MAINTENANCE)),
    )
    _validate(sheet)
    return sheet


def _render_section(title: str, body: dict) -> list[str]:
    if not body:
        return [f"## {title}", "_Not documented._", ""]
    lines = [f"## {title}"]
    for key, value in body.items():
        pretty_key = key.replace("_", " ").title()
        if isinstance(value, list):
            lines.append(f"### {pretty_key}")
            lines.extend(f"- {item}" for item in value)
        elif isinstance(value, dict):
            lines.append(f"### {pretty_key}")
            lines.append(f"```json\n{json.dumps(value, indent=2)}\n```")
        else:
            lines.append(f"- **{pretty_key}**: {value}")
    lines.append("")
    return lines


def render_markdown(sheet: Datasheet) -> str:
    sections = [
        f"# Datasheet: {sheet.name} ({sheet.version})",
        f"_Generated {sheet.generated_at}_",
        "",
        "_Format: Gebru et al., 2018, \"Datasheets for Datasets\"._",
        "",
    ]
    sections += _render_section("Motivation", sheet.motivation)
    sections += _render_section("Composition", sheet.composition)
    sections += _render_section("Collection Process", sheet.collection_process)
    sections += _render_section("Preprocessing, Cleaning, Labeling", sheet.preprocessing)
    sections += _render_section("Uses", sheet.uses)
    sections += _render_section("Distribution", sheet.distribution)
    sections += _render_section("Maintenance", sheet.maintenance)
    return "\n".join(sections) + "\n"


def write_datasheet(
    sheet: Datasheet,
    output_dir: Optional[Union[str, Path]] = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    base = f"{sheet.name}_{sheet.version}"
    md_path = output_dir / f"{base}.md"
    json_path = output_dir / f"{base}.json"

    md_path.write_text(render_markdown(sheet))
    json_path.write_text(json.dumps(sheet.to_dict(), indent=2))

    logger.info("Wrote datasheet to %s and %s", md_path, json_path)
    return {"markdown": md_path, "json": json_path}


def generate(
    manifest: Union[dict, str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> dict[str, Path]:
    """End-to-end: load manifest, build the datasheet, render to disk."""
    sheet = build_datasheet(manifest, **kwargs)
    return write_datasheet(sheet, output_dir=output_dir)
