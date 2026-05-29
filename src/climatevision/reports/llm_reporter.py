"""
LLM-backed impact report generation for ClimateVision.

`LLMReporter` turns a structured prediction record (carbon analytics,
SHAP attributions, validation metrics, fairness flags) into a
narrative report ready for NGOs and government stakeholders.

A deterministic template-based renderer is always available so that
the module never blocks the pipeline when an LLM provider is
unreachable. When a provider is configured, the template output is
used as the prompt skeleton and the LLM smooths it into prose.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "reports"


@dataclass
class ReportContext:
    """Inputs the reporter draws on to compose an impact report."""

    region: str
    period: str
    analysis_type: str
    carbon: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    shap: dict = field(default_factory=dict)
    fairness: dict = field(default_factory=dict)
    run_id: Optional[Union[int, str]] = None

    def headline_metric(self) -> str:
        hectares = self.carbon.get("hectares")
        carbon_t = self.carbon.get("carbon_tonnes")
        if hectares is not None and carbon_t is not None:
            return (
                f"{hectares:,.1f} hectares of {self.analysis_type.replace('_', ' ')} "
                f"detected, equivalent to {carbon_t:,.1f} tCO2e."
            )
        if hectares is not None:
            return f"{hectares:,.1f} hectares of {self.analysis_type.replace('_', ' ')} detected."
        return f"Analysis run for {self.analysis_type} in {self.region} ({self.period})."


@dataclass
class ImpactReport:
    summary: str
    body: str
    context: ReportContext
    provider: str
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["context"] = asdict(self.context)
        return d


# Type alias for an LLM call: prompt -> completion
LLMCallable = Callable[[str], str]


def render_template(context: ReportContext, *, include_shap: bool = True) -> str:
    """Deterministic Markdown template — used both as a fallback and as an LLM prompt seed."""

    lines = [
        f"# Impact Report — {context.region.title()} ({context.period})",
        "",
        f"**Headline:** {context.headline_metric()}",
        "",
        "## Carbon Analytics",
    ]

    if context.carbon:
        for k, v in context.carbon.items():
            lines.append(f"- **{k.replace('_', ' ').title()}**: {v}")
    else:
        lines.append("- _Carbon analytics not provided._")

    lines += ["", "## Validation"]
    if context.validation:
        for k, v in context.validation.items():
            lines.append(f"- **{k.upper()}**: {v}")
    else:
        lines.append("- _No validation metrics attached._")

    if include_shap:
        lines += ["", "## Explainability"]
        if context.shap:
            top_bands = context.shap.get("top_bands", [])
            if top_bands:
                bands = ", ".join(b["band"] if isinstance(b, dict) else str(b) for b in top_bands)
                lines.append(f"- Most influential bands: {bands}")
            for k, v in context.shap.items():
                if k == "top_bands":
                    continue
                lines.append(f"- **{k.replace('_', ' ').title()}**: {v}")
        else:
            lines.append("- _No SHAP explanation attached._")

    if context.fairness:
        lines += ["", "## Fairness"]
        for k, v in context.fairness.items():
            lines.append(f"- **{k.replace('_', ' ').title()}**: {v}")

    return "\n".join(lines) + "\n"


def _build_prompt(context: ReportContext, template: str) -> str:
    return (
        "You are drafting a concise, factual environmental-impact report for "
        "conservation organisations and government stakeholders.\n\n"
        "Rules:\n"
        "- Do not invent numbers; only restate values from the data block below.\n"
        "- Keep tone neutral and policy-relevant; no promotional language.\n"
        "- Output Markdown with the same section structure as the seed.\n"
        "- Open with a 2–3 sentence executive summary.\n\n"
        f"DATA (JSON):\n```json\n{json.dumps(asdict(context), indent=2, default=str)}\n```\n\n"
        f"SEED:\n{template}\n\n"
        "FINAL REPORT:\n"
    )


class LLMReporter:
    """
    Reporter with pluggable LLM backend.

    Pass an `llm` callable (prompt -> string) to use a custom provider.
    Without one, set CLIMATEVISION_LLM_PROVIDER=anthropic and
    ANTHROPIC_API_KEY to use Anthropic's API; otherwise the template
    renderer alone is used.
    """

    def __init__(self, llm: Optional[LLMCallable] = None) -> None:
        self._llm = llm

    def _call_llm(self, prompt: str) -> Optional[str]:
        if self._llm is not None:
            try:
                return self._llm(prompt)
            except Exception as exc:  # pragma: no cover - external call
                logger.exception("user-provided LLM callable raised: %s", exc)
                return None

        provider = os.environ.get("CLIMATEVISION_LLM_PROVIDER", "").lower()
        if provider == "anthropic":
            return self._call_anthropic(prompt)
        return None

    def _call_anthropic(self, prompt: str) -> Optional[str]:  # pragma: no cover - external call
        try:
            import anthropic
        except ImportError:
            logger.warning("anthropic package not installed; using template only")
            return None

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set; using template only")
            return None

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=os.environ.get("CLIMATEVISION_LLM_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [b.text for b in message.content if getattr(b, "type", None) == "text"]
        return "".join(parts) if parts else None

    def generate(
        self,
        context: ReportContext,
        *,
        include_shap: bool = True,
    ) -> ImpactReport:
        template = render_template(context, include_shap=include_shap)
        prompt = _build_prompt(context, template)
        llm_text = self._call_llm(prompt)

        if llm_text:
            body = llm_text.strip()
            provider = "llm"
        else:
            body = template
            provider = "template"

        first_para = body.strip().split("\n\n", 1)[0]
        summary = first_para.replace("\n", " ").strip()

        return ImpactReport(
            summary=summary,
            body=body,
            context=context,
            provider=provider,
        )


def generate_impact_report(
    region: str,
    period: str,
    analysis_type: str = "deforestation",
    carbon: Optional[dict] = None,
    validation: Optional[dict] = None,
    shap: Optional[dict] = None,
    fairness: Optional[dict] = None,
    run_id: Optional[Union[int, str]] = None,
    *,
    llm: Optional[LLMCallable] = None,
    include_shap: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
) -> ImpactReport:
    """High-level entry point used by the API and CLI."""
    ctx = ReportContext(
        region=region,
        period=period,
        analysis_type=analysis_type,
        carbon=carbon or {},
        validation=validation or {},
        shap=shap or {},
        fairness=fairness or {},
        run_id=run_id,
    )
    report = LLMReporter(llm=llm).generate(ctx, include_shap=include_shap)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base = f"{region}_{period}_impact"
        (output_dir / f"{base}.md").write_text(report.body)
        (output_dir / f"{base}.json").write_text(json.dumps(report.to_dict(), indent=2, default=str))

    return report


def _default_output_dir() -> Path:
    return _DEFAULT_OUTPUT_DIR
