#!/usr/bin/env python
"""
Governance CI gate for ClimateVision model releases.

Reads an evaluation metrics JSON, an optional fairness report, and an
optional security scan report, and decides whether the release is
allowed to proceed.

Exit codes:
    0   all gates passed
    1   one or more gates failed (CI must fail the build)
    2   bad invocation / missing inputs

Threshold defaults can be overridden via --thresholds (JSON file). The
script prints a Markdown summary of which gates passed and which
failed; CI systems can capture this and post it back to the PR.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("governance_ci_gate")

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "metrics": {
        "iou": 0.70,
        "f1": 0.75,
    },
    "fairness": {
        "min_score": 0.80,
        "max_disparity_regions": 0,
    },
    "security": {
        "max_high": 0,
        "max_critical": 0,
    },
}

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_BAD_INPUT = 2


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def evaluate_metrics_gate(metrics: dict, thresholds: dict) -> list[GateResult]:
    results: list[GateResult] = []
    for metric, floor in thresholds.items():
        value = metrics.get(metric)
        if value is None:
            results.append(
                GateResult(
                    name=f"metrics.{metric}",
                    passed=False,
                    detail=f"missing metric '{metric}'",
                )
            )
            continue
        passed = value >= floor
        results.append(
            GateResult(
                name=f"metrics.{metric}",
                passed=passed,
                detail=f"value={value:.3f} threshold>={floor:.3f}",
            )
        )
    return results


def evaluate_fairness_gate(report: dict, thresholds: dict) -> list[GateResult]:
    results: list[GateResult] = []
    score = report.get("score")
    if score is not None:
        passed = score >= thresholds["min_score"]
        results.append(
            GateResult(
                name="fairness.score",
                passed=passed,
                detail=f"score={score:.3f} threshold>={thresholds['min_score']:.3f}",
            )
        )
    else:
        results.append(
            GateResult(
                name="fairness.score",
                passed=False,
                detail="missing score",
            )
        )

    disparity = report.get("disparity_regions") or []
    passed = len(disparity) <= thresholds["max_disparity_regions"]
    results.append(
        GateResult(
            name="fairness.disparity_regions",
            passed=passed,
            detail=f"count={len(disparity)} threshold<={thresholds['max_disparity_regions']}",
        )
    )
    return results


def evaluate_security_gate(report: dict, thresholds: dict) -> list[GateResult]:
    findings = report.get("findings", [])
    high = sum(1 for f in findings if f.get("severity") == "high")
    critical = sum(1 for f in findings if f.get("severity") == "critical")

    return [
        GateResult(
            name="security.high",
            passed=high <= thresholds["max_high"],
            detail=f"high={high} threshold<={thresholds['max_high']}",
        ),
        GateResult(
            name="security.critical",
            passed=critical <= thresholds["max_critical"],
            detail=f"critical={critical} threshold<={thresholds['max_critical']}",
        ),
    ]


def render_summary(results: list[GateResult]) -> str:
    rows = ["| Gate | Status | Detail |", "| --- | --- | --- |"]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        rows.append(f"| {r.name} | {status} | {r.detail} |")
    overall = "PASS" if all(r.passed for r in results) else "FAIL"
    return f"## Governance CI Gate — {overall}\n\n" + "\n".join(rows) + "\n"


def run_gate(
    metrics_path: Path,
    fairness_path: Optional[Path],
    security_path: Optional[Path],
    thresholds: dict,
) -> tuple[bool, list[GateResult]]:
    results: list[GateResult] = []

    metrics = _load_json(metrics_path)
    results.extend(evaluate_metrics_gate(metrics, thresholds["metrics"]))

    if fairness_path is not None:
        fairness = _load_json(fairness_path)
        results.extend(evaluate_fairness_gate(fairness, thresholds["fairness"]))

    if security_path is not None:
        security = _load_json(security_path)
        results.extend(evaluate_security_gate(security, thresholds["security"]))

    return all(r.passed for r in results), results


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="Evaluation metrics JSON")
    parser.add_argument("--fairness", type=Path, default=None, help="Fairness report JSON")
    parser.add_argument("--security", type=Path, default=None, help="Security scan JSON")
    parser.add_argument("--thresholds", type=Path, default=None, help="Override thresholds JSON")
    parser.add_argument("--summary-out", type=Path, default=None, help="Write Markdown summary")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _merge_thresholds(custom: Optional[dict]) -> dict:
    if not custom:
        return {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}
    merged: dict[str, Any] = {}
    for section, defaults in DEFAULT_THRESHOLDS.items():
        merged[section] = {**defaults, **(custom.get(section) or {})}
    return merged


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    thresholds = _merge_thresholds(
        json.loads(args.thresholds.read_text()) if args.thresholds else None
    )

    try:
        passed, results = run_gate(
            metrics_path=args.metrics,
            fairness_path=args.fairness,
            security_path=args.security,
            thresholds=thresholds,
        )
    except FileNotFoundError as exc:
        logger.error("input file missing: %s", exc)
        return EXIT_BAD_INPUT

    summary = render_summary(results)
    print(summary)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(summary)

    return EXIT_OK if passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
