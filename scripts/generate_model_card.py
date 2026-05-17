#!/usr/bin/env python
"""
Generate a Model Card for a ClimateVision release.

Usage:
    python scripts/generate_model_card.py \\
        --config config.yaml \\
        --metrics outputs/eval/metrics.json \\
        --fairness outputs/governance/fairness.json \\
        --output-dir outputs/model_cards/

The script is intended to run inside the release CI pipeline so that
every model version published has a card committed alongside it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from climatevision.governance.model_card import generate

logger = logging.getLogger("generate_model_card")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Training config (yaml/json)")
    parser.add_argument("--metrics", type=Path, required=True, help="Evaluation metrics JSON")
    parser.add_argument("--fairness", type=Path, default=None, help="Fairness report JSON")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write the card")
    parser.add_argument("--name", default=None, help="Override model name")
    parser.add_argument("--version", default=None, help="Override model version")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    paths = generate(
        config=args.config,
        metrics=args.metrics,
        fairness_report=args.fairness,
        output_dir=args.output_dir,
        name=args.name,
        version=args.version,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
