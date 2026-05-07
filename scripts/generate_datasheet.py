#!/usr/bin/env python
"""
Generate a Datasheet for a ClimateVision training dataset.

Usage:
    python scripts/generate_datasheet.py \\
        --manifest data/manifests/sentinel2-deforestation.yaml \\
        --output-dir outputs/datasheets/

Runs inside the release CI pipeline so every dataset version published
ships with a Gebru-style datasheet alongside its model cards.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from climatevision.governance.datasheet import generate

logger = logging.getLogger("generate_datasheet")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Dataset manifest (yaml/json)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write the datasheet")
    parser.add_argument("--name", default=None, help="Override dataset name")
    parser.add_argument("--version", default=None, help="Override dataset version")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    paths = generate(
        manifest=args.manifest,
        output_dir=args.output_dir,
        name=args.name,
        version=args.version,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
