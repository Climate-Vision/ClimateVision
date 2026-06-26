#!/usr/bin/env python
"""
Download the Sen1Floods11 benchmark for flood-detection evaluation.

Sen1Floods11 (Bonafilia et al., 2020) provides Sentinel-1 (VV/VH, sigma0 in dB)
and Sentinel-2 chips with flood/water labels for 11 global flood events. The
446 *hand-labeled* chips are the gold standard; the ~4,300 weakly-labeled chips
(Otsu / permanent-water) are useful for training but NOT for reporting accuracy.

Layout produced under <dest>:
    <dest>/
      v1.1/data/flood_events/HandLabeled/S1Hand/*.tif      # 2 bands: VV, VH (dB)
      <dest>/v1.1/data/flood_events/HandLabeled/LabelHand/*.tif  # -1 nodata, 0 land, 1 water
      <dest>/v1.1/splits/flood_handlabeled/flood_{train,valid,test}_data.csv

Usage:
    # Hand-labeled only (small, ~a few GB) -- enough to compute test accuracy
    python scripts/download_sen1floods11.py --subset handlabeled --dest data/sen1floods11

    # Everything (hand + weak labels, tens of GB)
    python scripts/download_sen1floods11.py --subset all --dest data/sen1floods11

    # Show what would be copied without downloading
    python scripts/download_sen1floods11.py --subset handlabeled --dry-run

Requirements:
    Google Cloud SDK -- either `gcloud storage` (preferred) or `gsutil`.
    The bucket is public; no auth is required for read.

IMPORTANT: dataset hosting moves around (Radiant MLHub sunset; mirrors on
Source Cooperative / Hugging Face). VERIFY the --bucket value below is still the
live location before relying on it. Override with --bucket if it has moved.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Canonical public GCS bucket as of early 2026. VERIFY before use (see module docstring).
DEFAULT_BUCKET = "gs://sen1floods11-data/v1.1"

# Sub-paths within the bucket, relative to <bucket>/data/flood_events/
SUBSET_PATHS = {
    "handlabeled": [
        "HandLabeled/S1Hand",
        "HandLabeled/LabelHand",
    ],
    "weak": [
        "WeaklyLabeled/S1Weak",
        "WeaklyLabeled/S2IndexLabelWeak",
    ],
}
# Splits CSVs live alongside the data directory.
SPLITS_SUBPATH = "splits/flood_handlabeled"


def _find_gcs_tool() -> list[str]:
    """Return the argv prefix for a working GCS copy tool, or exit with guidance."""
    if shutil.which("gcloud"):
        return ["gcloud", "storage", "rsync", "-r"]
    if shutil.which("gsutil"):
        return ["gsutil", "-m", "rsync", "-r"]
    sys.exit(
        "ERROR: neither `gcloud` nor `gsutil` found on PATH.\n"
        "Install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install\n"
        "The Sen1Floods11 bucket is public, so no login is needed after install."
    )


def _rsync(tool: list[str], src: str, dst: Path, dry_run: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    cmd = list(tool)
    if dry_run:
        # Both gcloud storage rsync and gsutil rsync support -n for dry-run.
        cmd.append("-n")
    cmd += [src, str(dst)]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(
            f"ERROR: copy failed for {src} (exit {result.returncode}).\n"
            "If this is a 'bucket not found' / 404 error, the dataset has likely "
            "moved -- re-run with --bucket pointing at the current mirror."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--subset",
        choices=["handlabeled", "weak", "all"],
        default="handlabeled",
        help="Which split to fetch. 'handlabeled' is enough to report accuracy (default).",
    )
    parser.add_argument("--dest", type=Path, default=Path("data/sen1floods11"), help="Local destination root.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="GCS bucket root (verify it is current).")
    parser.add_argument("--dry-run", action="store_true", help="List what would be copied without downloading.")
    args = parser.parse_args()

    tool = _find_gcs_tool()
    data_root = f"{args.bucket}/data/flood_events"

    subsets = ["handlabeled", "weak"] if args.subset == "all" else [args.subset]

    print(f"Sen1Floods11 download")
    print(f"  bucket : {args.bucket}")
    print(f"  dest   : {args.dest.resolve()}")
    print(f"  subset : {args.subset}")
    print(f"  tool   : {tool[0]}")
    print()

    for subset in subsets:
        for rel in SUBSET_PATHS[subset]:
            src = f"{data_root}/{rel}"
            dst = args.dest / "v1.1" / "data" / "flood_events" / rel
            print(f"[{subset}] {rel}")
            _rsync(tool, src, dst, args.dry_run)

    # Always grab the split CSVs (tiny) so eval can use the official test split.
    splits_src = f"{args.bucket}/{SPLITS_SUBPATH}"
    splits_dst = args.dest / "v1.1" / SPLITS_SUBPATH
    print(f"[splits] {SPLITS_SUBPATH}")
    _rsync(tool, splits_src, splits_dst, args.dry_run)

    print()
    if args.dry_run:
        print("Dry run complete -- nothing was written.")
    else:
        print("Download complete. Evaluate the ensemble with:")
        print(
            f"  python scripts/eval_flood.py "
            f"--data-root {args.dest} --split test"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
