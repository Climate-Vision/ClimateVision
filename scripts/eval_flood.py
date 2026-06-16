#!/usr/bin/env python
"""
Evaluate the SAR flood-detection ensemble against labeled ground truth.

Runs EnsembleFloodPipeline (or a single detector) over Sen1Floods11 chips and
reports surface-water detection metrics against the hand labels. Produces REAL
numbers -- it computes nothing unless pointed at real labeled data.

What it measures
----------------
Sen1Floods11 hand labels mark *surface water* (permanent + flood water together)
as class 1, dry land as 0, and no-data/cloud as -1. The SAR ensemble detects
open water from backscatter, so this evaluates **water detection**, which is the
core skill behind flood mapping. It is NOT a pure "flood-only" metric, because
the benchmark does not separate flood water from permanent water in its masks.
Report it as such.

Caveat on the change-detection branch: Sen1Floods11 hand-labeled chips have no
pre-event image, so the LIST (pre/post differencing) detector cannot run here.
With --detector ensemble and no pre-event, the majority vote reduces to "DLR AND
TUW must agree". Use Kuro Siwo (which ships pre/post pairs) to exercise LIST.

Metrics (for the water/positive class): precision, recall, F1, IoU (Jaccard),
overall pixel accuracy, plus the raw confusion matrix. Pixels labeled -1 are
ignored.

Usage:
    python scripts/eval_flood.py --data-root data/sen1floods11 --split test
    python scripts/eval_flood.py --data-root data/sen1floods11 --detector tuw
    python scripts/eval_flood.py --data-root data/sen1floods11 --limit 20 --json out.json

Requirements: rasterio (GeoTIFF I/O), numpy, scikit-image (for the DLR detector).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make `climatevision` importable when run from a source checkout.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climatevision.analysis.flooding_ensemble import (  # noqa: E402
    EnsembleFloodPipeline,
    DLRFloodDetector,
    TUWFloodDetector,
)

# Sen1Floods11 label encoding.
LABEL_NODATA = -1
LABEL_LAND = 0
LABEL_WATER = 1


def find_pairs(data_root: Path, split: str | None, limit: int | None) -> list[tuple[Path, Path]]:
    """Pair each LabelHand tif with its matching S1Hand tif.

    Pairs by the shared `<Region>_<id>` prefix so this is robust to minor
    differences in the split-CSV format across dataset versions. If a split
    CSV is present it is used to filter to that split; otherwise all
    hand-labeled chips are evaluated.
    """
    hand = data_root / "v1.1" / "data" / "flood_events" / "HandLabeled"
    s1_dir = hand / "S1Hand"
    label_dir = hand / "LabelHand"
    if not label_dir.is_dir() or not s1_dir.is_dir():
        sys.exit(
            f"ERROR: expected Sen1Floods11 HandLabeled dirs under {hand}.\n"
            "Download first: python scripts/download_sen1floods11.py --dest "
            f"{data_root}"
        )

    # Index S1 chips by their Region_id prefix.
    s1_by_key = {}
    for p in s1_dir.glob("*.tif"):
        key = p.name.replace("_S1Hand.tif", "")
        s1_by_key[key] = p

    allow_keys = _load_split_keys(data_root, split)

    pairs: list[tuple[Path, Path]] = []
    for label_path in sorted(label_dir.glob("*_LabelHand.tif")):
        key = label_path.name.replace("_LabelHand.tif", "")
        if allow_keys is not None and key not in allow_keys:
            continue
        s1_path = s1_by_key.get(key)
        if s1_path is None:
            print(f"  warning: no S1 chip for label {label_path.name}, skipping")
            continue
        pairs.append((s1_path, label_path))

    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def _load_split_keys(data_root: Path, split: str | None) -> set[str] | None:
    """Return the set of `Region_id` keys for the requested split, or None."""
    if not split:
        return None
    csv = data_root / "v1.1" / "splits" / "flood_handlabeled" / f"flood_{split}_data.csv"
    if not csv.is_file():
        print(f"  note: split file {csv.name} not found; evaluating ALL hand-labeled chips")
        return None
    keys: set[str] = set()
    for line in csv.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # Rows reference filenames like 'Bolivia_103757_S1Hand.tif'; extract the prefix.
        for field in line.replace(",", " ").split():
            name = Path(field).name
            for suffix in ("_S1Hand.tif", "_LabelHand.tif", "_S2Hand.tif"):
                if name.endswith(suffix):
                    keys.add(name.replace(suffix, ""))
    return keys or None


def _read_band(path: Path, band: int) -> np.ndarray:
    try:
        import rasterio
    except ImportError:
        sys.exit("ERROR: rasterio is required to read GeoTIFFs. Install: pip install rasterio")
    with rasterio.open(path) as ds:
        return ds.read(band).astype(np.float32)


def _predict_water(detector: str, post_vh: np.ndarray) -> np.ndarray:
    """Return a binary water/flood mask (1=water) for the chosen detector."""
    if detector == "ensemble":
        # No pre-event image in Sen1Floods11 hand labels -> LIST branch is skipped.
        out = EnsembleFloodPipeline().detect(post_vh=post_vh, pre_vh=None)
        return out["ensemble_mask"]
    if detector == "dlr":
        return DLRFloodDetector().detect(post_vh)
    if detector == "tuw":
        return TUWFloodDetector().detect(post_vh)
    raise ValueError(f"unknown detector {detector!r}")


def evaluate(pairs, detector: str, vh_band: int) -> dict:
    """Accumulate a confusion matrix over all chips and derive metrics."""
    tp = fp = fn = tn = 0
    per_scene_iou: list[float] = []

    for i, (s1_path, label_path) in enumerate(pairs, 1):
        vh = _read_band(s1_path, vh_band)
        label = _read_band(label_path, 1).astype(np.int32)

        valid = label != LABEL_NODATA
        if not valid.any():
            continue

        pred = _predict_water(detector, vh).astype(bool)
        gt = label == LABEL_WATER

        p = pred[valid]
        g = gt[valid]

        s_tp = int(np.sum(p & g))
        s_fp = int(np.sum(p & ~g))
        s_fn = int(np.sum(~p & g))
        s_tn = int(np.sum(~p & ~g))
        tp += s_tp
        fp += s_fp
        fn += s_fn
        tn += s_tn

        denom = s_tp + s_fp + s_fn
        per_scene_iou.append(s_tp / denom if denom else 1.0)

        if i % 25 == 0 or i == len(pairs):
            print(f"  processed {i}/{len(pairs)} chips")

    return _metrics(tp, fp, fn, tn, per_scene_iou)


def _metrics(tp, fp, fn, tn, per_scene_iou) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
        "pixel_accuracy": round(accuracy, 4),
        "mean_iou_per_scene": round(float(np.mean(per_scene_iou)), 4) if per_scene_iou else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "n_pixels_evaluated": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/sen1floods11"), help="Sen1Floods11 root.")
    parser.add_argument("--split", default="test", help="Split to evaluate: test/valid/train, or '' for all chips.")
    parser.add_argument(
        "--detector",
        choices=["ensemble", "dlr", "tuw"],
        default="ensemble",
        help="Which detector to evaluate (default: ensemble).",
    )
    parser.add_argument("--vh-band", type=int, default=2, help="1-based band index of VH in S1 chips (VV=1, VH=2).")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N chips (quick check).")
    parser.add_argument("--json", type=Path, default=None, help="Optional path to write the metrics as JSON.")
    args = parser.parse_args()

    pairs = find_pairs(args.data_root, args.split or None, args.limit)
    if not pairs:
        sys.exit("ERROR: no (S1, label) chip pairs found. Check --data-root and that the download completed.")

    print(f"Evaluating detector='{args.detector}' on {len(pairs)} chips (split={args.split or 'all'})")
    metrics = evaluate(pairs, args.detector, args.vh_band)

    print("\n=== Sen1Floods11 surface-water detection metrics ===")
    print(f"  detector        : {args.detector}")
    print(f"  chips           : {len(pairs)}")
    print(f"  precision       : {metrics['precision']}")
    print(f"  recall          : {metrics['recall']}")
    print(f"  F1              : {metrics['f1']}")
    print(f"  IoU (water)     : {metrics['iou']}")
    print(f"  mean IoU/scene  : {metrics['mean_iou_per_scene']}")
    print(f"  pixel accuracy  : {metrics['pixel_accuracy']}")
    print(f"  confusion (px)  : {metrics['confusion_matrix']}")
    print("\nNote: measures surface-water detection (permanent + flood water), per the")
    print("Sen1Floods11 label definition -- not flood-only. See script docstring.")

    if args.json:
        payload = {"detector": args.detector, "split": args.split, "n_chips": len(pairs), **metrics}
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote metrics to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
