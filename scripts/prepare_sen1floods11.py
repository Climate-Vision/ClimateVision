"""
Convert the Sen1Floods11 hand-labeled data into the ClimateVision training layout.

Input (downloaded from gs://sen1floods11/v1.1/data/flood_events/HandLabeled/):
    S2Hand/    EVENT_CHIPID_S2Hand.tif     13-band Sentinel-2 L1C (TOA * 10000)
    LabelHand/ EVENT_CHIPID_LabelHand.tif  1-band labels: -1 no-data, 0 not-water, 1 water
    (optional) JRCWaterHand/ EVENT_CHIPID_JRCWaterHand.tif  permanent-water reference

Output (what scripts/train_real.py + data/dataset.py expect):
    <out>/train|val|test/
        images/ EVENT_CHIPID.tif   3-band float32  (config bands: B03,B08,B11)
        masks/  EVENT_CHIPID.tif   1-band uint8 class indices

Class mapping (matches config.yaml flooding: dry_land=0, permanent_water=1, flooded=2):
    - binary mode (default): not-water/no-data -> 0 (dry), water -> 1.
      Class 2 is simply unused; the 3-class model still trains and serves correctly.
    - 3-class mode (--jrc-dir given): water AND permanent -> 1, water AND NOT permanent -> 2.

Splits: uses the official CSVs in splits/ when present (so results are comparable to
published numbers); otherwise falls back to a deterministic 80/10/10 split.

Usage:
    python scripts/prepare_sen1floods11.py \
        --s2-dir    data/sen1floods11/S2Hand \
        --label-dir data/sen1floods11/LabelHand \
        --splits-dir data/sen1floods11/splits \
        --out-dir   data/datasets/flooding
    # optional 3-class: add  --jrc-dir data/sen1floods11/JRCWaterHand
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("prepare_sen1floods11")

# Config bands for flooding (0-indexed into the 13-band S2 stack):
#   B03 Green = 2, B08 NIR = 7, B11 SWIR-1 = 11
DEFAULT_BANDS = (2, 7, 11)
STEM_RE = re.compile(r"([A-Za-z\-]+_\d+)")   # e.g. "Bolivia_103757", "Sri-Lanka_31559"


def chip_stem(name: str) -> str | None:
    """Extract the EVENT_CHIPID stem from a filename or CSV token."""
    m = STEM_RE.search(name)
    return m.group(1) if m else None


def assign_splits(stems: list[str], splits_dir: Path | None) -> dict[str, str]:
    """Return {stem: split}. Use official CSVs if found, else deterministic 80/10/10."""
    mapping: dict[str, str] = {}
    if splits_dir and splits_dir.exists():
        wanted = {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}
        for split, keys in wanted.items():
            for csv in splits_dir.rglob("*.csv"):
                if any(k in csv.name.lower() for k in keys):
                    for line in csv.read_text().splitlines():
                        s = chip_stem(line)
                        if s:
                            mapping[s] = split
    if mapping:
        logger.info("Loaded splits from CSVs: %d chips assigned.", len(mapping))
        return mapping

    # Deterministic fallback split
    logger.warning("No usable split CSVs found — using deterministic 80/10/10 split.")
    for i, s in enumerate(sorted(stems)):
        r = i % 10
        mapping[s] = "train" if r < 8 else ("val" if r == 8 else "test")
    return mapping


def build_mask(label: np.ndarray, jrc: np.ndarray | None) -> np.ndarray:
    """Map raw label (-1/0/1) [+ optional JRC permanent water] to class indices 0/1/2."""
    out = np.zeros(label.shape, dtype=np.uint8)          # default dry (0); -1 no-data -> 0
    water = label == 1
    if jrc is None:
        out[water] = 1                                    # binary: all water -> class 1
    else:
        permanent = jrc == 1
        out[water & permanent] = 1                        # permanent_water
        out[water & ~permanent] = 2                       # flooded
    return out


def convert_one(stem: str, s2_path: Path, label_path: Path, jrc_path: Path | None,
                bands: tuple[int, ...], out_dir: Path) -> bool:
    try:
        with rasterio.open(s2_path) as src:
            img = src.read().astype(np.float32)           # (13, H, W)
            profile = src.profile
        sel = img[list(bands), :, :]                      # (3, H, W)

        with rasterio.open(label_path) as src:
            label = src.read(1)
        jrc = None
        if jrc_path and jrc_path.exists():
            with rasterio.open(jrc_path) as src:
                jrc = src.read(1)
        mask = build_mask(label, jrc)
    except Exception as exc:
        logger.warning("Skip %s (%s)", stem, exc)
        return False

    img_out = out_dir / "images" / f"{stem}.tif"
    msk_out = out_dir / "masks" / f"{stem}.tif"
    img_out.parent.mkdir(parents=True, exist_ok=True)
    msk_out.parent.mkdir(parents=True, exist_ok=True)

    prof_img = profile.copy()
    prof_img.update(count=len(bands), dtype="float32")
    with rasterio.open(img_out, "w", **prof_img) as dst:
        dst.write(sel)

    prof_msk = profile.copy()
    prof_msk.update(count=1, dtype="uint8", nodata=None)
    with rasterio.open(msk_out, "w", **prof_msk) as dst:
        dst.write(mask[np.newaxis, :, :])
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert Sen1Floods11 to ClimateVision layout.")
    ap.add_argument("--s2-dir", required=True)
    ap.add_argument("--label-dir", required=True)
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--jrc-dir", default=None, help="optional, enables 3-class output")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bands", default="2,7,11", help="0-indexed S2 bands (B03,B08,B11)")
    args = ap.parse_args()

    s2_dir, label_dir = Path(args.s2_dir), Path(args.label_dir)
    jrc_dir = Path(args.jrc_dir) if args.jrc_dir else None
    out_root = Path(args.out_dir)
    bands = tuple(int(b) for b in args.bands.split(","))

    s2_files = {chip_stem(p.name): p for p in s2_dir.glob("*.tif") if chip_stem(p.name)}
    label_files = {chip_stem(p.name): p for p in label_dir.glob("*.tif") if chip_stem(p.name)}
    stems = sorted(set(s2_files) & set(label_files))
    if not stems:
        logger.error("No matching S2/Label pairs found. Check --s2-dir and --label-dir.")
        return 2
    logger.info("Found %d image/label pairs.", len(stems))

    splits = assign_splits(stems, Path(args.splits_dir) if args.splits_dir else None)

    counts = {"train": 0, "val": 0, "test": 0}
    for stem in stems:
        split = splits.get(stem, "train")
        jrc_path = (jrc_dir / f"{stem}_JRCWaterHand.tif") if jrc_dir else None
        if convert_one(stem, s2_files[stem], label_files[stem], jrc_path, bands,
                       out_root / split):
            counts[split] += 1

    logger.info("Done. train=%(train)d  val=%(val)d  test=%(test)d", counts)
    logger.info("Output: %s/{train,val,test}/{images,masks}", out_root)
    logger.info("Next: python scripts/train_real.py --analysis-type flooding "
                "--data-dir %s --epochs 50 --batch-size 8 --out models", out_root)
    return 0 if sum(counts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
