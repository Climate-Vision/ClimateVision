"""
Dataset downloader for ClimateVision real-world model training.

Replaces the synthetic data path in run_training.py with real, openly-licensed
Earth-observation training datasets — one per analysis type:

    flooding      -> Sen1Floods11        (CC-BY 4.0, Sentinel-1 SAR + labels)
    deforestation -> MultiEarth Amazon   (Sentinel-2, binary deforestation masks)
    ice_melting   -> AI4Arctic Sea Ice   (Sentinel-1 SAR + expert ice charts)

Each dataset lands under data/datasets/<name>/ together with a provenance.json
file recording its source and license, so governance/datasheet.py can pick it up.

Usage:
    python scripts/download_datasets.py --dataset flooding
    python scripts/download_datasets.py --dataset all --out data/datasets

Most sources need an extra CLI tool (gsutil, eotdl, or the huggingface client).
The script checks for them and prints the exact install command if missing,
rather than failing silently. Nothing here bakes in credentials.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("download_datasets")

# ---------------------------------------------------------------------------
# Dataset registry: source location, fetch method, and license/provenance.
# ---------------------------------------------------------------------------
DATASETS: dict[str, dict] = {
    "flooding": {
        "name": "Sen1Floods11",
        "analysis_type": "flooding",
        "method": "gsutil",
        "uri": "gs://sen1floods11/",
        "license": "CC-BY 4.0",
        "homepage": "https://github.com/cloudtostreet/Sen1Floods11",
        "tool_hint": "pip install gsutil  (or install the Google Cloud SDK)",
        "notes": "4,831 512x512 chips, 11 flood events, 6 continents. "
                 "Pairs with src/climatevision/data/sar_preprocessing.py.",
    },
    "deforestation": {
        "name": "MultiEarth-2023-Amazon",
        "analysis_type": "deforestation",
        "method": "huggingface",
        "uri": "rdkulkarni/multiearth-deforestation",  # mirror; swap for the org-of-record copy
        "license": "see dataset card (research use)",
        "homepage": "https://arxiv.org/abs/2307.04715",
        "tool_hint": "pip install huggingface_hub",
        "notes": "Sentinel-2 256x256 tiles with binary deforestation masks (2018-2021). "
                 "Bands map to config.yaml deforestation: B04,B03,B02,B08.",
    },
    "ice_melting": {
        "name": "AI4Arctic-Sea-Ice-v2",
        "analysis_type": "ice_melting",
        "method": "eotdl",
        "uri": "ai4arctic-sea-ice-challenge-ready-to-train",
        "version": "2",
        "license": "CC-BY 4.0",
        "homepage": "https://www.eotdl.com/datasets/",
        "tool_hint": "pip install eotdl",
        "notes": "533 netCDF scenes, expert ice charts (DMI/CIS), 2018-2021.",
    },
}


def _have(tool: str) -> bool:
    return shutil.which(tool) is not None


def _run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _write_provenance(dest: Path, spec: dict) -> None:
    prov = {
        "dataset": spec["name"],
        "analysis_type": spec["analysis_type"],
        "source_uri": spec["uri"],
        "license": spec["license"],
        "homepage": spec["homepage"],
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "notes": spec.get("notes", ""),
    }
    (dest / "provenance.json").write_text(json.dumps(prov, indent=2))
    logger.info("Wrote provenance: %s", dest / "provenance.json")


def download(key: str, out_root: Path) -> bool:
    spec = DATASETS[key]
    dest = out_root / key
    dest.mkdir(parents=True, exist_ok=True)
    method = spec["method"]

    try:
        if method == "gsutil":
            if not _have("gsutil"):
                logger.error("Missing 'gsutil'. Install: %s", spec["tool_hint"])
                return False
            _run(["gsutil", "-m", "cp", "-r", spec["uri"], str(dest)])

        elif method == "eotdl":
            if not _have("eotdl"):
                logger.error("Missing 'eotdl'. Install: %s", spec["tool_hint"])
                return False
            _run(["eotdl", "datasets", "get", spec["uri"], "-v", spec.get("version", "1"),
                  "-p", str(dest)])

        elif method == "huggingface":
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                logger.error("Missing 'huggingface_hub'. Install: %s", spec["tool_hint"])
                return False
            snapshot_download(repo_id=spec["uri"], repo_type="dataset",
                              local_dir=str(dest))
        else:
            logger.error("Unknown fetch method: %s", method)
            return False

    except subprocess.CalledProcessError as exc:
        logger.error("Download failed for %s: %s", key, exc)
        return False

    _write_provenance(dest, spec)
    logger.info("Done: %s -> %s", spec["name"], dest)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Download ClimateVision training datasets.")
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], required=True)
    ap.add_argument("--out", default="data/datasets", help="output root directory")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]

    logger.info("Target datasets: %s", ", ".join(keys))
    results = {k: download(k, out_root) for k in keys}

    ok = sum(results.values())
    logger.info("Completed %d/%d datasets.", ok, len(results))
    for k, v in results.items():
        logger.info("  %-14s %s", k, "OK" if v else "FAILED (see hint above)")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
