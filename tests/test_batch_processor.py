"""Tests for inference.batch_processor.

Imports the module directly via importlib to avoid the
``climatevision.inference`` package __init__ pulling in the rest of the
inference pipeline at test-collection time. Once the data package
__init__ is repaired we can drop the importlib shim.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_BATCH_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "climatevision"
    / "inference"
    / "batch_processor.py"
)
_spec = importlib.util.spec_from_file_location("cv_batch_processor", _BATCH_PATH)
batch = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["cv_batch_processor"] = batch
_spec.loader.exec_module(batch)

BatchProcessor = batch.BatchProcessor
BatchSummary = batch.BatchSummary


def _ok_inference(source, analysis_type):
    return {
        "hectares": 10.0,
        "carbon_tonnes": 35.0,
        "mean_confidence": 0.82,
        "mask": np.ones((4, 4), dtype=np.uint8),
    }


def _flaky_inference(state):
    def _fn(source, analysis_type):
        state["calls"] += 1
        if state["calls"] < 2:
            raise RuntimeError("transient")
        return {"hectares": 1.0, "carbon_tonnes": 3.0}
    return _fn


def _always_fail(source, analysis_type):
    raise ValueError(f"bad source: {source}")


def test_run_succeeds_for_all_jobs(tmp_path):
    proc = BatchProcessor(
        max_workers=2,
        manifest_path=tmp_path / "manifest.jsonl",
        inference_fn=_ok_inference,
    )
    jobs, summary = proc.run(["a.tif", "b.tif", "c.tif"])

    assert summary.total == 3
    assert summary.succeeded == 3
    assert summary.failed == 0
    assert all(j.status == "succeeded" for j in jobs)
    assert all(j.duration_ms is not None and j.duration_ms >= 0 for j in jobs)
    assert all(j.attempts == 1 for j in jobs)


def test_failed_jobs_are_isolated(tmp_path):
    proc = BatchProcessor(
        max_workers=2,
        manifest_path=tmp_path / "manifest.jsonl",
        inference_fn=_always_fail,
    )
    jobs, summary = proc.run(["a.tif", "b.tif"])

    assert summary.failed == 2
    assert summary.succeeded == 0
    assert all(j.status == "failed" and j.error.startswith("ValueError") for j in jobs)


def test_retry_succeeds_after_transient_failure(tmp_path):
    state = {"calls": 0}
    proc = BatchProcessor(
        max_workers=1,
        max_attempts=3,
        manifest_path=tmp_path / "manifest.jsonl",
        inference_fn=_flaky_inference(state),
    )
    jobs, summary = proc.run(["only.tif"])
    assert summary.succeeded == 1
    assert jobs[0].attempts == 2


def test_manifest_records_each_job(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    proc = BatchProcessor(
        max_workers=2,
        manifest_path=manifest,
        inference_fn=_ok_inference,
    )
    proc.run(["a.tif", "b.tif"])

    lines = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    statuses = {l["status"] for l in lines}
    assert statuses == {"succeeded"}
    for line in lines:
        assert line["result_summary"]["hectares"] == 10.0
        assert line["result_summary"]["positive_pixels"] == 16


def test_get_job_returns_record(tmp_path):
    proc = BatchProcessor(
        manifest_path=tmp_path / "manifest.jsonl",
        inference_fn=_ok_inference,
    )
    jobs, _ = proc.run(["a.tif"])
    fetched = proc.get_job(jobs[0].job_id)
    assert fetched is not None
    assert fetched.status == "succeeded"


def test_dict_source_roundtrips(tmp_path):
    captured = {}

    def fn(source, analysis_type):
        captured["source"] = source
        captured["analysis_type"] = analysis_type
        return {"hectares": 0.0, "carbon_tonnes": 0.0}

    proc = BatchProcessor(
        manifest_path=tmp_path / "manifest.jsonl",
        inference_fn=fn,
    )
    jobs, summary = proc.run([{"bbox": [0, 0, 1, 1], "date": "2026-01-01"}], analysis_type="flooding")
    assert summary.succeeded == 1
    assert captured["analysis_type"] == "flooding"
    assert captured["source"]["bbox"] == [0, 0, 1, 1]
