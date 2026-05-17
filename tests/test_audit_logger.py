"""Tests for governance.audit_logger."""

from __future__ import annotations

import json

import numpy as np
import pytest

from climatevision.governance.audit_logger import (
    GENESIS_HASH,
    AuditLogger,
    log_prediction,
)


def _fake_inputs():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(4, 32, 32), dtype=np.uint8)
    output = rng.uniform(0, 1, size=(32, 32)).astype(np.float32)
    return image, output


def test_first_entry_chains_to_genesis(tmp_path):
    log = AuditLogger(log_path=tmp_path / "audit.jsonl")
    image, output = _fake_inputs()
    entry = log.log_prediction(
        model_version="unet-v0.1.0",
        input_data=image,
        output=output,
        request_id="r-1",
    )
    assert entry.prev_hash == GENESIS_HASH
    assert len(entry.entry_hash) == 64


def test_chain_links_correctly(tmp_path):
    log = AuditLogger(log_path=tmp_path / "audit.jsonl")
    image, output = _fake_inputs()

    e1 = log.log_prediction(model_version="v1", input_data=image, output=output)
    e2 = log.log_prediction(model_version="v1", input_data=image, output=output)
    e3 = log.log_prediction(model_version="v2", input_data=image, output=output)

    assert e2.prev_hash == e1.entry_hash
    assert e3.prev_hash == e2.entry_hash

    ok, failure = log.verify_chain()
    assert ok is True
    assert failure is None


def test_tampered_entry_breaks_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLogger(log_path=path)
    image, output = _fake_inputs()
    log.log_prediction(model_version="v1", input_data=image, output=output)
    log.log_prediction(model_version="v1", input_data=image, output=output)

    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["model_version"] = "tampered"
    lines[0] = json.dumps(record, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")

    fresh = AuditLogger(log_path=path)
    ok, failure = fresh.verify_chain()
    assert ok is False
    assert failure is not None


def test_resumes_chain_across_logger_instances(tmp_path):
    path = tmp_path / "audit.jsonl"
    image, output = _fake_inputs()

    AuditLogger(log_path=path).log_prediction(
        model_version="v1", input_data=image, output=output
    )
    new_logger = AuditLogger(log_path=path)
    e2 = new_logger.log_prediction(
        model_version="v1", input_data=image, output=output
    )

    entries = new_logger.iter_entries()
    assert len(entries) == 2
    assert e2.prev_hash == entries[0].entry_hash


def test_module_level_helper_writes_to_default_path(tmp_path, monkeypatch):
    target = tmp_path / "audit.jsonl"
    monkeypatch.setattr(
        "climatevision.governance.audit_logger._DEFAULT_AUDIT_LOG", target
    )
    image, output = _fake_inputs()
    entry = log_prediction(model_version="v1", input_data=image, output=output)
    assert target.exists()
    assert entry.model_version == "v1"


def test_dict_input_and_output_are_supported(tmp_path):
    log = AuditLogger(log_path=tmp_path / "audit.jsonl")
    entry = log.log_prediction(
        model_version="v1",
        input_data={"bbox": [-60, -15, -45, 5], "date": "2026-04-01"},
        output={"hectares": 1247.0, "carbon_tonnes": 4321.0},
    )
    assert entry.output_summary["hectares"] == pytest.approx(1247.0)
