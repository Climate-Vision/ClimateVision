"""
Immutable audit trail for ClimateVision model versions and predictions.

Every prediction logged by this module produces a chained record that
includes:

- A SHA-256 hash of the input payload (image + parameters).
- The model version that produced the result.
- A summary of the output (positive fraction, mean confidence, threshold).
- A `prev_hash` linking the entry to the previous one, forming an
  append-only hash chain. Tampering with any historical record breaks
  the chain and is detected by `verify_chain()`.

The chain is persisted as JSON Lines so that downstream tooling
(MLflow, BigQuery, regulators) can ingest it without parsing custom
formats.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_AUDIT_LOG = _PROJECT_ROOT / "outputs" / "audit" / "predictions.jsonl"

GENESIS_HASH = "0" * 64


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_signature(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


@dataclass
class AuditEntry:
    timestamp: str
    model_version: str
    input_hash: str
    output_summary: dict
    request_id: Optional[str]
    user_id: Optional[str]
    prev_hash: str
    entry_hash: str = ""
    metadata: dict = field(default_factory=dict)

    def compute_hash(self) -> str:
        body = {k: v for k, v in asdict(self).items() if k != "entry_hash"}
        return _stable_hash(body)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class AuditLogger:
    """
    Append-only audit logger backed by a hash-chained JSONL file.

    The logger is process-safe via an in-memory lock; for cross-process
    safety wrap calls in your own filelock or write through a queue.
    """

    def __init__(self, log_path: Optional[Union[str, Path]] = None) -> None:
        self.log_path = Path(log_path) if log_path else _DEFAULT_AUDIT_LOG
        self._lock = threading.Lock()
        self._last_hash: Optional[str] = None

    def _read_last_hash(self) -> str:
        if not self.log_path.exists():
            return GENESIS_HASH
        last = GENESIS_HASH
        with self.log_path.open() as fh:
            for line in fh:
                if line.strip():
                    last = json.loads(line)["entry_hash"]
        return last

    def log_prediction(
        self,
        model_version: str,
        input_data: Union[np.ndarray, dict],
        output: Union[np.ndarray, dict],
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        threshold: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> AuditEntry:
        if isinstance(input_data, np.ndarray):
            input_payload = _array_signature(input_data)
        else:
            input_payload = dict(input_data)

        if isinstance(output, np.ndarray):
            output_payload = {
                **_array_signature(output),
                "mean_confidence": float(output.mean()),
                "positive_fraction": float((output > threshold).mean()),
                "threshold": threshold,
            }
        else:
            output_payload = dict(output)

        with self._lock:
            if self._last_hash is None:
                self._last_hash = self._read_last_hash()

            entry = AuditEntry(
                timestamp=_utcnow(),
                model_version=model_version,
                input_hash=_stable_hash(input_payload),
                output_summary=output_payload,
                request_id=request_id,
                user_id=user_id,
                prev_hash=self._last_hash,
                metadata=metadata or {},
            )
            entry.entry_hash = entry.compute_hash()

            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as fh:
                fh.write(entry.to_json() + "\n")

            self._last_hash = entry.entry_hash
            logger.info(
                "Logged audit entry %s for model %s",
                entry.entry_hash[:12],
                model_version,
            )
            return entry

    def iter_entries(self) -> list[AuditEntry]:
        if not self.log_path.exists():
            return []
        entries: list[AuditEntry] = []
        with self.log_path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                entries.append(AuditEntry(**json.loads(line)))
        return entries

    def verify_chain(self) -> tuple[bool, Optional[str]]:
        """
        Walk the chain from genesis and confirm each entry hashes correctly
        and references the previous entry.

        Returns:
            (ok, failure_hash) — failure_hash is the entry where the chain
            breaks, or None when the chain is valid.
        """
        prev = GENESIS_HASH
        for entry in self.iter_entries():
            if entry.prev_hash != prev:
                return False, entry.entry_hash
            recomputed = entry.compute_hash()
            if recomputed != entry.entry_hash:
                return False, entry.entry_hash
            prev = entry.entry_hash
        return True, None


def log_prediction(
    model_version: str,
    input_data: Union[np.ndarray, dict],
    output: Union[np.ndarray, dict],
    **kwargs: Any,
) -> AuditEntry:
    """Module-level convenience wrapper using the default audit log path."""
    return AuditLogger().log_prediction(
        model_version=model_version,
        input_data=input_data,
        output=output,
        **kwargs,
    )
