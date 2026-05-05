"""
Batch processor for ClimateVision inference jobs.

Submits a list of image paths (or numpy arrays) to the inference
pipeline in parallel, tracks per-job state, and produces a structured
result manifest. The processor is designed to be driven from either
a CLI script or the FastAPI background-task layer.

Job state machine:

    queued -> running -> (succeeded | failed)

Each job is appended to a JSONL manifest as soon as its terminal
state is reached so a long-running batch can be resumed or audited
without waiting for the whole queue to finish.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MANIFEST = _PROJECT_ROOT / "outputs" / "batches" / "manifest.jsonl"

JobInput = Union[str, Path, dict]
InferenceFn = Callable[[JobInput, str], dict]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BatchJob:
    job_id: str
    source: str
    analysis_type: str
    status: str = "queued"
    submitted_at: str = field(default_factory=_utcnow)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: Optional[int] = None
    result_summary: Optional[dict] = None
    error: Optional[str] = None
    attempts: int = 0


@dataclass
class BatchSummary:
    total: int
    succeeded: int
    failed: int
    duration_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def _default_inference_fn(source: JobInput, analysis_type: str) -> dict:
    """
    Default inference adapter — calls run_inference_from_file or run_inference
    depending on the input shape. Imported lazily so unit tests can stub it.
    """
    from climatevision.inference.pipeline import (
        run_inference,
        run_inference_from_file,
    )

    if isinstance(source, (str, Path)):
        return run_inference_from_file(str(source), analysis_type=analysis_type)
    if isinstance(source, dict):
        return run_inference(**source, analysis_type=analysis_type)
    raise TypeError(f"Unsupported source type: {type(source).__name__}")


class BatchProcessor:
    """
    Parallel batch executor for inference jobs.

    Args:
        max_workers: Thread pool size. Defaults to 4.
        max_attempts: Retry count for transient failures.
        manifest_path: Where to append per-job records. Created on first write.
        inference_fn: Override the actual inference call (handy for tests
            and for swapping in batch_predict implementations later).
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_attempts: int = 1,
        manifest_path: Optional[Union[str, Path]] = None,
        inference_fn: Optional[InferenceFn] = None,
    ) -> None:
        self.max_workers = max_workers
        self.max_attempts = max(1, max_attempts)
        self.manifest_path = Path(manifest_path) if manifest_path else _DEFAULT_MANIFEST
        self._inference_fn = inference_fn or _default_inference_fn
        self._jobs: dict[str, BatchJob] = {}
        self._lock = threading.Lock()

    def _persist(self, job: BatchJob) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.manifest_path.open("a") as fh:
            fh.write(json.dumps(asdict(job)) + "\n")

    def _summarize_result(self, result: Any) -> dict:
        if isinstance(result, dict):
            keep = {}
            for key in ("hectares", "carbon_tonnes", "iou", "f1", "mean_confidence"):
                if key in result:
                    keep[key] = result[key]
            if "mask" in result:
                import numpy as np

                arr = np.asarray(result["mask"])
                keep["positive_pixels"] = int(arr.sum())
                keep["total_pixels"] = int(arr.size)
            return keep
        return {"raw": str(result)[:200]}

    def _run_one(self, job: BatchJob, source: JobInput) -> BatchJob:
        for attempt in range(1, self.max_attempts + 1):
            job.attempts = attempt
            job.status = "running"
            job.started_at = _utcnow()
            t0 = time.perf_counter()
            try:
                result = self._inference_fn(source, job.analysis_type)
                job.result_summary = self._summarize_result(result)
                job.status = "succeeded"
                job.error = None
                break
            except Exception as exc:  # noqa: BLE001 - we want to capture all
                logger.exception("Job %s attempt %d failed", job.job_id, attempt)
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "failed"
            finally:
                job.duration_ms = int((time.perf_counter() - t0) * 1000)
                job.finished_at = _utcnow()
        self._persist(job)
        return job

    def submit_batch(
        self,
        sources: Iterable[JobInput],
        analysis_type: str = "deforestation",
    ) -> list[BatchJob]:
        sources = list(sources)
        jobs = [
            BatchJob(
                job_id=str(uuid.uuid4()),
                source=str(s) if isinstance(s, (str, Path)) else json.dumps(s, default=str),
                analysis_type=analysis_type,
            )
            for s in sources
        ]
        for j in jobs:
            self._jobs[j.job_id] = j
        return jobs

    def run(
        self,
        sources: Iterable[JobInput],
        analysis_type: str = "deforestation",
    ) -> tuple[list[BatchJob], BatchSummary]:
        sources = list(sources)
        jobs = self.submit_batch(sources, analysis_type=analysis_type)
        t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._run_one, job, source): job
                for job, source in zip(jobs, sources)
            }
            for fut in as_completed(futures):
                fut.result()

        duration = time.perf_counter() - t0
        succeeded = sum(1 for j in jobs if j.status == "succeeded")
        failed = sum(1 for j in jobs if j.status == "failed")
        summary = BatchSummary(
            total=len(jobs),
            succeeded=succeeded,
            failed=failed,
            duration_seconds=round(duration, 3),
        )
        logger.info(
            "Batch finished: total=%d succeeded=%d failed=%d in %.2fs",
            summary.total,
            summary.succeeded,
            summary.failed,
            duration,
        )
        return jobs, summary

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        return self._jobs.get(job_id)
