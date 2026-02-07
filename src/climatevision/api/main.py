from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from climatevision.db import get_connection, init_db


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PredictRequest(BaseModel):
    kind: str = Field(default="demo")
    bbox: Optional[list[float]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class RunRow(BaseModel):
    id: int
    kind: str
    status: str
    bbox: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    created_at: str
    updated_at: str


class ResultRow(BaseModel):
    id: int
    run_id: int
    payload: dict[str, Any]
    mask_path: Optional[str] = None
    created_at: str


def _load_template_result(*, bbox: Optional[list[float]], start_date: Optional[str], end_date: Optional[str]) -> dict[str, Any]:
    outputs_dir = Path(__file__).resolve().parents[3] / "outputs"
    template_path = outputs_dir / "inference_results.json"
    if template_path.exists():
        template: dict[str, Any] = json.loads(template_path.read_text(encoding="utf-8"))
    else:
        template = {
            "region": {"bbox": bbox or None},
            "ndvi_stats": {"NDVI_min": 0.0, "NDVI_mean": 0.0, "NDVI_max": 0.0},
            "inference": {
                "image_size": [256, 256],
                "forest_pixels": 0,
                "non_forest_pixels": 0,
                "forest_percentage": 0.0,
                "mean_confidence": 0.0,
            },
        }

    if bbox is not None:
        template.setdefault("region", {})["bbox"] = bbox
    if start_date and end_date:
        template.setdefault("region", {})["date_range"] = f"{start_date} to {end_date}"

    return template


async def _persist_upload(*, run_id: int, file: UploadFile) -> str:
    outputs_dir = Path(__file__).resolve().parents[3] / "outputs"
    uploads_dir = outputs_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / f"run_{run_id}_{file.filename}"
    dest.write_bytes(await file.read())
    return str(dest)


def create_app() -> FastAPI:
    init_db()

    app = FastAPI(title="ClimateVision API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runs")
    def list_runs(limit: int = 50) -> list[RunRow]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [RunRow(**dict(r)) for r in rows]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: int) -> dict[str, Any]:
        with get_connection() as conn:
            run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
            result = conn.execute(
                "SELECT * FROM results WHERE run_id = ? ORDER BY id DESC LIMIT 1", (run_id,)
            ).fetchone()

        payload: Optional[dict[str, Any]] = None
        mask_path: Optional[str] = None
        if result is not None:
            payload = json.loads(result["payload_json"])
            mask_path = result["mask_path"]

        return {
            "run": dict(run),
            "result": None
            if result is None
            else {
                "id": result["id"],
                "run_id": result["run_id"],
                "payload": payload,
                "mask_path": mask_path,
                "created_at": result["created_at"],
            },
        }

    @app.post("/api/predict")
    async def predict_json(body: PredictRequest) -> dict[str, Any]:
        """JSON endpoint (bbox/date-range, no file upload)."""

        created_at = _utc_now_iso()
        bbox_json = json.dumps(body.bbox) if body.bbox else None

        with get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (kind, status, bbox, start_date, end_date, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    body.kind,
                    "completed",
                    bbox_json,
                    body.start_date,
                    body.end_date,
                    created_at,
                    created_at,
                ),
            )
            run_id = int(cur.lastrowid)

        template = _load_template_result(bbox=body.bbox, start_date=body.start_date, end_date=body.end_date)
        result_created_at = _utc_now_iso()

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO results (run_id, payload_json, mask_path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, json.dumps(template), None, result_created_at),
            )

        return {"run_id": run_id, "result": template}

    @app.post("/api/predict/upload")
    async def predict_upload(
        kind: str = Form(default="upload"),
        bbox: str | None = Form(default=None),
        start_date: str | None = Form(default=None),
        end_date: str | None = Form(default=None),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        """Multipart endpoint for file upload. `bbox` is expected to be JSON (e.g. "[-62, -3.1, -61.8, -2.9]")."""

        created_at = _utc_now_iso()

        parsed_bbox: Optional[list[float]] = None
        if bbox:
            try:
                parsed_bbox = json.loads(bbox)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail="Invalid bbox JSON") from e

        with get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (kind, status, bbox, start_date, end_date, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    "completed",
                    json.dumps(parsed_bbox) if parsed_bbox else None,
                    start_date,
                    end_date,
                    created_at,
                    created_at,
                ),
            )
            run_id = int(cur.lastrowid)

        template = _load_template_result(bbox=parsed_bbox, start_date=start_date, end_date=end_date)
        template.setdefault("input", {})["file"] = await _persist_upload(run_id=run_id, file=file)

        result_created_at = _utc_now_iso()
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO results (run_id, payload_json, mask_path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, json.dumps(template), None, result_created_at),
            )

        return {"run_id": run_id, "result": template}

    frontend_dir = Path(__file__).resolve().parents[3] / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


app = create_app()
