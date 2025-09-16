# server_jobs_router.py
from __future__ import annotations

import os
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
EVENTS_MAX = int(os.getenv("JOBS_EVENTS_MAX", "500"))
AUTOCREATE_ON_CHECKPOINT = os.getenv("JOBS_AUTOCREATE", "0") in {"1", "true", "True"}

jobs_router = APIRouter(prefix="/jobs", tags=["jobs"])

# In-memory store
_JOBS: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_job(goal: str, autostart: bool = True, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    job_id = uuid4().hex[:8]
    job = {
        "job_id": job_id,
        "goal": goal,
        "status": "running" if autostart else "created",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "steps_total": 0,
        "steps_done": 0,
        "last_event": "",
        "meta": meta or {},
        "events": deque(maxlen=EVENTS_MAX),  # type: Deque[Dict[str, Any]]
    }
    # primer evento meta
    job["events"].append({"type": "job_meta", "status": job["status"], "meta": "", "ts": _now_iso()})
    _JOBS[job_id] = job
    return job


def _get_job(job_id: str) -> Dict[str, Any]:
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    return job


def _append_event(job: Dict[str, Any], payload: Dict[str, Any]) -> None:
    evt = dict(payload)
    evt["ts"] = _now_iso()
    job["events"].append(evt)
    job["updated_at"] = evt["ts"]


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class JobStartReq(BaseModel):
    goal: str = Field(..., min_length=1)
    autostart: bool = True
    meta: Dict[str, Any] = Field(default_factory=dict)


class CheckpointReq(BaseModel):
    job_id: str = Field(..., min_length=1)
    type: str = Field(..., description="plan | step_start | step_end | status | meta")
    # datos opcionales
    step_id: Optional[str] = None
    steps: Optional[List[Dict[str, Any]]] = None
    status: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    analysis: Optional[Any] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@jobs_router.post("/start")
def start_job(req: JobStartReq) -> Dict[str, Any]:
    job = _new_job(req.goal.strip(), autostart=req.autostart, meta=req.meta)
    return {"ok": True, "job": {k: v for k, v in job.items() if k != "events"}}


@jobs_router.post("/checkpoint")
def push_checkpoint(req: CheckpointReq) -> Dict[str, Any]:
    # autocreate si se habilita por env
    job = _JOBS.get(req.job_id)
    if not job:
        if AUTOCREATE_ON_CHECKPOINT:
            job = _new_job(goal=f"auto:{req.job_id}", autostart=True)
            job["job_id"] = req.job_id  # respeta el ID recibido
            _JOBS[req.job_id] = job
        else:
            raise HTTPException(status_code=404, detail="job_id not found")

    payload = req.model_dump()
    _append_event(job, payload)

    t = (req.type or "").lower()

    if t == "plan":
        # sólo fija total cuando venga la lista de steps
        if isinstance(req.steps, list):
            job["steps_total"] = max(0, int(len(req.steps)))
        job["status"] = job.get("status", "running") or "running"
        job["last_event"] = "plan"

    elif t == "step_start":
        job["last_event"] = f"step_start:{req.step_id or ''}"

    elif t == "step_end":
        # incrementa sin superar el total
        job["steps_done"] = min(job.get("steps_total", 0), job.get("steps_done", 0) + 1)
        job["last_event"] = f"step_end:{req.step_id or ''}"

    elif t == "status":
        if req.status:
            job["status"] = req.status
        job["last_event"] = "status"

    else:
        # evento genérico (no toca contadores)
        job["last_event"] = t or "event"

    return {"ok": True}


@jobs_router.get("/status")
def job_status(job_id: str = Query(...)) -> Dict[str, Any]:
    job = _get_job(job_id)
    status_obj = {
        "job_id": job["job_id"],
        "updated_at": job["updated_at"],
        "status": job["status"],         # estado textual (running/completed/...)
        "steps_total": job["steps_total"],
        "steps_done": job["steps_done"],
        "last_event": job["last_event"],
        "meta": job.get("meta", {}),
    }
    # <-- ojo: ajustamos la forma del payload a la que espera el test
    return {"ok": True, "status": status_obj}


@jobs_router.get("/tail")
def job_tail(job_id: str = Query(...), n: int = Query(10, ge=1, le=EVENTS_MAX)) -> Dict[str, Any]:
    job = _get_job(job_id)
    ev = list(job["events"])[-n:]
    return {"ok": True, "events": ev}


# (Opcional) cancelar job
@jobs_router.post("/cancel")
def cancel_job(job_id: str = Query(...)) -> Dict[str, Any]:
    job = _get_job(job_id)
    job["status"] = "cancelled"
    _append_event(job, {"type": "status", "status": "cancelled"})
    job["last_event"] = "status"
    return {"ok": True}
