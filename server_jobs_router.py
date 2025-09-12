# server_jobs_router.py
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from jobs.manager import JobManager

API_KEY_ENV = "API_KEY"

jobs_router = APIRouter(prefix="/jobs", tags=["jobs"])
_manager = JobManager()

# ---------- auth ----------
def _require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    expected = os.getenv(API_KEY_ENV, "")
    if not expected:
        # Para evitar abrir endpoints sin llave por un .env mal configurado
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY not set")
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Forbidden: invalid or missing API Key")

# ---------- schemas ----------
class StartJobBody(BaseModel):
    goal: str = Field(..., description="Objetivo del job")
    plan: Optional[List[Dict[str, Any]]] = Field(default=None, description="Plan opcional (lista de pasos)")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    autostart: bool = Field(default=False, description="Colocar el job en running inmediatamente")

class CheckpointBody(BaseModel):
    job_id: str
    step_id: Optional[str] = None
    status: Optional[str] = Field(default=None, description="running|completed|failed|paused|cancelled")
    analysis: Optional[Any] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    type: Optional[str] = Field(default="step", description="step|step_start|step_end|plan|job_meta|status|custom")

# ---------- endpoints ----------
@jobs_router.post("/start")
def start_job(body: StartJobBody, _: None = Depends(_require_api_key)):
    """
    Crea un job con estado queued|running, adjunta plan opcional y emite checkpoints.
    """
    job = _manager.create_job(goal=body.goal, plan=body.plan, params=body.params, autostart=body.autostart)
    return {"ok": True, "job": job.__dict__}

@jobs_router.get("/status")
def job_status(job_id: str = Query(...), _: None = Depends(_require_api_key)):
    try:
        latest = _manager.latest(job_id)
        return {"ok": True, "status": latest}
    except KeyError:
        raise HTTPException(status_code=404, detail="job_id not found")

@jobs_router.get("/tail")
def job_tail(job_id: str = Query(...), n: int = Query(100, ge=1, le=1000), _: None = Depends(_require_api_key)):
    try:
        events = _manager.tail(job_id, n=n)
        return {"ok": True, "events": events}
    except KeyError:
        raise HTTPException(status_code=404, detail="job_id not found")

@jobs_router.get("/list")
def list_jobs(_: None = Depends(_require_api_key)):
    return {"ok": True, "jobs": _manager.list_jobs()}

@jobs_router.post("/pause")
def pause_job(job_id: str, _: None = Depends(_require_api_key)):
    try:
        job = _manager.update_status(job_id, "paused")
        return {"ok": True, "job": job.__dict__}
    except KeyError:
        raise HTTPException(status_code=404, detail="job_id not found")

@jobs_router.post("/resume")
def resume_job(job_id: str, _: None = Depends(_require_api_key)):
    try:
        job = _manager.update_status(job_id, "running")
        return {"ok": True, "job": job.__dict__}
    except KeyError:
        raise HTTPException(status_code=404, detail="job_id not found")

@jobs_router.post("/cancel")
def cancel_job(job_id: str, _: None = Depends(_require_api_key)):
    try:
        job = _manager.update_status(job_id, "cancelled")
        return {"ok": True, "job": job.__dict__}
    except KeyError:
        raise HTTPException(status_code=404, detail="job_id not found")

@jobs_router.post("/checkpoint")
def push_checkpoint(body: CheckpointBody, _: None = Depends(_require_api_key)):
    """
    Punto de integración: desde /plan/execute o /reason/execute
    puedes POSTear aquí para registrar análisis, resultados, errores, etc.
    """
    ev = body.dict()
    job_id = ev.pop("job_id")
    _manager.checkpoint(job_id, ev)
    return {"ok": True}
