# server_jobs_router.py — Fase 6: Jobs + persistencia + índice + rotación + métricas
from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, cast
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Persistencia / índice (Fase 6)
from jobs_storage import (
    index_all,
    index_remove,
    index_add_or_update,
    rotate_if_needed,
    index_get,
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
JOBS_DIR = os.getenv("JOBS_DIR", "data/jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

EVENTS_MAX = int(os.getenv("JOBS_EVENTS_MAX", "2000"))
AUTOCREATE_ON_CHECKPOINT = os.getenv("JOBS_AUTOCREATE", "0") in {"1", "true", "True"}

jobs_router = APIRouter(prefix="/jobs", tags=["jobs"])

# In-memory cache (rehidratación on-demand)
_JOBS: Dict[str, Dict[str, Any]] = {}


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _job_path(job_id: str) -> str:
    return os.path.join(JOBS_DIR, f"{job_id}.json")


def _snapshot(job: Dict[str, Any]) -> Dict[str, Any]:
    snap = dict(job)
    snap["events"] = list(job.get("events", []))
    snap["schema_version"] = 1
    return snap


def _atomic_write(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _save(job: Dict[str, Any]) -> None:
    """Guarda snapshot, actualiza índice y aplica rotación."""
    p = _job_path(job["job_id"])
    _atomic_write(p, _snapshot(job))
    # Actualiza índice y dispara rotación
    index_add_or_update(job)
    purged = rotate_if_needed()
    # Saca del cache los purgados, para que tail/status devuelvan 404
    for jid in purged:
        _JOBS.pop(jid, None)


from typing import Any, Dict, List, Deque, Optional, cast
from collections import deque
import json, os

def _load(job_id: str) -> Optional[Dict[str, Any]]:
    p = _job_path(job_id)
    if not os.path.exists(p):
        return None

    with open(p, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = cast(Dict[str, Any], json.load(f))

    raw = data.get("events")
    ev: List[Dict[str, Any]] = cast(List[Dict[str, Any]], raw or [])

    # ✅ Tipamos la variable 'events' y luego la metemos al dict (sin type-comment)
    events: Deque[Dict[str, Any]] = deque(ev[-EVENTS_MAX:], maxlen=EVENTS_MAX)
    data["events"] = events

    return data


def _get_job(job_id: str) -> Dict[str, Any]:
    """
    Obtiene un job desde el caché; si no está o está obsoleto, intenta cargarlo
    desde disco. Si no existe en disco/índice, responde 404 e invalida el caché.
    """
    job = _JOBS.get(job_id)

    # Si está en caché, valida que siga existiendo en disco/índice.
    if job is not None:
        file_exists = os.path.exists(_job_path(job_id))
        in_index = index_get(job_id) is not None
        if not (file_exists and in_index):
            _JOBS.pop(job_id, None)  # inválida caché obsoleto
            job = None

    # Si no lo tenemos (o lo invalidamos), intenta cargar de disco
    if job is None:
        job = _load(job_id)
        if job:
            _JOBS[job_id] = job

    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    return job


def _append_event(job: Dict[str, Any], payload: Dict[str, Any]) -> None:
    evt = dict(payload)
    evt["ts"] = _now_iso()
    evt["seq"] = job.get("cursor", 0) + 1
    job["cursor"] = evt["seq"]
    job["events"].append(evt)
    job["updated_at"] = evt["ts"]


def _new_job(
    goal: str,
    autostart: bool = True,
    meta: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
    initial_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    jid = (job_id or uuid4().hex[:8]).strip()
    if jid in _JOBS or os.path.exists(_job_path(jid)):
        # Idempotencia: devuelve el existente si ya hay registro
        existing = _JOBS.get(jid) or _load(jid)
        if existing:
            _JOBS[jid] = existing
            return existing

    job = {
        "job_id": jid,
        "goal": goal,
        "status": "running" if autostart else "created",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "steps_total": 0,
        "steps_done": 0,
        "last_event": "",
        "meta": meta or {},
        "state": initial_state or {},
        "cursor": 0,
        "events": deque(maxlen=EVENTS_MAX),  # type: Deque[Dict[str, Any]]
    }
    # Primer evento
    job["events"].append({"type": "job_meta", "status": job["status"], "meta": "", "ts": _now_iso(), "seq": 1})
    job["cursor"] = 1
    _JOBS[jid] = job
    _save(job)
    return job


def _apply_checkpoint_side_effects(job: Dict[str, Any], t: str, req: "CheckpointReq") -> None:
    if t == "plan":
        if isinstance(req.steps, list):
            job["steps_total"] = max(0, int(len(req.steps)))
        job["status"] = job.get("status", "running") or "running"
        job["last_event"] = "plan"
    elif t == "step_start":
        job["last_event"] = f"step_start:{req.step_id or ''}"
    elif t == "step_end":
        job["steps_done"] = min(job.get("steps_total", 0), job.get("steps_done", 0) + 1)
        job["last_event"] = f"step_end:{req.step_id or ''}"
    elif t == "status":
        if req.status:
            job["status"] = req.status
        job["last_event"] = "status"
    else:
        job["last_event"] = t or "event"

    if req.state:
        job_state = job.setdefault("state", {})
        job_state.update(req.state)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class JobStartReq(BaseModel):
    goal: str = Field(..., min_length=1)
    autostart: bool = True
    meta: Dict[str, Any] = Field(default_factory=dict)
    job_id: Optional[str] = Field(default=None)
    state: Dict[str, Any] = Field(default_factory=dict)


class CheckpointReq(BaseModel):
    job_id: str = Field(..., min_length=1)
    type: str = Field(..., description="plan | step_start | step_end | status | meta | event")
    step_id: Optional[str] = None
    steps: Optional[List[Dict[str, Any]]] = None
    status: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    analysis: Optional[Any] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    state: Optional[Dict[str, Any]] = None


class ResumeReq(BaseModel):
    job_id: str = Field(..., min_length=1)
    mode: str = Field(default="continue", description="continue | reset")
    from_step: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class ReplayReq(BaseModel):
    job_id: str = Field(..., min_length=1)
    from_event: Optional[int] = None
    to_event: Optional[int] = None
    # aliases de compatibilidad
    from_event_idx: Optional[int] = None
    to_event_idx: Optional[int] = None
    mode: str = Field(default="dry-run", description="dry-run | apply")


# -----------------------------------------------------------------------------
# Endpoints (base + fase 6)
# -----------------------------------------------------------------------------
@jobs_router.post("/start")
def start_job(req: JobStartReq) -> Dict[str, Any]:
    job = _new_job(
        req.goal.strip(),
        autostart=req.autostart,
        meta=req.meta,
        job_id=req.job_id,
        initial_state=req.state or {},
    )
    idempotent = job["goal"] == req.goal.strip() or req.job_id is not None
    out = {k: v for k, v in job.items() if k not in {"events"}}
    return {"ok": True, "idempotent": idempotent, "job": out}


@jobs_router.post("/checkpoint")
def push_checkpoint(req: CheckpointReq) -> Dict[str, Any]:
    job = _JOBS.get(req.job_id)
    if not job:
        if AUTOCREATE_ON_CHECKPOINT:
            job = _new_job(goal=f"auto:{req.job_id}", autostart=True, job_id=req.job_id)
        else:
            job = _load(req.job_id)
            if job:
                _JOBS[req.job_id] = job
            else:
                raise HTTPException(status_code=404, detail="job_id not found")

    payload = req.model_dump()
    _append_event(job, payload)
    _apply_checkpoint_side_effects(job, (req.type or "").lower(), req)
    _save(job)
    return {"ok": True}


@jobs_router.get("/status")
def job_status(job_id: str = Query(...)) -> Dict[str, Any]:
    job = _get_job(job_id)
    status_obj = {
        "job_id": job["job_id"],
        "updated_at": job["updated_at"],
        "status": job["status"],
        "steps_total": job["steps_total"],
        "steps_done": job["steps_done"],
        "last_event": job["last_event"],
        "cursor": job.get("cursor", len(job["events"])),
        "meta": job.get("meta", {}),
    }
    return {"ok": True, "status": status_obj}


@jobs_router.get("/tail")
def job_tail(
    job_id: str = Query(...),
    n: Optional[int] = Query(None, ge=1, le=EVENTS_MAX),
    since: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=EVENTS_MAX),
) -> Dict[str, Any]:
    # Guardia contra jobs purgados o borrados (invalida caché si quedó)
    if (index_get(job_id) is None) or (not os.path.exists(_job_path(job_id))):
        _JOBS.pop(job_id, None)
        raise HTTPException(status_code=404, detail="job_id not found")

    job = _get_job(job_id)
    events = list(job["events"])
    total = len(events)

    if n is not None:
        ev = events[-n:]
        return {"ok": True, "events": ev, "total": total}

    # modo incremental
    start_idx = 0
    if since > 0:
        for i, e in enumerate(events):
            if int(e.get("seq", i + 1)) > since:
                start_idx = i
                break
        else:
            start_idx = total

    end_idx = min(total, start_idx + limit)
    ev = events[start_idx:end_idx]
    next_since = ev[-1]["seq"] if ev else since
    return {"ok": True, "events": ev, "total": total, "next_since": next_since}


@jobs_router.get("/dump")
def job_dump(job_id: str = Query(...)) -> Dict[str, Any]:
    job = _get_job(job_id)
    return {"ok": True, "job": _snapshot(job)}


@jobs_router.post("/resume")
def job_resume(req: ResumeReq) -> Dict[str, Any]:
    job = _JOBS.get(req.job_id)
    if not job:
        job = _load(req.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_id not found")
        _JOBS[req.job_id] = job

    # evento cmd (compat con tests)
    _append_event(job, {"type": "cmd", "cmd": "resume", "args": {"mode": req.mode, "from_step": req.from_step, "meta": req.meta}})

    resumed = False
    if req.mode == "reset":
        job["steps_done"] = 0
        job["status"] = "running"
        job["last_event"] = "status"
        resumed = True
    else:
        if job.get("status") != "running":
            job["status"] = "running"
            job["last_event"] = "status"
            resumed = True

    _append_event(job, {"type": "status", "status": "running", "meta": req.meta})
    _save(job)

    out = {k: v for k, v in job.items() if k != "events"}
    return {"ok": True, "resumed": resumed, "job": out}


@jobs_router.post("/replay")
def job_replay(req: ReplayReq) -> Dict[str, Any]:
    job = _get_job(req.job_id)
    start_seq = req.from_event if req.from_event is not None else (req.from_event_idx or 0)
    end_seq = req.to_event if req.to_event is not None else req.to_event_idx

    # evento cmd (compat con tests)
    _append_event(job, {"type": "cmd", "cmd": "replay", "args": {"from_event": start_seq, "to_event": end_seq, "mode": req.mode}})
    _save(job)

    events = list(job["events"])
    total = len(events)

    # recorta ventana
    start_idx = 0
    if start_seq and start_seq > 0:
        for i, e in enumerate(events):
            if int(e.get("seq", i + 1)) >= start_seq:
                start_idx = i
                break
        else:
            start_idx = total
    end_idx = total
    if end_seq is not None:
        for i, e in enumerate(events):
            if int(e.get("seq", i + 1)) > end_seq:
                end_idx = i
                break

    window = events[start_idx:end_idx]
    steps_total = job["steps_total"]
    steps_done = 0
    status = job["status"]

    for e in window:
        t = (e.get("type") or "").lower()
        if t == "plan" and isinstance(e.get("steps"), list):
            steps_total = max(steps_total, int(len(e["steps"])))
        elif t == "step_end":
            steps_done = min(steps_total, steps_done + 1)
        elif t == "status" and e.get("status"):
            status = e["status"]

    summary = {
        "from_seq": window[0]["seq"] if window else start_seq,
        "to_seq": window[-1]["seq"] if window else end_seq,
        "events": len(window),
        "steps_total": steps_total,
        "steps_done": steps_done,
        "status": status,
    }

    if req.mode == "apply":
        job["steps_total"] = steps_total
        job["steps_done"] = steps_done
        job["status"] = status
        job["last_event"] = "replay"
        _append_event(job, {"type": "meta", "meta": {"replay_applied": True, **summary}})
        _save(job)

    return {"ok": True, "summary": summary, "preview": window}


@jobs_router.post("/cancel")
def cancel_job(job_id: str = Query(...)) -> Dict[str, Any]:
    job = _get_job(job_id)
    job["status"] = "cancelled"
    _append_event(job, {"type": "status", "status": "cancelled"})
    job["last_event"] = "status"
    _save(job)
    return {"ok": True}


# ---------------------- Endpoints NUEVOS Fase 6 ------------------------------#
@jobs_router.get("/list")
def list_jobs() -> Dict[str, Any]:
    """Lista de jobs desde el índice (sin abrir archivos)."""
    jobs = index_all()
    return {"ok": True, "jobs": jobs, "count": len(jobs)}


@jobs_router.get("/load")
def load_job(job_id: str = Query(...)) -> Dict[str, Any]:
    """
    Rehidrata un job desde disco (si existe) hacia el cache y devuelve snapshot completo.
    Además sincroniza índice/rotación.
    """
    job = _load(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    _JOBS[job_id] = job
    _save(job)  # sincroniza índice y updated_at; puede purgar terceros
    return {"ok": True, "job": _snapshot(job)}


@jobs_router.delete("/delete")
def delete_job(job_id: str = Query(...)) -> Dict[str, Any]:
    """Borra del índice y del disco (hard delete)."""
    existed = index_remove(job_id, soft_delete=False)
    _JOBS.pop(job_id, None)
    if not existed:
        raise HTTPException(status_code=404, detail="job_id not found")
    return {"ok": True}
