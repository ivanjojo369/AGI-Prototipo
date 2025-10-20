# server_ttl_router.py
"""
TTL sweep + métricas Prometheus para jobs persistidos (modelo: archivos .json e index.json).
- Endpoint:  POST /jobs/ttl/purge   (dry_run opcional)
- Métricas:  GET  /metrics           (text/plain; version=0.0.4)
Auth: respeta ALLOW_TEST_NO_AUTH=1; en prod requiere x-api-key o Authorization.
Config:
  - JOBS_TTL_SECONDS (si no se pasa ttl_seconds en query)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from starlette.status import HTTP_401_UNAUTHORIZED

# Integramos con tu capa real de persistencia/índice
import jobs_storage

ttl_router = APIRouter(tags=["jobs"])

# ------------------- Utils ------------------- #
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}

def _auth(request: Request):
    if _env_bool("ALLOW_TEST_NO_AUTH", False):
        return
    api_key = request.headers.get("x-api-key")
    bearer = request.headers.get("authorization")
    if not api_key and not bearer:
        raise HTTPException(HTTP_401_UNAUTHORIZED, "Missing API key")

def _parse_iso(ts: str) -> datetime:
    # admite "....Z" o ISO con offset
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts[:-1]).replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.now(timezone.utc)

def _expired_job_ids(ttl_seconds: int) -> List[str]:
    """Calcula candidatos a purga (por TTL) sin borrar nada."""
    ttl = max(0, int(ttl_seconds or 0))
    if ttl <= 0:
        return []
    idx = jobs_storage.load_index()
    jobs: Dict[str, Dict] = (idx.get("jobs") or {})
    now = datetime.now(timezone.utc)
    out: List[str] = []
    for jid, meta in jobs.items():
        ts_raw = meta.get("updated_at") or meta.get("created_at")
        ts = _parse_iso(ts_raw) if ts_raw else datetime.now(timezone.utc)
        age = (now - ts).total_seconds()
        if age > ttl:
            out.append(jid)
    return out

# ------------------- Métricas locales (en memoria) ------------------- #
@dataclass
class _TTLMetrics:
    ttl_seconds: int = 0
    ttl_purger_runs_total: int = 0
    jobs_expired_total: int = 0
    ttl_purger_last_run_epoch: float = 0.0

_metrics = _TTLMetrics()

# ------------------- Endpoints ------------------- #
@ttl_router.post("/jobs/ttl/purge")
async def purge_jobs_ttl(
    request: Request,
    ttl_seconds: Optional[int] = None,
    dry_run: bool = False,
):
    """
    Ejecuta un barrido TTL sobre el índice (archivos .json).
    - dry_run=True => sólo lista candidatos (no borra).
    - dry_run=False => borra y actualiza métricas persistentes.
    """
    _auth(request)

    ttl = int(ttl_seconds or os.getenv("JOBS_TTL_SECONDS", "0") or 0)
    if ttl <= 0:
        return {"ok": False, "error": "TTL no configurado (ttl_seconds o env JOBS_TTL_SECONDS)"}

    candidates = _expired_job_ids(ttl)

    purged = 0
    if not dry_run and candidates:
        for jid in candidates:
            try:
                jobs_storage.index_remove(jid, soft_delete=False)
                purged += 1
            except Exception:
                # continúa aunque falle alguno
                continue
        # Actualiza contador persistente en index.json
        try:
            jobs_storage.record_purged_jobs(purged)
        except Exception:
            pass

    # Actualiza métricas en memoria (para /metrics)
    _metrics.ttl_seconds = ttl
    _metrics.ttl_purger_runs_total += 1
    _metrics.jobs_expired_total += purged
    _metrics.ttl_purger_last_run_epoch = time.time()

    # Métricas de persistencia reales (coherentes con tu index.json)
    persist = jobs_storage.get_persistence_metrics()

    return {
        "ok": True,
        "ttl_seconds": ttl,
        "dry_run": dry_run,
        "candidates": candidates,
        "purged": purged,
        **persist,
    }

@ttl_router.get("/metrics")
async def prometheus_metrics(_: Request):
    """
    Exposición en formato Prometheus (text/plain; version=0.0.4).
    Combina métricas TTL en memoria + métricas de persistencia reales.
    """
    persist = jobs_storage.get_persistence_metrics()

    lines: List[str] = []

    def add(name: str, typ: str, val, help_text: Optional[str] = None):
        if help_text:
            lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {typ}")
        try:
            v = float(val)
        except Exception:
            v = 0.0
        lines.append(f"{name} {v}")

    # TTL
    add("jobs_ttl_seconds", "gauge", _metrics.ttl_seconds, "Configured TTL for jobs in seconds")
    add("ttl_purger_runs_total", "counter", _metrics.ttl_purger_runs_total, "Total TTL sweeps executed")
    add("jobs_expired_total", "counter", _metrics.jobs_expired_total, "Total jobs purged by TTL")
    add("ttl_purger_last_run_epoch", "gauge", _metrics.ttl_purger_last_run_epoch, "Last TTL sweep UNIX epoch")

    # Persistencia (desde index.json)
    add("persisted_jobs", "gauge", persist.get("persisted_jobs", 0), "Current jobs persisted in index")
    add("jobs_disk_usage_bytes", "gauge", persist.get("jobs_disk_usage_bytes", 0), "Disk usage for jobs dir")
    add("jobs_purged_total", "counter", persist.get("purged_jobs_total", 0), "Total jobs purged (persistent counter)")

    text = "\n".join(lines) + "\n"
    return Response(content=text, media_type="text/plain; version=0.0.4")
