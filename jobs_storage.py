# jobs_storage.py — Fase 6: persistencia de jobs, índice, rotación y métricas
from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

# -----------------------------------------------------------------------------
# Config dinámica (se lee en cada operación para permitir monkeypatch en tests)
# -----------------------------------------------------------------------------
def _cfg_jobs_dir() -> str:
    return os.getenv("JOBS_DIR", "data/jobs")

def _cfg_jobs_max() -> int:
    v = os.getenv("JOBS_MAX", "0")
    try:
        n = int(v)
        return max(0, n)
    except Exception:
        return 0

def _cfg_jobs_ttl_seconds() -> int:
    v = os.getenv("JOBS_TTL_SECONDS", "0")
    try:
        n = int(v)
        return max(0, n)
    except Exception:
        return 0

def _ensure_dirs() -> str:
    d = _cfg_jobs_dir()
    os.makedirs(d, exist_ok=True)
    return d

def _index_path() -> str:
    return os.path.join(_cfg_jobs_dir(), "index.json")

def _trash_dir() -> str:
    return os.path.join(_cfg_jobs_dir(), "trash")


# -----------------------------------------------------------------------------
# Tiempo / utilidades
# -----------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _parse_iso(ts: str) -> datetime:
    # admite "...Z" y offsets ISO, fallback naive-utc
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts[:-1]).replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.now(timezone.utc)


# -----------------------------------------------------------------------------
# Carga/guardado atómico de JSON
# -----------------------------------------------------------------------------
def _atomic_write(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# -----------------------------------------------------------------------------
# Estructura del índice en memoria
# -----------------------------------------------------------------------------
_INDEX: Dict[str, Any] = {
    "schema_version": 1,
    "jobs": {},                 # job_id -> meta
    "purged_jobs_total": 0
}

def load_index() -> Dict[str, Any]:
    _ensure_dirs()
    p = _index_path()
    if not os.path.exists(p):
        save_index()
        return _INDEX
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("index.json corrupto")
        # actualiza _INDEX en caliente
        _INDEX.clear()
        _INDEX.update(data)
    except Exception:
        # reconstruye limpio
        _INDEX.clear()
        _INDEX.update({"schema_version": 1, "jobs": {}, "purged_jobs_total": 0})
        save_index()
    return _INDEX

def save_index() -> None:
    _ensure_dirs()
    _atomic_write(_index_path(), _INDEX)

def index_get(job_id: str) -> Optional[Dict[str, Any]]:
    load_index()
    return _INDEX.get("jobs", {}).get(job_id)

def index_all() -> List[Dict[str, Any]]:
    load_index()
    jobs = list((_INDEX.get("jobs") or {}).values())
    # orden: más recientes primero
    jobs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return jobs

def _job_file(job_id: str) -> str:
    return os.path.join(_cfg_jobs_dir(), f"{job_id}.json")

def index_add_or_update(job_snapshot: Dict[str, Any]) -> None:
    """Actualiza metadatos mínimos del índice basado en el snapshot del job."""
    load_index()
    job_id = job_snapshot.get("job_id")
    if not job_id:
        return
    p = _job_file(job_id)
    size = os.path.getsize(p) if os.path.exists(p) else 0
    meta = {
        "job_id": job_id,
        "status": job_snapshot.get("status", "unknown"),
        "created_at": job_snapshot.get("created_at"),
        "updated_at": job_snapshot.get("updated_at") or _now_iso(),
        "steps_total": job_snapshot.get("steps_total", 0),
        "steps_done": job_snapshot.get("steps_done", 0),
        "last_seq": job_snapshot.get("cursor", 0),
        "file": os.path.basename(p),
        "size_bytes": int(size),
    }
    _INDEX.setdefault("jobs", {})[job_id] = meta
    save_index()

def index_remove(job_id: str, *, soft_delete: bool = False) -> bool:
    """Elimina del índice y borra/mueve el archivo; devuelve True si existía."""
    load_index()
    existed = False
    if job_id in (_INDEX.get("jobs") or {}):
        existed = True
        _INDEX["jobs"].pop(job_id, None)
        save_index()

    # Archivo en disco
    p = _job_file(job_id)
    if os.path.exists(p):
        if soft_delete:
            os.makedirs(_trash_dir(), exist_ok=True)
            shutil.move(p, os.path.join(_trash_dir(), os.path.basename(p)))
        else:
            os.remove(p)
        existed = True
    return existed

def rotate_if_needed() -> List[str]:
    """Aplica rotación por TTL y/o MAX. Devuelve lista de job_ids purgados."""
    load_index()
    purged: List[str] = []

    # TTL
    ttl = _cfg_jobs_ttl_seconds()
    if ttl > 0:
        now = datetime.now(timezone.utc)
        to_purge = []
        for jid, meta in list((_INDEX.get("jobs") or {}).items()):
            ts = _parse_iso(meta.get("updated_at") or meta.get("created_at") or _now_iso())
            age = (now - ts).total_seconds()
            if age > ttl:
                to_purge.append(jid)
        for jid in to_purge:
            index_remove(jid, soft_delete=False)
            purged.append(jid)

    # MAX
    max_jobs = _cfg_jobs_max()
    if max_jobs > 0:
        jobs = list((_INDEX.get("jobs") or {}).values())
        jobs.sort(key=lambda m: m.get("updated_at", ""))  # más viejos primero
        while len(jobs) > max_jobs:
            victim = jobs.pop(0)
            jid = victim.get("job_id")
            if jid:
                index_remove(jid, soft_delete=False)
                purged.append(jid)
        # re-sync lista en memoria (por si removimos)
        load_index()

    if purged:
        _INDEX["purged_jobs_total"] = int(_INDEX.get("purged_jobs_total", 0)) + len(purged)
        save_index()
    return purged

def disk_usage_bytes() -> int:
    d = _cfg_jobs_dir()
    if not os.path.exists(d):
        return 0
    total = 0
    for name in os.listdir(d):
        if not name.endswith(".json"):
            continue
        if name == "index.json":
            continue
        p = os.path.join(d, name)
        if os.path.isfile(p):
            total += os.path.getsize(p)
    return total

def get_persistence_metrics() -> Dict[str, Any]:
    load_index()
    return {
        "jobs_dir": _cfg_jobs_dir(),
        "persisted_jobs": len(_INDEX.get("jobs") or {}),
        "purged_jobs_total": int(_INDEX.get("purged_jobs_total", 0)),
        "jobs_disk_usage_bytes": disk_usage_bytes(),
    }
