# -*- coding: utf-8 -*-
"""
Jobs runner actualizado
- Lee jobs de data/jobs/jobs.jsonl (JOBS_JSONL) o, en su defecto, de JOBS_FILE (legado).
- Tipos de job:
    * loop_query: ejecuta el QuipuLoop con la 'query'
    * index_folder: invoca scripts.index_folder (fresh/append)
    * repair_semantic_store: repara data/semantic_store.json
- Programación:
    * {"cron": {"every_minutes": 30}}  -> ejecuta si han pasado ≥30 min
    * {"cron": {"daily_time": "09:30"}} -> una vez al día a la hora dada
- Persistencia de last_run_ts sólo si ok=true.
"""
from __future__ import annotations
import json, time, subprocess, sys, os, traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

# settings nuevo + fallback a legado
try:
    from root.settings import JOBS_JSONL as _JOBS_PATH
except Exception:
    try:
        from root.settings import JOBS_FILE as _JOBS_PATH  # legacy
    except Exception:
        _JOBS_PATH = str(Path("data") / "jobs" / "jobs.jsonl")

from root.settings import SEMANTIC_STORE_JSON
from rag.retriever import repair as repair_semantic_store

# QuipuLoop (si está disponible)
try:
    from root.quipu_loop import QuipuLoop  # type: ignore
except Exception:
    QuipuLoop = None  # pragma: no cover


# ---------- IO ----------
def _load_jobs() -> List[Dict[str, Any]]:
    p = Path(_JOBS_PATH)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(("#", "//")):
                continue
            if line and line[0] == "\ufeff":
                line = line[1:]  # limpia BOM
            try:
                j = json.loads(line)
                j.setdefault("enabled", True)
                j.setdefault("project_id", "default")
                j.setdefault("last_run_ts", 0)
                out.append(j)
            except Exception:
                continue
    return out

def _save_jobs(jobs: List[Dict[str, Any]]) -> None:
    p = Path(_JOBS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")


# ---------- Scheduling ----------
def _today_at(now_ts: int, hh: int, mm: int) -> int:
    lt = time.localtime(now_ts)
    sched = time.struct_time((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0,
                              lt.tm_wday, lt.tm_yday, lt.tm_isdst))
    return int(time.mktime(sched))

def due(job: Dict[str, Any], now_ts: Optional[int] = None) -> bool:
    now_ts = now_ts or int(time.time())
    if not job.get("enabled", True):
        return False
    cron = job.get("cron", {}) or {}
    last = int(job.get("last_run_ts", 0))

    if "every_minutes" in cron:
        return (now_ts - last) >= int(cron["every_minutes"]) * 60

    if "daily_time" in cron:
        hh, mm = map(int, str(cron["daily_time"]).split(":"))
        sched_ts = _today_at(now_ts, hh, mm)
        return (now_ts >= sched_ts) and (last < sched_ts)

    return False


# ---------- Ejecución ----------
def _run_loop_query(job: Dict[str, Any]) -> Dict[str, Any]:
    if QuipuLoop is None:
        return {"ok": False, "error": "QuipuLoop no disponible"}
    q = str(job.get("query", ""))
    project_id = str(job.get("project_id", "default"))
    t0 = time.perf_counter()
    res = QuipuLoop(project_id=project_id).run(q)
    dt_ms = int((time.perf_counter() - t0) * 1000)
    verified_ok = res.get("verified", {}).get("ok", True)
    ok = bool(res.get("output")) and bool(verified_ok)
    return {"ok": ok, "result": res, "latency_ms": dt_ms, "steps": res.get("steps"), "stats": res.get("stats")}

def _run_index_folder(job: Dict[str, Any]) -> Dict[str, Any]:
    args = job.get("args", {}) or {}
    cmd = [sys.executable, "-m", "scripts.index_folder", "--path", args.get("path", "./data")]
    if "ext" in args: cmd += ["--ext", args["ext"]]
    if "chunk" in args: cmd += ["--chunk", str(args["chunk"])]
    if "overlap" in args: cmd += ["--overlap", str(args["overlap"])]
    if "mode" in args: cmd += ["--mode", args["mode"]]
    if "dedupe" in args: cmd += ["--dedupe", str(args["dedupe"])]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {"ok": proc.returncode == 0, "cmd": " ".join(cmd), "stdout": proc.stdout, "stderr": proc.stderr}

def _run_repair(_job: Dict[str, Any]) -> Dict[str, Any]:
    res = repair_semantic_store()
    res["path"] = SEMANTIC_STORE_JSON
    res["ok"] = bool(res.get("ok", False))
    return res

def run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    jtype = job.get("job") or job.get("type") or "loop_query"
    try:
        if jtype == "loop_query":
            return _run_loop_query(job)
        elif jtype == "index_folder":
            return _run_index_folder(job)
        elif jtype == "repair_semantic_store":
            return _run_repair(job)
        else:
            return {"ok": False, "error": f"job type '{jtype}' no soportado"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=3)}

def run_due_jobs(now_ts: Optional[int] = None) -> List[Dict[str, Any]]:
    now_ts = now_ts or int(time.time())
    jobs = _load_jobs()
    results: List[Dict[str, Any]] = []
    dirty = False

    for j in jobs:
        if not due(j, now_ts):
            continue
        started = int(time.time())
        r = run_job(j)
        item = {
            "job_id": j.get("id"),
            "job_type": j.get("job") or j.get("type") or "loop_query",
            "project_id": j.get("project_id", "default"),
            "query": j.get("query", ""),
            "scheduled_at": now_ts,
            "started_at": started,
            **r,
        }
        results.append(item)
        if item.get("ok"):
            j["last_run_ts"] = now_ts
            dirty = True

    if dirty:
        _save_jobs(jobs)
    return results

def run_job_by_id(job_id: str) -> Dict[str, Any]:
    jobs = _load_jobs()
    for j in jobs:
        if j.get("id") == job_id:
            started = int(time.time())
            r = run_job(j)
            if r.get("ok"):
                j["last_run_ts"] = int(time.time())
                _save_jobs(jobs)
            return {
                "job_id": j.get("id"),
                "job_type": j.get("job") or j.get("type") or "loop_query",
                "project_id": j.get("project_id", "default"),
                "query": j.get("query", ""),
                "started_at": started,
                **r,
            }
    return {"ok": False, "error": f"job '{job_id}' not found"}

if __name__ == "__main__":
    # pequeño runner manual
    print(json.dumps(run_due_jobs(), ensure_ascii=False, indent=2))
