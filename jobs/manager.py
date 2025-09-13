# jobs/manager.py
from __future__ import annotations
import json, os, uuid, threading
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from .checkpoints import CheckpointStore

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"

def _utcnow() -> str:
    return datetime.utcnow().strftime(ISO)

@dataclass
class Job:
    job_id: str
    goal: str
    status: str = "queued"   # queued|running|paused|completed|failed|cancelled
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    params: Dict[str, Any] = field(default_factory=dict)
    plan: List[Dict[str, Any]] = field(default_factory=list)
    progress: Dict[str, Any] = field(default_factory=dict)

class JobManager:
    """
    Indexa jobs y persiste su metadata en data/jobs/index.json
    Los checkpoints de cada job se guardan vía CheckpointStore (JSONL).
    """
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir or "data/jobs").resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.json"
        self._lock = threading.RLock()
        self.ckpts = CheckpointStore(self.base_dir)
        self._jobs: Dict[str, Job] = self._load_index()

        # Auto-sanar estados "running" de ejecuciones previas → "paused"
        for j in self._jobs.values():
            if j.status == "running":
                j.status = "paused"
                j.updated_at = _utcnow()
        self._save_index()

    # --------- persistencia del índice ---------

    def _load_index(self) -> Dict[str, Job]:
        if not self.index_path.exists():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            jobs = {k: Job(**v) for k, v in data.items()}
            return jobs
        except Exception:
            return {}

    def _save_index(self) -> None:
        with self._lock:
            data = {k: asdict(v) for k, v in self._jobs.items()}
            self.index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # --------- API pública ---------

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [asdict(j) for j in self._jobs.values()]

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def create_job(self, goal: str, plan: Optional[List[Dict[str, Any]]] = None,
                   params: Optional[Dict[str, Any]] = None, autostart: bool = False) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(job_id=job_id, goal=goal, status="running" if autostart else "queued",
                  params=params or {}, plan=plan or [])
        with self._lock:
            self._jobs[job_id] = job
            self._save_index()
        # Primeros checkpoints
        self.ckpts.append(job_id, {"type": "job_meta", "status": job.status, "meta": {"goal": goal, "params": job.params}})
        if job.plan:
            self.ckpts.append(job_id, {"type": "plan", "steps": job.plan, "status": job.status})
        return job

    def update_status(self, job_id: str, status: str) -> Job:
        job = self._require(job_id)
        job.status = status
        job.updated_at = _utcnow()
        with self._lock:
            self._jobs[job_id] = job
            self._save_index()
        self.ckpts.append(job_id, {"type": "status", "status": status})
        return job

    def attach_plan(self, job_id: str, plan: List[Dict[str, Any]]) -> Job:
        job = self._require(job_id)
        job.plan = plan or []
        job.updated_at = _utcnow()
        with self._lock:
            self._jobs[job_id] = job
            self._save_index()
        self.ckpts.append(job_id, {"type": "plan", "steps": job.plan, "status": job.status})
        return job

    def checkpoint(self, job_id: str, event: Dict[str, Any]) -> None:
        _ = self._require(job_id)
        self.ckpts.append(job_id, event)

    def latest(self, job_id: str) -> Dict[str, Any]:
        _ = self._require(job_id)
        return self.ckpts.latest(job_id)

    def tail(self, job_id: str, n: int = 100) -> List[Dict[str, Any]]:
        _ = self._require(job_id)
        return self.ckpts.tail(job_id, n=n)

    # --------- helpers ---------

    def _require(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"job_id not found: {job_id}")
        return job
