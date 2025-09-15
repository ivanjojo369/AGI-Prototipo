# jobs/manager.py
from __future__ import annotations
import json, uuid, threading
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
    Indexa jobs y persiste su metadata en data/jobs/index.json.
    Los checkpoints de cada job se guardan vía CheckpointStore (JSONL).
    Además mantenemos un snapshot resumido por job: data/jobs/<id>.latest.json
    con status/steps/último evento para /jobs/status.
    """
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir or "data/jobs").resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.json"
        self._lock = threading.RLock()
        self.ckpts = CheckpointStore(self.base_dir)
        self._jobs: Dict[str, Job] = self._load_index()

        # Auto-sanar estados "running" de ejecuciones previas → "paused"
        changed = False
        for j in self._jobs.values():
            if j.status == "running":
                j.status = "paused"
                j.updated_at = _utcnow()
                # también snapshot
                info = self._load_latest(j.job_id)
                info["status"] = "paused"
                info["updated_at"] = j.updated_at
                self._save_latest(j.job_id, info)
                changed = True
        if changed:
            self._save_index()

    # ---------------- persistencia índice ----------------

    def _load_index(self) -> Dict[str, Job]:
        if not self.index_path.exists():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return {k: Job(**v) for k, v in data.items()}
        except Exception:
            return {}

    def _save_index(self) -> None:
        with self._lock:
            data = {k: asdict(v) for k, v in self._jobs.items()}
            self.index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- snapshot helpers ----------------

    def _latest_path(self, job_id: str) -> Path:
        return self.base_dir / f"{job_id}.latest.json"

    def _load_latest(self, job_id: str) -> Dict[str, Any]:
        p = self._latest_path(job_id)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        # default mínimo
        job = self._jobs.get(job_id)
        return {
            "job_id": job_id,
            "updated_at": _utcnow(),
            "status": (job.status if job else "queued"),
            "steps_total": len(job.plan) if job else 0,
            "steps_done": 0,
            "last_event": "",
        }

    def _save_latest(self, job_id: str, info: Dict[str, Any]) -> None:
        info = dict(info)
        info.setdefault("job_id", job_id)
        info["updated_at"] = _utcnow()
        self._latest_path(job_id).write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")

    # ---------------- API pública ----------------

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [asdict(j) for j in self._jobs.values()]

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def create_job(
        self,
        goal: str,
        plan: Optional[List[Dict[str, Any]]] = None,
        params: Optional[Dict[str, Any]] = None,
        autostart: bool = False,
    ) -> Job:
        job_id = str(uuid.uuid4())[:8]
        status = "running" if autostart else "queued"
        job = Job(job_id=job_id, goal=goal, status=status, params=params or {}, plan=plan or [])
        with self._lock:
            self._jobs[job_id] = job
            self._save_index()

        # Snapshot inicial
        snap = {
            "job_id": job_id,
            "status": status,
            "steps_total": len(job.plan),
            "steps_done": 0,
            "last_event": "job_meta",
        }
        self._save_latest(job_id, snap)

        # Checkpoints iniciales
        self.ckpts.append(job_id, {"type": "job_meta", "status": status, "meta": {"goal": goal, "params": job.params}})
        if job.plan:
            self.ckpts.append(job_id, {"type": "plan", "steps": job.plan, "status": status})
        return job

    def update_status(self, job_id: str, status: str) -> Job:
        job = self._require(job_id)
        job.status = status
        job.updated_at = _utcnow()
        with self._lock:
            self._jobs[job_id] = job
            self._save_index()
        # checkpoint + snapshot
        self.ckpts.append(job_id, {"type": "status", "status": status})
        snap = self._load_latest(job_id)
        snap["status"] = status
        snap["last_event"] = "status"
        self._save_latest(job_id, snap)
        return job

    def attach_plan(self, job_id: str, plan: List[Dict[str, Any]]) -> Job:
        job = self._require(job_id)
        job.plan = plan or []
        job.updated_at = _utcnow()
        with self._lock:
            self._jobs[job_id] = job
            self._save_index()
        # checkpoint + snapshot
        self.ckpts.append(job_id, {"type": "plan", "steps": job.plan, "status": job.status})
        snap = self._load_latest(job_id)
        snap["steps_total"] = len(job.plan)
        snap["last_event"] = "plan"
        self._save_latest(job_id, snap)
        return job

    def checkpoint(self, job_id: str, event: Dict[str, Any]) -> None:
        _ = self._require(job_id)
        # persistimos en JSONL
        self.ckpts.append(job_id, event)

        # y actualizamos snapshot
        et = (event.get("type") or "step").lower()
        snap = self._load_latest(job_id)

        if et == "plan":
            steps = event.get("steps") or []
            if isinstance(steps, list):
                snap["steps_total"] = len(steps)
            if event.get("status"):
                snap["status"] = event["status"]
            snap["last_event"] = "plan"

        elif et in ("step_end", "step"):
            if (event.get("status") or "").lower() == "completed":
                snap["steps_done"] = int(snap.get("steps_done", 0)) + 1
            snap["last_event"] = "step_end"

        elif et == "status":
            if event.get("status"):
                snap["status"] = event["status"]
            snap["last_event"] = "status"

        else:
            # step_start, job_meta, custom, etc.
            snap["last_event"] = et

        self._save_latest(job_id, snap)

    def latest(self, job_id: str) -> Dict[str, Any]:
        _ = self._require(job_id)
        return self._load_latest(job_id)

    def tail(self, job_id: str, n: int = 100) -> List[Dict[str, Any]]:
        _ = self._require(job_id)
        return self.ckpts.tail(job_id, n=n)

    # ---------------- helpers ----------------

    def _require(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"job_id not found: {job_id}")
        return job
