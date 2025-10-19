# jobs/checkpoints.py
from __future__ import annotations
import json, os, threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"

def _utcnow() -> str:
    return datetime.utcnow().strftime(ISO)

class CheckpointStore:
    """
    Persistencia de checkpoints en JSONL por job, con tail eficiente y snapshot.
    Archivos:
      data/jobs/{job_id}.jsonl           -> stream de eventos
      data/jobs/{job_id}.latest.json     -> estado resumido rápido
    """
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir or "data/jobs").resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _job_paths(self, job_id: str) -> Dict[str, Path]:
        jl = self.base_dir / f"{job_id}.jsonl"
        latest = self.base_dir / f"{job_id}.latest.json"
        return {"jsonl": jl, "latest": latest}

    def append(self, job_id: str, event: Dict[str, Any]) -> None:
        """
        Agrega un checkpoint (línea JSON) y actualiza snapshot 'latest'.
        """
        with self._lock:
            paths = self._job_paths(job_id)
            event = dict(event or {})
            event.setdefault("ts", _utcnow())
            # Escribir JSONL
            with paths["jsonl"].open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            # Actualizar resumen (best-effort)
            try:
                latest = self._build_latest(paths["jsonl"])
                with paths["latest"].open("w", encoding="utf-8") as f:
                    json.dump(latest, f, ensure_ascii=False, indent=2)
            except Exception:
                # No bloquear por errores de resumen
                pass

    def tail(self, job_id: str, n: int = 100) -> List[Dict[str, Any]]:
        """
        Últimos n eventos del job.
        """
        paths = self._job_paths(job_id)
        if not paths["jsonl"].exists():
            return []
        lines: List[str] = []
        with self._lock, paths["jsonl"].open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while len(lines) <= n and size > 0:
                read = min(block, size)
                size -= read
                f.seek(size)
                chunk = f.read(read)
                data = chunk + data
                lines = data.splitlines()
        tail_lines = lines[-n:]
        out: List[Dict[str, Any]] = []
        for ln in tail_lines:
            try:
                out.append(json.loads(ln.decode("utf-8")))
            except Exception:
                continue
        return out

    def latest(self, job_id: str) -> Dict[str, Any]:
        """
        Devuelve el snapshot rápido; si no existe, intenta construirlo.
        """
        paths = self._job_paths(job_id)
        if paths["latest"].exists():
            try:
                return json.loads(paths["latest"].read_text(encoding="utf-8"))
            except Exception:
                pass
        if paths["jsonl"].exists():
            return self._build_latest(paths["jsonl"])
        return {"job_id": job_id, "status": "unknown", "steps_done": 0, "steps_total": 0}

    # ---------- internos ----------

    def _build_latest(self, jsonl_path: Path) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"status": "unknown", "steps_total": 0, "steps_done": 0}
        last: Dict[str, Any] = {}
        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    ev = json.loads(line)
                    last = ev
                    # Agregar pequeños agregados
                    if ev.get("type") == "job_meta":
                        meta.update(ev.get("meta", {}))
                    if ev.get("type") == "plan":
                        meta["steps_total"] = max(meta.get("steps_total", 0), len(ev.get("steps", [])))
                    if ev.get("type") == "step_end":
                        meta["steps_done"] = meta.get("steps_done", 0) + 1
                    if ev.get("status"):
                        meta["status"] = ev["status"]
        except Exception:
            pass
        latest = {
            "job_id": jsonl_path.stem,
            "updated_at": _utcnow(),
            "status": meta.get("status", "unknown"),
            "steps_total": meta.get("steps_total", 0),
            "steps_done": meta.get("steps_done", 0),
            "last_event": last,
            "meta": meta,
        }
        return latest
