# jobs_ttl.py
"""
Sweep TTL de jobs persistidos.
- Desacoplado del servidor (útil para tests y cron).
- Política: solo estados terminales; referencia ended_at o created_at.
- Var env (para integración futura): JOBS_TTL_SECONDS (0 deshabilita).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


TERMINAL_STATES = {"completed", "failed", "canceled"}
RUNNING_STATES = {"running", "queued", "pending"}  # no se purgan


@dataclass
class PurgeReport:
    scanned: int = 0
    expired: int = 0
    purged: int = 0
    errors: int = 0
    skipped_running: int = 0
    skipped_missing_meta: int = 0
    purged_job_ids: list[str] = field(default_factory=list)
    last_run_epoch: float = 0.0

    def to_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "expired": self.expired,
            "purged": self.purged,
            "errors": self.errors,
            "skipped_running": self.skipped_running,
            "skipped_missing_meta": self.skipped_missing_meta,
            "purged_job_ids": list(self.purged_job_ids),
            "last_run_epoch": self.last_run_epoch,
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso8601(s: str) -> Optional[datetime]:
    if not s:
        return None
    # Soportar formato con 'Z'
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _load_job_meta(job_dir: Path) -> Optional[dict]:
    meta_file = job_dir / "job.json"
    if not meta_file.exists():
        return None
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def _job_is_running(meta: dict) -> bool:
    status = str(meta.get("status", "")).lower()
    return status in RUNNING_STATES


def _job_is_terminal(meta: dict) -> bool:
    status = str(meta.get("status", "")).lower()
    return status in TERMINAL_STATES


def _job_expired(meta: dict, now: datetime, ttl_seconds: int) -> bool:
    if ttl_seconds <= 0:
        return False  # TTL deshabilitado
    if _job_is_running(meta):
        return False  # nunca purgar activos
    # Solo consideramos terminales; si no lo es, no expira
    if not _job_is_terminal(meta):
        return False
    ended_at = _parse_iso8601(meta.get("ended_at", "") or "")
    created_at = _parse_iso8601(meta.get("created_at", "") or "")
    ref = ended_at or created_at
    if ref is None:
        return False
    # ref debe ser timezone-aware; si no lo es, asumimos UTC
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    age = (now - ref).total_seconds()
    return age >= ttl_seconds


def purge_expired_jobs(
    jobs_dir: str | Path,
    ttl_seconds: int,
    now: Optional[datetime] = None,
    *,
    dry_run: bool = False,
    purge_fn: Optional[callable] = None,
) -> PurgeReport:
    """
    Escanea subdirectorios en jobs_dir, identifica expirados y los purga.

    - ttl_seconds: 0 o menor => no hace nada.
    - dry_run: true => solo calcula, no borra.
    - purge_fn: si se provee (p.ej. tu función purge_job(job_id)), se usa en lugar de rmtree.
    """
    jobs_root = Path(jobs_dir)
    report = PurgeReport()
    now = now or _utcnow()
    report.last_run_epoch = now.timestamp()

    if ttl_seconds <= 0:
        # Sweep "vacío": aún computamos scanned/skip para consistencia si se quiere
        for entry in jobs_root.iterdir():
            if entry.is_dir():
                report.scanned += 1
        return report

    if not jobs_root.exists() or not jobs_root.is_dir():
        return report

    for entry in jobs_root.iterdir():
        if not entry.is_dir():
            continue
        report.scanned += 1

        meta = _load_job_meta(entry)
        if meta is None:
            report.skipped_missing_meta += 1
            continue

        if _job_is_running(meta):
            report.skipped_running += 1
            continue

        try:
            if _job_expired(meta, now, ttl_seconds):
                report.expired += 1
                job_id = str(meta.get("job_id") or entry.name)
                if not dry_run:
                    if purge_fn is not None:
                        # Deja que la integración haga el borrado (logs, índices, etc.)
                        purge_fn(job_id)
                    else:
                        shutil.rmtree(entry, ignore_errors=False)
                report.purged += 1
                report.purged_job_ids.append(job_id)
        except Exception:
            report.errors += 1

    return report


# CLI opcional para pruebas manuales:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Purge expired jobs by TTL")
    parser.add_argument("--jobs-dir", required=True, help="Directorio raíz de jobs")
    parser.add_argument("--ttl-seconds", type=int, default=int(os.getenv("JOBS_TTL_SECONDS", "0")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rep = purge_expired_jobs(args.jobs_dir, ttl_seconds=args.ttl_seconds, dry_run=args.dry_run)
    print(json.dumps(rep.to_dict(), indent=2))
