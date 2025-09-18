# tests/test_jobs_ttl.py

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import jobs_ttl  # módulo del patch inicial


def iso(ts: datetime) -> str:
    # ISO-8601 con 'Z'
    return ts.astimezone(timezone.utc).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def mk_job(dir_root: Path, job_id: str, *, status: str,
           created_delta_sec: int, ended_delta_sec: int | None) -> Path:
    """
    Crea un job de prueba con job.json consistente.
    created_delta_sec y ended_delta_sec son offsets negativos respecto a 'now' (en segundos).
    """
    now = datetime.now(timezone.utc)
    job_dir = dir_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    created_at = now + timedelta(seconds=created_delta_sec)
    meta = {
        "job_id": job_id,
        "status": status,
        "created_at": iso(created_at),
    }
    if ended_delta_sec is not None:
        meta["ended_at"] = iso(now + timedelta(seconds=ended_delta_sec))

    (job_dir / "job.json").write_text(json.dumps(meta), encoding="utf-8")
    # Un archivo dummy para asegurar contenido
    (job_dir / "stdout.log").write_text(f"{job_id} output\n", encoding="utf-8")
    return job_dir


@pytest.fixture
def tmp_jobs_dir(tmp_path: Path) -> Path:
    p = tmp_path / "jobs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_ttl_disabled_does_not_purge(tmp_jobs_dir: Path):
    # TTL deshabilitado => no purga aunque sea viejo
    mk_job(tmp_jobs_dir, "old-completed",
           status="completed", created_delta_sec=-3600, ended_delta_sec=-3500)

    report = jobs_ttl.purge_expired_jobs(tmp_jobs_dir, ttl_seconds=0, now=datetime.now(timezone.utc))
    assert report.purged == 0
    assert (tmp_jobs_dir / "old-completed").exists()


def test_completed_expired_job_is_purged(tmp_jobs_dir: Path):
    # TTL = 1s, job terminó hace 10s => purga
    mk_job(tmp_jobs_dir, "j1",
           status="completed", created_delta_sec=-20, ended_delta_sec=-10)

    report = jobs_ttl.purge_expired_jobs(tmp_jobs_dir, ttl_seconds=1, now=datetime.now(timezone.utc))
    assert report.purged == 1
    assert "j1" in report.purged_job_ids
    assert not (tmp_jobs_dir / "j1").exists()


def test_running_job_is_not_purged(tmp_jobs_dir: Path):
    # Aunque sea viejo, si está "running" no debe purgarse
    mk_job(tmp_jobs_dir, "running-old",
           status="running", created_delta_sec=-7200, ended_delta_sec=None)

    report = jobs_ttl.purge_expired_jobs(tmp_jobs_dir, ttl_seconds=60, now=datetime.now(timezone.utc))
    assert report.purged == 0
    assert report.skipped_running >= 1
    assert (tmp_jobs_dir / "running-old").exists()


def test_fresh_completed_job_not_purged(tmp_jobs_dir: Path):
    # Job recién terminado: no expira aún
    mk_job(tmp_jobs_dir, "fresh",
           status="completed", created_delta_sec=-5, ended_delta_sec=-1)

    report = jobs_ttl.purge_expired_jobs(tmp_jobs_dir, ttl_seconds=60, now=datetime.now(timezone.utc))
    assert report.purged == 0
    assert (tmp_jobs_dir / "fresh").exists()


def test_missing_meta_is_skipped(tmp_jobs_dir: Path):
    # Directorio sin job.json => se salta (no rompe)
    orphan = tmp_jobs_dir / "orphan"
    orphan.mkdir()
    (orphan / "stdout.log").write_text("orphan\n", encoding="utf-8")

    report = jobs_ttl.purge_expired_jobs(tmp_jobs_dir, ttl_seconds=1, now=datetime.now(timezone.utc))
    assert report.purged == 0
    assert report.skipped_missing_meta >= 1
    assert orphan.exists()


def test_idempotent_purge_run(tmp_jobs_dir: Path):
    mk_job(tmp_jobs_dir, "to-expire",
           status="failed", created_delta_sec=-100, ended_delta_sec=-90)

    now = datetime.now(timezone.utc)
    r1 = jobs_ttl.purge_expired_jobs(tmp_jobs_dir, ttl_seconds=10, now=now)
    r2 = jobs_ttl.purge_expired_jobs(tmp_jobs_dir, ttl_seconds=10, now=now)

    assert r1.purged == 1
    assert r2.purged == 0
    assert not (tmp_jobs_dir / "to-expire").exists()
