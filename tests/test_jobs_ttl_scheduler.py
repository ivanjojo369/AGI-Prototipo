# tests/test_jobs_ttl_scheduler.py
import os, time
from pathlib import Path
from fastapi.testclient import TestClient

def test_scheduler_purges_old_jobs(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    # crea un job viejo
    d = jobs_dir / "job_old"
    d.mkdir()
    # fuerza mtime al pasado
    past = time.time() - 999
    os.utime(d, (past, past))

    monkeypatch.setenv("JOBS_DIR", str(jobs_dir))
    monkeypatch.setenv("JOBS_TTL_SECONDS", "1")
    monkeypatch.setenv("JOBS_TTL_SWEEP_INTERVAL_SECONDS", "1")
    monkeypatch.setenv("ALLOW_TEST_NO_AUTH", "1")

    import importlib
    server = importlib.import_module("server")

    # al entrar al contexto se dispara startup → scheduler
    with TestClient(server.app):
        # espera a que corra 1-2 ciclos del scheduler
        time.sleep(2.5)

        # el job ya no debe existir
        assert not d.exists(), "El scheduler debió purgar el job viejo"

        # el status refleja corridas del purger
        r = TestClient(server.app).get("/status")
        assert r.status_code == 200
        data = r.json()
        assert data.get("vector_status") is not None
