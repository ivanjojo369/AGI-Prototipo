# tests/test_metrics_prometheus.py
import os
from fastapi.testclient import TestClient

def test_metrics_prometheus(monkeypatch):
    monkeypatch.setenv("ALLOW_TEST_NO_AUTH", "1")
    monkeypatch.setenv("JOBS_TTL_SECONDS", "60")

    import importlib
    server = importlib.import_module("server")

    with TestClient(server.app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers.get("content-type", "")

        body = r.text
        assert "jobs_ttl_seconds" in body
        assert "persisted_jobs" in body
        assert "ttl_purger_runs_total" in body  # puede ser 0 si no corrió aún
