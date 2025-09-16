# tests/test_jobs_resume.py
import os
import pytest
from fastapi.testclient import TestClient

# importa la app principal
from server import app

# ---- Fixtures locales (independientes de conftest) ----
@pytest.fixture(scope="module")
def client():
    return TestClient(app)

@pytest.fixture
def h():
    key = os.getenv("API_KEY", "dev-key-123")
    def _h(extra=None):
        headers = {
            "x-api-key": key,
            "authorization": f"Bearer {key}",
            "content-type": "application/json",
        }
        if extra:
            headers.update(extra)
        return headers
    return _h

# ---- Helpers ----
def _start_demo_job(client, h):
    r = client.post("/jobs/start", json={"goal": "demo", "autostart": True}, headers=h())
    assert r.status_code == 200
    job_id = r.json()["job"]["job_id"]

    r1 = client.post("/plan/solve", json={"goal": "demo", "context": {}}, headers=h())
    assert r1.status_code == 200
    plan = r1.json()["plan"]

    r2 = client.post("/plan/execute", json={"plan": plan, "context": {}, "job_id": job_id}, headers=h())
    assert r2.status_code == 200
    return job_id

# ---- Tests ----
def test_dump_contains_events(client, h):
    job_id = _start_demo_job(client, h)

    s = client.get(f"/jobs/status?job_id={job_id}", headers=h())
    assert s.status_code == 200
    assert s.json()["ok"] is True

    d = client.get(f"/jobs/dump?job_id={job_id}", headers=h())
    assert d.status_code == 200
    snap = d.json()["job"]
    assert snap["job_id"] == job_id
    assert isinstance(snap["events"], list)
    assert len(snap["events"]) >= 1

def test_resume_enqueues_cmd_event(client, h):
    job_id = _start_demo_job(client, h)

    r = client.post("/jobs/resume", json={"job_id": job_id, "from_step": "s2"}, headers=h())
    assert r.status_code == 200
    assert r.json()["ok"] is True

    t = client.get(f"/jobs/tail?job_id={job_id}&n=50", headers=h())
    assert t.status_code == 200
    events = t.json()["events"]
    assert any(e.get("type") == "cmd" and e.get("cmd") == "resume" for e in events)

def test_replay_enqueues_cmd_event(client, h):
    job_id = _start_demo_job(client, h)

    r = client.post("/jobs/replay", json={"job_id": job_id, "from_event_idx": 0}, headers=h())
    assert r.status_code == 200
    assert r.json()["ok"] is True

    t = client.get(f"/jobs/tail?job_id={job_id}&n=50", headers=h())
    assert t.status_code == 200
    events = t.json()["events"]
    assert any(e.get("type") == "cmd" and e.get("cmd") == "replay" for e in events)
