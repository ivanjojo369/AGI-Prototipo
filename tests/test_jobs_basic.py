# tests/test_jobs_basic.py
import os
from fastapi.testclient import TestClient

# Asegura que API_KEY esté en el entorno al correr los tests
os.environ.setdefault("API_KEY", "test-key")

from server import app  # importa tu app principal (server.py)
client = TestClient(app)

def _h():
    return {"x-api-key": os.getenv("API_KEY")}

def test_start_and_status():
    r = client.post("/jobs/start", json={"goal": "demo job", "autostart": True}, headers=_h())
    assert r.status_code == 200, r.text
    job = r.json()["job"]
    job_id = job["job_id"]

    s = client.get(f"/jobs/status?job_id={job_id}", headers=_h())
    assert s.status_code == 200
    payload = s.json()
    assert payload["ok"] is True
    assert payload["status"]["job_id"] == job_id

def test_checkpoint_flow():
    r = client.post("/jobs/start", json={"goal": "checkpoint flow"}, headers=_h())
    job_id = r.json()["job"]["job_id"]

    c1 = client.post("/jobs/checkpoint", json={
        "job_id": job_id, "type": "step_start", "step_id": "s1", "status": "running", "analysis": "start"
    }, headers=_h())
    assert c1.status_code == 200

    c2 = client.post("/jobs/checkpoint", json={
        "job_id": job_id, "type": "step_end", "step_id": "s1", "status": "completed", "result": {"ok": True}
    }, headers=_h())
    assert c2.status_code == 200

    tail = client.get(f"/jobs/tail?job_id={job_id}&n=5", headers=_h())
    assert tail.status_code == 200
    evts = tail.json()["events"]
    assert any(e.get("type") == "step_end" for e in evts)
