# tests/test_jobs_persistence.py
"""
Fase 6 — Persistencia/Rotación/Métricas de /jobs

Este archivo es autosuficiente: define fixtures de respaldo (client, h) por si
el conftest del repo no las provee. Aísla el almacenamiento en un tmp dir
para no ensuciar el árbol del proyecto.
"""

import os
import time
import pytest
from starlette.testclient import TestClient


# ----------------------------- Fixtures de respaldo ---------------------------

@pytest.fixture(scope="session")
def _jobs_tmpdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("jobs_store")
    return str(d)


@pytest.fixture(scope="session")
def client(_jobs_tmpdir):
    # Permite requests sin auth en tests
    os.environ.setdefault("ALLOW_TEST_NO_AUTH", "1")
    # Aísla persistencia en un dir temporal para la sesión de tests
    os.environ["JOBS_DIR"] = _jobs_tmpdir
    # Evita interferir con otros tests; límite de eventos razonable
    os.environ.setdefault("JOBS_EVENTS_MAX", "2000")

    # Importa la app después de setear envs
    try:
        from server import app  # usa el server real
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"No se pudo importar server.app: {type(e).__name__}: {e}")

    with TestClient(app) as c:
        yield c


@pytest.fixture
def h():
    """Encabezados para auth (vacío si ALLOW_TEST_NO_AUTH=1)."""
    return lambda: {}


# ----------------------------- Helpers de uso común ---------------------------

def _start_job(client, h, job_id=None, goal="phase6-demo"):
    r = client.post("/jobs/start", json={"goal": goal, "job_id": job_id}, headers=h())
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is True
    return j["job"]["job_id"]


def _checkpoint(client, h, job_id, type_, **kw):
    payload = {"job_id": job_id, "type": type_}
    payload.update(kw)
    r = client.post("/jobs/checkpoint", json=payload, headers=h())
    assert r.status_code == 200
    assert r.json()["ok"] is True


def _tail(client, h, job_id, since=0, limit=100):
    r = client.get(f"/jobs/tail?job_id={job_id}&since={since}&limit={limit}", headers=h())
    assert r.status_code == 200
    return r.json()["events"]


# --------------------------------- Tests --------------------------------------

def test_list_returns_index_of_jobs(client, h):
    j1 = _start_job(client, h, goal="list-A")
    _checkpoint(client, h, j1, "status", status="running")
    j2 = _start_job(client, h, goal="list-B")
    _checkpoint(client, h, j2, "status", status="running")

    r = client.get("/jobs/list", headers=h())
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    jobs = body.get("jobs") or []
    ids = {j.get("job_id") for j in jobs}
    assert j1 in ids and j2 in ids
    # metadatos mínimos por entrada
    for row in jobs:
        assert "status" in row and "updated_at" in row and "steps_total" in row and "steps_done" in row


def test_persist_and_rehydrate_via_load(client, h):
    job_id = _start_job(client, h, goal="persist-rehydrate")
    _checkpoint(client, h, job_id, "plan", steps=[{"id": "s1"}, {"id": "s2"}])
    _checkpoint(client, h, job_id, "step_end", step_id="s1")
    _checkpoint(client, h, job_id, "status", status="running")

    # Rehidratación forzada
    r = client.get(f"/jobs/load?job_id={job_id}", headers=h())
    assert r.status_code == 200
    snap = r.json().get("job") or {}
    assert snap.get("job_id") == job_id
    assert "events" in snap and len(snap["events"]) >= 3

    # Tail sigue funcionando y contiene tipos esperados
    events = _tail(client, h, job_id, since=0, limit=100)
    kinds = {e.get("type") for e in events}
    assert {"plan", "step_end"} <= kinds


def test_rotation_by_max_purges_old_jobs(client, h, monkeypatch):
    # La implementación lee JOBS_MAX dinámicamente
    monkeypatch.setenv("JOBS_MAX", "2")

    j_old = _start_job(client, h, goal="old")
    _checkpoint(client, h, j_old, "status", status="running")
    time.sleep(0.12)  # Windows FS ≈ 100ms de resolución

    j_mid = _start_job(client, h, goal="mid")
    _checkpoint(client, h, j_mid, "status", status="running")
    time.sleep(0.12)

    j_new = _start_job(client, h, goal="new")
    _checkpoint(client, h, j_new, "status", status="running")

    r = client.get("/jobs/list", headers=h())
    assert r.status_code == 200
    jobs = r.json().get("jobs") or []
    ids = {j.get("job_id") for j in jobs}
    assert len(ids) <= 2
    assert j_new in ids
    assert j_mid in ids
    assert j_old not in ids

    r2 = client.get(f"/jobs/tail?job_id={j_old}&n=5", headers=h())
    assert r2.status_code in (404, 410, 422)


def test_delete_job_removes_file_and_index(client, h):
    job_id = _start_job(client, h, goal="delete-me")
    _checkpoint(client, h, job_id, "status", status="running")

    r = client.delete(f"/jobs/delete?job_id={job_id}", headers=h())
    assert r.status_code in (200, 204)
    if r.status_code == 200:
        assert r.json().get("ok") is True

    r2 = client.get("/jobs/list", headers=h())
    assert r2.status_code == 200
    ids = {j.get("job_id") for j in (r2.json().get("jobs") or [])}
    assert job_id not in ids

    r3 = client.get(f"/jobs/tail?job_id={job_id}&n=5", headers=h())
    assert r3.status_code in (404, 410, 422)


def test_status_exposes_persistence_metrics(client, h):
    r = client.get("/status", headers=h())
    assert r.status_code == 200
    s = r.json()
    for k in ("jobs_dir", "persisted_jobs", "purged_jobs_total", "jobs_disk_usage_bytes"):
        assert k in s
