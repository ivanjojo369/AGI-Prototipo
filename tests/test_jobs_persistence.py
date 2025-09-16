# tests/test_jobs_persistence.py
import os
import time
import pytest

# Nota: estos tests asumen que ya existen los endpoints /jobs/start, /jobs/checkpoint, /jobs/tail
# y añaden la especificación para nuevos endpoints Fase 6:
#   - GET /jobs/list
#   - GET /jobs/load?job_id=...
#   - DELETE /jobs/delete?job_id=...
#
# Todos los tests están marcados xfail hasta que implementes Fase 6.


def _h_or_default(h):
    """
    Usa el fixture h() si existe (como en tus otros tests),
    de lo contrario, devuelve una callable que regresa {} (sin headers).
    """
    if callable(h):
        return h
    return lambda: {}


def _start_job(client, h, job_id=None, goal="phase6-persistence-demo"):
    r = client.post("/jobs/start", json={"goal": goal, "job_id": job_id})
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


@pytest.mark.xfail(reason="Fase 6: /jobs/list aún no implementado", strict=False)
def test_list_returns_index_of_jobs(client, h):
    h = _h_or_default(h)
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
    # metadatos mínimos esperados por entrada
    for row in jobs:
        assert "status" in row and "updated_at" in row and "steps_total" in row and "steps_done" in row


@pytest.mark.xfail(reason="Fase 6: /jobs/load + persistencia aún no implementados", strict=False)
def test_persist_and_rehydrate_via_load(client, h):
    h = _h_or_default(h)
    job_id = _start_job(client, h, goal="persist-rehydrate")
    _checkpoint(client, h, job_id, "plan", steps=[{"id": "s1"}, {"id": "s2"}])
    _checkpoint(client, h, job_id, "step_end", step_id="s1")
    _checkpoint(client, h, job_id, "status", status="running")

    # Fuerza rehidratación (carga desde disco a memoria y regresa snapshot)
    r = client.get(f"/jobs/load?job_id={job_id}", headers=h())
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    snap = body.get("job") or {}
    assert snap.get("job_id") == job_id
    assert "events" in snap and len(snap["events"]) >= 3

    # Compat: tail funciona después de load
    events = _tail(client, h, job_id, since=0, limit=100)
    assert any(e.get("type") == "plan" for e in events)
    assert any(e.get("type") == "step_end" for e in events)


@pytest.mark.xfail(reason="Fase 6: rotación por JOBS_MAX no implementada", strict=False)
def test_rotation_by_max_purges_old_jobs(client, h, monkeypatch):
    h = _h_or_default(h)
    # Configura un límite pequeño de jobs p/rotación
    monkeypatch.setenv("JOBS_MAX", "2")

    j_old = _start_job(client, h, goal="old")
    _checkpoint(client, h, j_old, "status", status="running")
    time.sleep(0.01)

    j_mid = _start_job(client, h, goal="mid")
    _checkpoint(client, h, j_mid, "status", status="running")
    time.sleep(0.01)

    j_new = _start_job(client, h, goal="new")
    _checkpoint(client, h, j_new, "status", status="running")

    # Esperado: al listar, solo 2 (mid, new). El old fue purgado.
    r = client.get("/jobs/list", headers=h())
    assert r.status_code == 200
    jobs = r.json().get("jobs") or []
    ids = {j.get("job_id") for j in jobs}
    assert len(ids) <= 2
    assert j_new in ids
    assert j_mid in ids
    assert j_old not in ids

    # Tail sobre purgado debe fallar con 404
    r2 = client.get(f"/jobs/tail?job_id={j_old}&n=5", headers=h())
    assert r2.status_code in (404, 410)


@pytest.mark.xfail(reason="Fase 6: /jobs/delete no implementado", strict=False)
def test_delete_job_removes_file_and_index(client, h):
    h = _h_or_default(h)
    job_id = _start_job(client, h, goal="delete-me")
    _checkpoint(client, h, job_id, "status", status="running")

    r = client.delete(f"/jobs/delete?job_id={job_id}", headers=h())
    assert r.status_code in (200, 204)
    if r.status_code == 200:
        assert r.json().get("ok") is True

    # No debe aparecer en /jobs/list
    r2 = client.get("/jobs/list", headers=h())
    assert r2.status_code == 200
    ids = {j.get("job_id") for j in (r2.json().get("jobs") or [])}
    assert job_id not in ids

    # Tail debe fallar
    r3 = client.get(f"/jobs/tail?job_id={job_id}&n=5", headers=h())
    assert r3.status_code in (404, 410)


@pytest.mark.xfail(reason="Fase 6: métricas de persistencia aún no expuestas en /status", strict=False)
def test_status_exposes_persistence_metrics(client, h):
    h = _h_or_default(h)
    r = client.get("/status", headers=h())
    assert r.status_code == 200
    s = r.json()
    # Nuevas métricas esperadas
    assert "jobs_dir" in s
    assert "persisted_jobs" in s
    assert "purged_jobs_total" in s
    assert "jobs_disk_usage_bytes" in s
