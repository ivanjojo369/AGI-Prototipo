# tests/test_task_planner.py
import json
import os
from pathlib import Path

import pytest

from planner.task_planner import Plan, Step, TaskPlanner, BuiltinActions


# -------------------------------
# Helpers
# -------------------------------

def _mk_planner(curriculum_path: Path | str):
    """Planner con acciones por defecto."""
    return TaskPlanner(
        actions={
            "python_exec": BuiltinActions.python_exec,
            "filesystem_read": BuiltinActions.filesystem_read,
            "memory_search": BuiltinActions.memory_search,
            "search_web": BuiltinActions.search_web,
        },
        curriculum_path=curriculum_path,
    )


# -------------------------------
# Unit tests del Planner (HTN)
# -------------------------------

def test_python_exec_and_postconditions(tmp_path: Path):
    planner = _mk_planner(tmp_path / "cur.jsonl")

    plan = Plan(
        goal="Calcular y verificar suma; luego duplicar",
        steps=[
            Step(
                id="s1",
                kind="action",
                name="python_exec",
                inputs={"code": "x = 40 + 2\nresult = x"},
                postconditions=["result == 42"],
            ),
            Step(
                id="s2",
                kind="action",
                name="python_exec",
                # templating: {{results.s1}} -> 42
                inputs={"code": "result = {{results.s1}} * 2"},
                postconditions=["result == 84"],
            ),
        ],
    )

    out = planner.execute_plan(plan, context={})
    assert out["status"] == "success"
    assert out["results"]["s1"] == 42
    assert out["results"]["s2"] == 84
    assert out["curriculum_entries"] == 0


def test_filesystem_read(tmp_path: Path):
    f = tmp_path / "hello.txt"
    f.write_text("hola mundo", encoding="utf-8")

    planner = _mk_planner(tmp_path / "cur.jsonl")
    plan = Plan(
        goal="Leer archivo",
        steps=[
            Step(
                id="r1",
                name="filesystem_read",
                inputs={"path": str(f)},
                postconditions=["len(result) > 0"],
            )
        ],
    )

    out = planner.execute_plan(plan, context={})
    assert out["status"] == "success"
    assert out["results"]["r1"] == "hola mundo"


def test_retries_eventual_success_no_curriculum(tmp_path: Path):
    # Acción que falla 1 vez y luego funciona
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("fallo temporal")
        return "ok"

    actions = {
        "python_exec": BuiltinActions.python_exec,
        "filesystem_read": BuiltinActions.filesystem_read,
        "memory_search": BuiltinActions.memory_search,
        "search_web": BuiltinActions.search_web,
        "flaky": flaky,
    }

    cur = tmp_path / "cur.jsonl"
    planner = TaskPlanner(actions=actions, curriculum_path=cur)

    plan = Plan(goal="Probar retries", steps=[Step(id="a1", name="flaky", retries=1)])

    out = planner.execute_plan(plan, context={})
    assert out["status"] == "success"
    assert out["results"]["a1"] == "ok"
    # Curriculum sólo se escribe si hay fallos definitivos
    assert not cur.exists() or cur.read_text(encoding="utf-8").strip() == ""


def test_failure_writes_curriculum(tmp_path: Path):
    cur = tmp_path / "curriculum.jsonl"
    planner = _mk_planner(cur)

    plan = Plan(
        goal="Fallar por archivo inexistente",
        steps=[Step(id="f1", name="filesystem_read", inputs={"path": str(tmp_path / "nope.txt")})],
    )

    out = planner.execute_plan(plan, context={})
    assert out["status"] in ("failed", "partial")

    # Debe existir al menos una línea JSONL válida
    content = cur.read_text(encoding="utf-8").strip()
    assert content
    for line in content.splitlines():
        obj = json.loads(line)
        assert obj["step_id"] == "f1"
        assert obj["goal"] == "Fallar por archivo inexistente"
        assert obj["error_type"] in ("FileNotFoundError",)


def test_htn_task_handler_expansion(tmp_path: Path):
    """Comprueba que un task handler genere sub-steps y use templating."""
    planner = _mk_planner(tmp_path / "cur.jsonl")

    def handler(step: Step, state: dict):
        return [
            Step(
                id="a1",
                kind="action",
                name="python_exec",
                inputs={"code": "result = 2"},
                postconditions=["result == 2"],
            ),
            Step(
                id="a2",
                kind="action",
                name="python_exec",
                inputs={"code": "result = {{results.a1}} + 40"},
                postconditions=["result == 42"],
            ),
        ]

    planner.task_handlers["double_then_add"] = handler

    plan = Plan(goal="HTN handler", steps=[Step(id="T", kind="task", name="double_then_add")])

    out = planner.execute_plan(plan, context={})
    assert out["status"] == "success"
    assert out["results"]["a1"] == 2
    assert out["results"]["a2"] == 42


# -------------------------------
# Smoke test del endpoint /plan/execute
# -------------------------------

def test_server_plan_execute_smoke(tmp_path: Path):
    """
    Usa TestClient si 'server' se puede importar.
    Se salta automáticamente si el módulo no está disponible.
    """
    server = pytest.importorskip("server")  # evita fallar si no existe
    from fastapi.testclient import TestClient

    client = TestClient(server.app)
    headers = {}
    api_key = os.getenv("API_KEY")
    if api_key:
        headers["x-api-key"] = api_key  # respeta tu middleware

    # Plan mínimo
    plan = {
        "goal": "Calcular 42",
        "steps": [
            {
                "id": "s1",
                "kind": "action",
                "name": "python_exec",
                "inputs": {"code": "result = 40 + 2"},
                "postconditions": ["result == 42"],
            }
        ],
        "metadata": {"test": True},
    }

    resp = client.post("/plan/execute", json={"plan": plan, "context": {}}, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["results"]["s1"] == 42
