# planner/task_planner.py — HTN MVP (Pydantic v2)
from __future__ import annotations

import ast
import json
import os
import time
from datetime import datetime
from pathlib import Path
from collections.abc import Sequence  # para isinstance con listas/tuplas
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator


__all__ = ["Step", "Plan", "BuiltinActions", "TaskPlanner"]


# =============================================================================
# Modelos (Step / Plan)
# =============================================================================

class Step(BaseModel):
    """
    kind='action'  → ejecuta una acción atómica (python_exec, filesystem_read, …)
    kind='task'    → se expande a sub-steps en runtime (HTN).
    """
    id: str = Field(..., description="Identificador único del step")
    kind: Literal["action", "task"] = "action"
    name: str = Field(..., description="Nombre de la acción o tarea")
    inputs: Dict[str, Any] = Field(default_factory=dict)
    preconditions: List[str] = Field(default_factory=list)    # expresiones sobre 'state'
    postconditions: List[str] = Field(default_factory=list)   # expresiones sobre 'state' o 'result'
    retries: int = Field(0, ge=0, le=5)
    continue_on_error: bool = Field(False)

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Step.id no puede ser vacío")
        return v


class Plan(BaseModel):
    goal: str = Field(..., description="Objetivo textual del plan")
    steps: List[Step] = Field(..., description="Secuencia lineal de steps")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_step_ids(self) -> "Plan":
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Los Step.id deben ser únicos")
        return self

    def __len__(self) -> int:
        return len(self.steps)


# =============================================================================
# Utilidades: templating y evaluación segura
# =============================================================================

def _flatten_scope(scope: Dict[str, Any], prefix: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if out is None:
        out = {}
    for k, v in scope.items():
        key = f"{prefix}.{k}" if prefix else k
        out[key] = v
        if isinstance(v, dict):
            _flatten_scope(v, key, out)
    return out


def _render_template(value: Any, scope: Dict[str, Any]) -> Any:
    """
    Reemplaza {{var}} (incluye notación tipo 'results.s1') en strings.
    Aplica recursivamente a dicts/listas.
    """
    if isinstance(value, str):
        if "{{" in value and "}}" in value:
            flat = _flatten_scope(scope)
            out = value
            for k, v in flat.items():
                out = out.replace("{{" + k + "}}", str(v))
            return out
        return value
    if isinstance(value, dict):
        return {k: _render_template(v, scope) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_template(v, scope) for v in value]
    return value


_ALLOWED_EVAL_BUILTINS = {
    "len": len, "min": min, "max": max, "sum": sum, "any": any, "all": all,
    "sorted": sorted, "abs": abs
}

def _safe_eval_bool(expr: str, state: Dict[str, Any]) -> bool:
    """
    Evalúa expresiones booleanas para pre/postconditions de forma acotada.
    No permite imports, atributos, ni lambdas.
    """
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute, ast.Lambda)):
            raise ValueError("Expresión no permitida en pre/postcondition")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Nombre no permitido")
    env = {"__builtins__": _ALLOWED_EVAL_BUILTINS}
    locals_env = {"state": state}
    for k, v in state.items():
        if isinstance(k, str) and k.isidentifier():
            locals_env[k] = v
    code = compile(tree, "<pre/postcondition>", "eval")
    return bool(eval(code, env, locals_env))


# =============================================================================
# Acciones builtin (inyectables)
# =============================================================================

class BuiltinActions:
    @staticmethod
    def python_exec(**kwargs) -> Any:
        code: str = kwargs.get("code", "")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("python_exec requiere 'code' no vacío")

        tree = ast.parse(code, mode="exec")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.Try, ast.Raise)):
                raise ValueError("Construcción no permitida en python_exec (import/with/try/raise)")
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                raise ValueError("No se permiten global/nonlocal en python_exec")
            if isinstance(node, ast.Attribute):
                raise ValueError("Atributos no permitidos en python_exec")

        safe_builtins = {"range": range, "len": len, "min": min, "max": max, "sum": sum, "abs": abs}
        exec_globals = {"__builtins__": safe_builtins}
        user_globals = kwargs.get("globals") or {}
        user_locals = kwargs.get("locals") or {}
        exec_globals.update(user_globals)

        if len(code) > 5000:
            raise ValueError("Código demasiado largo (límite 5000 chars)")

        exec(compile(tree, "<python_exec>", "exec"), exec_globals, user_locals)
        return user_locals.get("result", None)

    @staticmethod
    def filesystem_read(**kwargs) -> str:
        path = kwargs.get("path")
        if not path:
            raise ValueError("filesystem_read requiere 'path'")
        max_bytes = int(kwargs.get("max_bytes", 2_000_000))
        encoding = kwargs.get("encoding", "utf-8")
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No existe el archivo: {p}")
        if p.is_dir():
            raise IsADirectoryError(f"Es un directorio: {p}")
        if p.stat().st_size > max_bytes:
            raise ValueError("Archivo demasiado grande")
        return p.read_text(encoding=encoding)

    @staticmethod
    def memory_search(**kwargs) -> Any:
        # Debe inyectarse desde el server.
        raise NotImplementedError("memory_search no implementado (inyectar handler real).")

    @staticmethod
    def search_web(**kwargs) -> Any:
        # Debe inyectarse desde el server.
        raise NotImplementedError("search_web no implementado (inyectar conector real).")


# =============================================================================
# TaskPlanner (HTN)
# =============================================================================

class TaskPlanner:
    def __init__(
        self,
        actions: Optional[Dict[str, Callable[..., Any]]] = None,
        task_handlers: Optional[Dict[str, Callable[[Step, Dict[str, Any]], List[Step]]]] = None,
        curriculum_path: Union[str, Path] = "data/curriculum/planner_curriculum.jsonl",
        conf_scale: Optional[float] = None,
    ):
        self.actions = actions or {
            "python_exec": BuiltinActions.python_exec,
            "filesystem_read": BuiltinActions.filesystem_read,
            "memory_search": BuiltinActions.memory_search,
            "memory_vector": BuiltinActions.memory_search,  # alias requerido por tests
            "search_web": BuiltinActions.search_web,
        }
        self.task_handlers = task_handlers or {}
        self.curriculum_path = Path(curriculum_path)
        self.curriculum_path.parent.mkdir(parents=True, exist_ok=True)

        env_conf = os.getenv("CONF_SCALE")
        try:
            self.conf_scale = float(conf_scale if conf_scale is not None else (env_conf if env_conf else 100.0))
        except Exception:
            self.conf_scale = 100.0

    # ------------------------------------------------------------------ Ejecutar
    def execute_plan(self, plan: Plan, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ejecuta los steps del plan (lineal) con:
          - pre/postconditions, retries, continue_on_error
          - logging de curriculum (fallos)
        """
        ctx = context or {}
        state = {"context": ctx, "results": {}, "errors": []}
        errors_for_curriculum: List[Tuple[Step, Exception]] = []
        steps_queue = list(plan.steps)

        while steps_queue:
            step = steps_queue.pop(0)

            # HTN: expandir tareas
            if step.kind == "task":
                expanded = self._expand_task(step, state)
                steps_queue = list(expanded) + steps_queue
                continue

            # Preconditions
            if not self._check_conditions(step.preconditions, state):
                err = AssertionError(f"Preconditions no satisfechas en step '{step.id}' ({step.name})")
                self._record_error(state, step, err, retry_index=None)
                errors_for_curriculum.append((step, err))
                if not step.continue_on_error:
                    break
                continue

            # Ejecutar con reintentos
            attempt = 0
            last_exc: Optional[Exception] = None
            while attempt <= step.retries:
                try:
                    scope = {"state": state, "context": ctx, "results": state["results"]}
                    inputs = _render_template(step.inputs, scope)
                    action = self.actions.get(step.name)
                    if action is None:
                        raise KeyError(f"Acción desconocida: {step.name}")
                    result = action(**inputs)
                    state["results"][step.id] = result

                    # Postconditions
                    pc_state = {**state, "result": result}
                    if not self._check_conditions(step.postconditions, pc_state):
                        raise AssertionError(f"Postconditions no satisfechas en step '{step.id}' ({step.name})")

                    self._prune_step_errors(state, step.id)  # éxito
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    self._record_error(state, step, e, retry_index=attempt)
                    attempt += 1
                    if attempt > step.retries:
                        errors_for_curriculum.append((step, e))
                        if not step.continue_on_error:
                            steps_queue = []
                            break
                    else:
                        time.sleep(min(0.05 * attempt, 0.25))
            if last_exc and not step.continue_on_error:
                break

        status = "success"
        if state["errors"] and len(state["results"]) > 0:
            status = "partial"
        elif state["errors"] and len(state["results"]) == 0:
            status = "failed"

        confidence = self._compute_confidence(status, state["errors"], len(plan.steps))
        curriculum_entries = self._write_curriculum(plan, errors_for_curriculum, state)
        return {
            "status": status,
            "goal": plan.goal,
            "results": state["results"],
            "errors": state["errors"],
            "curriculum_entries": curriculum_entries,
            "confidence": confidence,
        }

    # -------------------------------------------------------------- Internals
    def _expand_task(self, step: Step, state: Dict[str, Any]) -> List[Step]:
        handler = self.task_handlers.get(step.name)
        if handler is None:
            raise KeyError(f"Tarea desconocida (sin handler): {step.name}")
        scope = {"state": state, "context": state.get("context"), "results": state["results"]}
        inputs = _render_template(step.inputs, scope)
        expanded = handler(step.copy(update={"inputs": inputs}), state)
        if not isinstance(expanded, list) or not all(isinstance(s, Step) for s in expanded):
            raise TypeError("El task handler debe retornar List[Step]")
        return expanded

    def _check_conditions(self, conditions: List[str], state: Dict[str, Any]) -> bool:
        for expr in conditions or []:
            if not _safe_eval_bool(expr, state):
                return False
        return True

    def _record_error(self, state: Dict[str, Any], step: Step, exc: Exception, retry_index: Optional[int]):
        state["errors"].append({
            "step_id": step.id,
            "name": step.name,
            "error": f"{type(exc).__name__}: {exc}",
            "retry_index": retry_index,
        })

    def _prune_step_errors(self, state: Dict[str, Any], step_id: str) -> None:
        state["errors"] = [e for e in state.get("errors", []) if e.get("step_id") != step_id]

    def _compute_confidence(self, status: str, errors: List[Dict[str, Any]], total_steps: int) -> float:
        base = self.conf_scale
        penalty = 0.0 if status == "success" else (0.25 * base if status == "partial" else 0.50 * base)
        penalty += min(len(errors), total_steps) * (0.05 * base)
        return max(0.0, round(base - penalty, 3))

    def _write_curriculum(self, plan: Plan, failures: List[Tuple[Step, Exception]], state: Dict[str, Any]) -> int:
        if not failures:
            return 0
        entries = []
        for step, exc in failures:
            entries.append({
                "ts": datetime.utcnow().isoformat() + "Z",
                "goal": plan.goal,
                "step_id": step.id,
                "step_name": step.name,
                "inputs": step.inputs,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "hint": self._hint_from_exception(exc, step),
                "preconditions": step.preconditions,
                "postconditions": step.postconditions,
                "retries": step.retries,
                "context_keys": list((state.get("context") or {}).keys()),
            })
        self.curriculum_path.parent.mkdir(parents=True, exist_ok=True)
        with self.curriculum_path.open("a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return len(entries)

    @staticmethod
    def _hint_from_exception(exc: Exception, step: Step) -> str:
        if isinstance(exc, FileNotFoundError):
            return "Verifica ruta; agrega preconditions de existencia."
        if isinstance(exc, IsADirectoryError):
            return "Indica un archivo, no un directorio."
        if isinstance(exc, AssertionError):
            return "Ajusta pre/postconditions o valida el resultado previo."
        if isinstance(exc, SyntaxError):
            return "Corrige la sintaxis de 'code' en python_exec."
        if isinstance(exc, KeyError):
            return "La acción/tarea no existe; registra el handler o corrige 'name'."
        return "Revisa inputs renderizados y dependencias."

    # -------------------------------------------------------- Crear plan
    def create_plan(self, goal: Any, context: Optional[Dict[str, Any]] = None) -> Plan:
        """
        - Si goal es una *secuencia* (lista/tupla; no str/bytes), crea un plan con esos steps.
        - Si goal es string, usa heurísticas ('buscar','leer') o el plan por defecto (3 pasos).
        """
        _Step, _Plan = Step, Plan

        # A) Secuencia de acciones/tareas (lista/tupla; no strings)
        if isinstance(goal, Sequence) and not isinstance(goal, (str, bytes, bytearray)):
            allowed = {"memory_search", "memory_vector", "python_exec", "filesystem_read", "search_web"}
            steps: List[Step] = []
            for i, raw in enumerate(goal, 1):
                name = str(raw).strip() or "search_web"
                if name not in allowed:
                    name = "search_web"
                if name == "memory_vector":
                    name = "memory_search"  # alias → canónico
                steps.append(_Step(id=f"s{i}", kind="action", name=name))
            if not steps:
                steps = [
                    _Step(id="m1", kind="action", name="memory_search", inputs={"query": " "}),
                    _Step(id="s1", kind="action", name="python_exec",
                          inputs={"code": "result = 40 + 2"}, postconditions=["result == 42"]),
                    _Step(id="r1", kind="action", name="filesystem_read",
                          inputs={"path": (context or {}).get("path", "server.py")},
                          postconditions=["len(result) > 0"]),
                ]
            return _Plan(goal=" ".join(str(x) for x in goal), steps=steps, metadata={"auto": True, "source": "seq"})

        # B) String
        g_str = goal if isinstance(goal, str) else str(goal)
        g = g_str.lower()

        if "buscar" in g:
            return _Plan(
                goal=g_str,
                steps=[_Step(id="T", kind="task", name="buscar_y_leer",
                             inputs={"query": (context or {}).get("query", g_str),
                                     "path": (context or {}).get("path", "server.py")})],
                metadata={"auto": True},
            )

        if "leer" in g or "read" in g:
            return _Plan(
                goal=g_str,
                steps=[_Step(id="r1", kind="action", name="filesystem_read",
                             inputs={"path": (context or {}).get("path", "server.py")},
                             postconditions=["len(result) > 0"])],
                metadata={"auto": True},
            )

        # C) Default: 3 pasos (mem → py → fs)
        return _Plan(
            goal=g_str,
            steps=[
                _Step(id="m1", kind="action", name="memory_search", inputs={"query": g_str, "k": 3}),
                _Step(id="s1", kind="action", name="python_exec",
                      inputs={"code": "result = 40 + 2"}, postconditions=["result == 42"]),
                _Step(id="r1", kind="action", name="filesystem_read",
                      inputs={"path": (context or {}).get("path", "server.py")},
                      postconditions=["len(result) > 0"]),
            ],
            metadata={"auto": True},
        )

    # Alias por si en algún sitio usan plan(...)
    def plan(self, goal: Any, context: Optional[Dict[str, Any]] = None) -> Plan:
        return self.create_plan(goal, context)

    # -------------------------------------------------------- Priorizar
    def prioritize_tasks(self, plan: Union[Plan, List[Step]], strategy: str = "default") -> List[Dict[str, Any]]:
        """
        Devuelve una lista de **diccionarios** con la clave 'task' (lo que
        indexan los tests como item['task']).

        Prioridad:
          memory_search/memory_vector -> python_exec -> filesystem_read -> search_web -> resto.
        Orden estable dentro de cada nivel de prioridad.
        """
        steps: List[Step] = plan.steps if isinstance(plan, Plan) else list(plan)

        order = {
            "memory_search": 0,
            "memory_vector": 0,  # alias
            "python_exec": 1,
            "filesystem_read": 2,
            "search_web": 3,
        }
        indexed = list(enumerate(steps))
        sorted_indexed = sorted(indexed, key=lambda p: (order.get(p[1].name, 99), p[0]))
        sorted_steps = [s for _, s in sorted_indexed]

        items: List[Dict[str, Any]] = []
        for s in sorted_steps:
            items.append({
                "task": s.name,                 # <- clave que usan los tests
                "id": s.id,
                "kind": s.kind,
                "inputs": s.inputs,
                "preconditions": s.preconditions,
                "postconditions": s.postconditions,
                "retries": s.retries,
                "continue_on_error": s.continue_on_error,
                "_step": s,                     # Step original por si hace falta reconstruir
            })
        return items

    # Devuelve un Plan reordenado (a partir del formato dict de arriba)
    def prioritize_plan(self, plan: Plan, strategy: str = "default") -> Plan:
        items = self.prioritize_tasks(plan, strategy=strategy)
        steps_sorted = [it.get("_step", None) for it in items if isinstance(it, dict)]
        steps_sorted = [s for s in steps_sorted if isinstance(s, Step)]
        return Plan(goal=plan.goal, steps=steps_sorted,
                    metadata={**(plan.metadata or {}), "prioritized": strategy})

    # -------------------------------------------------------- Actualizar plan
    def update_plan(
        self,
        plan_or_items: Union['Plan', List['Step'], List[Dict[str, Any]]],
        analysis: Any
    ) -> Union['Plan', List[Dict[str, Any]]]:
        """
        Actualiza el plan con el resultado de análisis:
        - Si recibe la lista de dicts de `prioritize_tasks`, añade:
            • 'analysis' con el objeto recibido
            • 'result'  (si analysis es dict con 'result', usa ese valor; en otro caso, usa analysis)
          y retorna la misma lista (mutación in-place).
          Heurística para elegir el ítem a actualizar:
            1) Si analysis trae 'task'/'name'/'skill', actualiza ese ítem.
            2) Si no, actualiza el primer ítem que aún no tenga 'result'.
            3) Si todos tienen, actualiza el primero.
        - Si recibe un Plan o una lista de Step, los retorna sin cambios.
        Además registra una entrada ligera en el curriculum (best-effort).
        """
        # Log ligero a curriculum (no debe bloquear)
        try:
            entry = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "event": "update_plan",
                "analysis": analysis if isinstance(analysis, (str, int, float, dict, list)) else str(analysis),
            }
            self.curriculum_path.parent.mkdir(parents=True, exist_ok=True)
            with self.curriculum_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # Lista de dicts (formato de prioritize_tasks)
        if isinstance(plan_or_items, list) and plan_or_items and isinstance(plan_or_items[0], dict):
            target_idx = None
            task_name = None
            if isinstance(analysis, dict):
                task_name = analysis.get("task") or analysis.get("name") or analysis.get("skill")

            if task_name:
                for i, it in enumerate(plan_or_items):
                    if it.get("task") == task_name or it.get("name") == task_name:
                        target_idx = i
                        break

            if target_idx is None:
                for i, it in enumerate(plan_or_items):
                    if "result" not in it:
                        target_idx = i
                        break

            if target_idx is None:
                target_idx = 0

            item = plan_or_items[target_idx]
            item["analysis"] = analysis
            item["result"] = (
                analysis.get("result") if isinstance(analysis, dict) and "result" in analysis else analysis
            )
            return plan_or_items

        # Plan o lista de Step → sin cambios
        return plan_or_items
