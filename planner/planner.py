from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Any, Dict, List, Optional, Callable

# Intentamos encontrar un backend de planificación existente en tu repo
_BACKEND_FN: Optional[Callable[..., Any]] = None
_BACKEND_NAME = None

# 1) task_planner.plan (o plan_task) — preferente si existe
try:
    from .task_planner import plan as _tp_plan  # type: ignore
    _BACKEND_FN = _tp_plan
    _BACKEND_NAME = "task_planner.plan"
except Exception:
    try:
        from .task_planner import plan_task as _tp_plan_task  # type: ignore
        _BACKEND_FN = _tp_plan_task
        _BACKEND_NAME = "task_planner.plan_task"
    except Exception:
        pass

# 2) htn_simple.plan — alternativa
if _BACKEND_FN is None:
    try:
        from .htn_simple import plan as _htn_plan  # type: ignore
        _BACKEND_FN = _htn_plan
        _BACKEND_NAME = "htn_simple.plan"
    except Exception:
        pass


def _call_backend(fn: Callable[..., Any], **kwargs) -> Any:
    """Llama al backend filtrando kwargs desconocidos para evitar TypeError."""
    params = {k for k in signature(fn).parameters.keys()}
    safe_kwargs = {k: v for k, v in kwargs.items() if k in params}
    return fn(**safe_kwargs)


@dataclass
class PlannerConfig:
    max_depth: int = 3
    max_steps: int = 20
    temperature: float = 0.0


class Planner:
    """
    Shim unificado. Expone .plan() y delega al backend encontrado.
    Evita romper importaciones en tests y en agents/meta_agent.py.
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        if _BACKEND_FN is None:
            raise ImportError(
                "No se encontró backend de planificación. "
                "Asegúrate de tener task_planner.py o htn_simple.py con una función 'plan' o 'plan_task'."
            )
        self.config = config or PlannerConfig()
        self._backend = _BACKEND_FN
        self.backend_name = _BACKEND_NAME

    def plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        **kw: Any,
    ) -> List[Dict[str, Any]]:
        """
        Devuelve una lista de pasos/acciones (estructura la define el backend).
        """
        if self._backend is None:
            raise RuntimeError("Backend no inicializado")
        context = context or {}
        return _call_backend(
            self._backend,
            goal=goal,
            context=context,
            max_depth=self.config.max_depth,
            max_steps=self.config.max_steps,
            temperature=self.config.temperature,
            **kw,
        )


# API en español que esperan los tests y meta_agent
def planificar_tarea(objetivo: str, contexto: Optional[Dict[str, Any]] = None, **kw: Any):
    """
    Atajo a Planner().plan() para mantener compatibilidad con importaciones existentes.
    """
    return Planner().plan(goal=objetivo, context=contexto or {}, **kw)


__all__ = ["Planner", "PlannerConfig", "planificar_tarea"]
