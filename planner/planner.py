"""Planificador de Quipu Core 4.0 - adaptador de produccion.

Contrato del nucleo:
    Planner().plan(goal, llm_call=funcion) -> {"subgoals": [...]}

La unica autoridad para descomponer metas es planner.htn_simple.plan.
No se descubren backends legacy ni se realizan llamadas HTTP implicitas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .htn_simple import plan as htn_plan


PlannerLLM = Callable[[List[Dict[str, str]], int, float], str]


def _without_llm(
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    """Solicita el fallback local de htn_simple sin consultar APIs legacy."""
    return ""


@dataclass
class PlannerConfig:
    max_depth: int = 3       # Compatibilidad con llamadores existentes.
    max_steps: int = 5       # htn_simple admite como maximo cinco subobjetivos.
    temperature: float = 0.0


class Planner:
    """Fachada estable de Core 4.0 para un unico backend HTN."""

    VERSION = "3.1.1"

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        *,
        llm_call: Optional[PlannerLLM] = None,
    ) -> None:
        if llm_call is not None and not callable(llm_call):
            raise TypeError("llm_call debe ser una funcion o None")
        self.config = config or PlannerConfig()
        self.llm_call = llm_call
        self.backend_name = "htn_simple.plan"

    def plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        llm_call: Optional[PlannerLLM] = None,
        **_: Any,
    ) -> Dict[str, List[Dict[str, str]]]:
        # Se conserva context por compatibilidad. El backend no lo utiliza aun.
        del context
        if llm_call is not None and not callable(llm_call):
            raise TypeError("llm_call debe ser una funcion o None")
        active_llm = llm_call or self.llm_call or _without_llm
        max_subgoals = max(1, min(5, int(self.config.max_steps)))

        result = htn_plan(
            goal=str(goal or ""),
            max_subgoals=max_subgoals,
            llm_call=active_llm,
        )
        if not isinstance(result, dict) or not isinstance(result.get("subgoals"), list):
            raise RuntimeError("htn_simple.plan devolvio un formato inesperado")
        return result


def planificar_tarea(
    objetivo: str,
    contexto: Optional[Dict[str, Any]] = None,
    **kw: Any,
) -> Dict[str, List[Dict[str, str]]]:
    """API historica en espanol conservada para compatibilidad."""
    return Planner().plan(goal=objetivo, context=contexto, **kw)


__all__ = ["Planner", "PlannerConfig", "planificar_tarea"]
