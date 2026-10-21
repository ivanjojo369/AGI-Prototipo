# agents/project_runner.py
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional

from planner.htn_simple import plan
from agents.agent_loop import run_goal


PlannerFn = Callable[[str], Any]
RunnerFn = Callable[..., Any]


def _normalize_subgoals(
    plan_result: Any,
) -> List[Dict[str, str]]:
    """
    Corrige el contrato histórico:
    htn_simple.plan() devuelve {"subgoals": [...]}, no una lista directa.
    """
    if isinstance(plan_result, dict):
        raw = plan_result.get(
            "subgoals"
        ) or []
    elif isinstance(plan_result, list):
        raw = plan_result
    else:
        raw = []

    clean: List[Dict[str, str]] = []

    for item in raw:
        if isinstance(item, dict):
            desc = str(
                item.get(
                    "desc"
                )
                or ""
            ).strip()
            done_when = str(
                item.get(
                    "done_when"
                )
                or "sub-tarea ejecutada"
            ).strip()
        else:
            desc = str(item).strip()
            done_when = "sub-tarea ejecutada"

        if desc:
            clean.append(
                {
                    "desc": desc,
                    "done_when": done_when,
                }
            )

    return clean


def run_project(
    goal: str,
    max_steps_per_subgoal: int = 6,
    reflect_tokens: int = 60,
    project_timebox_sec: int = 300,
    *,
    planner_fn: PlannerFn = plan,
    runner_fn: RunnerFn = run_goal,
    event_sink: Optional[
        Callable[[Dict[str, Any]], None]
    ] = None,
) -> Dict[str, Any]:
    """
    Ejecuta una meta multi-etapa.

    planner_fn y runner_fn son inyectables para que Orchestrator 3.0 pueda
    usar el modelo/adaptador actual de Quipu en lugar del endpoint legacy.
    """
    t0 = time.time()

    plan_result = planner_fn(
        goal
    )
    subgoals = _normalize_subgoals(
        plan_result
    )

    results: List[
        Dict[str, Any]
    ] = []

    for index, subgoal in enumerate(
        subgoals,
        1,
    ):
        if (
            time.time() - t0
            > project_timebox_sec
        ):
            item = {
                "step": index,
                "desc": subgoal["desc"],
                "status":
                    "skipped_timebox",
            }
            results.append(item)

            if event_sink:
                event_sink(item)

            break

        desc = subgoal["desc"]

        try:
            output = runner_fn(
                desc,
                max_steps=
                    max_steps_per_subgoal,
                reflect_tokens=
                    reflect_tokens,
                timebox_sec=min(
                    90,
                    project_timebox_sec,
                ),
            )
            status = "done"
        except Exception as exc:
            output = (
                f"error: {exc}"
            )
            status = "failed"

        item = {
            "step": index,
            "desc": desc,
            "done_when":
                subgoal.get(
                    "done_when"
                ),
            "output": output,
            "status": status,
        }

        results.append(item)

        if event_sink:
            event_sink(item)

        if status == "failed":
            break

    return {
        "goal": goal,
        "subgoals": subgoals,
        "results": results,
        "elapsed_s": round(
            time.time() - t0,
            2,
        ),
    }


if __name__ == "__main__":
    import sys

    goal = (
        "1) dime la hora; "
        "2) calcula (45+55)/2; "
        "3) guarda un fact; "
        "4) recupéralo."
    )

    if len(sys.argv) > 1:
        goal = " ".join(
            sys.argv[1:]
        )

    print(
        json.dumps(
            run_project(goal),
            ensure_ascii=False,
            indent=2,
        )
    )
