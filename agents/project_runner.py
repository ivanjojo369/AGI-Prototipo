# agents/project_runner.py
from __future__ import annotations
import time, json
from typing import List, Dict, Any
from planner.htn_simple import plan
from agents.agent_loop import run_goal

def run_project(goal: str, max_steps_per_subgoal: int = 6,
                reflect_tokens: int = 60, project_timebox_sec: int = 300) -> Dict[str,Any]:
    t0 = time.time()
    subgoals = plan(goal)
    results: List[Dict[str,Any]] = []
    for i, sg in enumerate(subgoals, 1):
        if time.time() - t0 > project_timebox_sec:
            results.append({"step": i, "desc": sg["desc"], "status":"skipped_timebox"})
            break
        desc = sg["desc"]
        res  = run_goal(desc, max_steps=max_steps_per_subgoal,
                        reflect_tokens=reflect_tokens, timebox_sec=min(90, project_timebox_sec))
        results.append({"step": i, "desc": desc, "output": res, "status":"done"})
    return {
        "goal": goal,
        "subgoals": subgoals,
        "results": results,
        "elapsed_s": round(time.time() - t0, 2)
    }

if __name__ == "__main__":
    import sys
    g = ("1) dime la hora; 2) calcula (45+55)/2; "
         "3) guarda el fact 'prefiere respuestas concisas' con tag 'pref'; "
         "4) busca en memoria 'concisas' y devuelve JSON {now, calc, mem_hits}.")
    if len(sys.argv) > 1: g = " ".join(sys.argv[1:])
    out = run_project(g)
    print(json.dumps(out, ensure_ascii=False, indent=2))
