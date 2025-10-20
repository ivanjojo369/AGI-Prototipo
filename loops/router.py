# -*- coding: utf-8 -*-
"""
Router simple: decide qué hacer con la entrada (nota/memoria, búsqueda, o pregunta)
y llama al loop del agente.

Formas aceptadas:
- "note: <texto>" | "nota: <texto>" | "recordar: <texto>"  -> memory.write()
- "mem?: <query>" | "search mem: <query>"                  -> memory.search()
- "reindex" | "prune"                                      -> memory.reindex/prune (si existen)
- por defecto                                               -> agent_loop.agent_turn()
"""
from __future__ import annotations
import re
from typing import Any, Dict, Optional

# Dependencias con fallback
try:
    from memory.memory import write as memory_write, search as memory_search, reindex as memory_reindex, prune as memory_prune
except Exception:  # pragma: no cover
    def memory_write(text: str, **meta): return {"ok": True, "id": "mem-dummy", "meta": meta}
    def memory_search(q: str, top_k: int = 5, **kw): return []
    def memory_reindex(project_id: Optional[str] = None): return {"ok": True, "reindexed": 0}
    def memory_prune(**kw): return {"ok": True}

try:
    from loops.agent_loop import agent_turn, AgentConfig
except Exception:  # pragma: no cover
    def agent_turn(query: str, **kw): return {"ok": True, "output": f"(stub) {query}", "stats": {}, "context": {}, "verified": {"ok": True}}
    class AgentConfig:  # type: ignore
        def __init__(self, **kw): pass

NOTE_PAT = re.compile(r"^(?:note|nota|recordar)\s*:\s*(.+)$", re.IGNORECASE)
MEMQ_PAT = re.compile(r"^(?:mem\?\s*|search\s*mem\s*:)\s*(.+)$", re.IGNORECASE)

def route_and_run(
    text: str,
    user: Optional[str] = None,
    project_id: Optional[str] = None,
    min_score: float = 0.45,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Decide acción y la ejecuta.
    """
    t = text.strip()

    # 1) Nota / escribir memoria
    m = NOTE_PAT.match(t)
    if m:
        body = m.group(1).strip()
        rec = memory_write(body, user=user or "anon", project_id=project_id or "default", tags=["note"])
        return {"ok": True, "mode": "memory.write", "result": rec}

    # 2) Buscar en memoria
    m = MEMQ_PAT.match(t)
    if m:
        q = m.group(1).strip()
        hits = memory_search(q, top_k=top_k, project_id=project_id, user=user)
        return {"ok": True, "mode": "memory.search", "result": hits}

    # 3) Mantenimiento memoria
    low = t.lower()
    if low.startswith("reindex"):
        out = memory_reindex(project_id=project_id)
        return {"ok": True, "mode": "memory.reindex", "result": out}
    if low.startswith("prune"):
        out = memory_prune()
        return {"ok": True, "mode": "memory.prune", "result": out}

    # 4) Por defecto: Ask → agente
    cfg = AgentConfig(min_score=min_score, top_k=top_k)
    out = agent_turn(t, user=user, project_id=project_id, cfg=cfg)
    return {"ok": True, "mode": "ask", "result": out}

if __name__ == "__main__":
    import sys
    txt = " ".join(sys.argv[1:]) or "nota: Prefiero RAG con citas."
    print(route_and_run(txt))
