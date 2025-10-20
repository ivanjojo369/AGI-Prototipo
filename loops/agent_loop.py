# -*- coding: utf-8 -*-
"""
Agent loop: orquesta una consulta end-to-end con:
- RAG (búsqueda + respuesta con citas)
- Memoria (hits recientes, opcional)
- Verificación + autocorrección (si procede)

Diseñado para ser resistente a import errors: si falta algún módulo,
expone stubs que mantienen el flujo funcionando.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Tuple, Optional

# --------- Imports con fallback seguros ---------
try:
    from rag.retriever import search as rag_search, citations as rag_citations, rag_answer
except Exception:  # pragma: no cover
    def rag_search(q: str, top_k: int = 5, min_score: float = 0.45) -> List[Dict[str, Any]]:
        return []
    def rag_citations(hits: List[Dict[str, Any]]) -> str:
        return ""
    def rag_answer(q: str, top_k: int = 5, min_score: float = 0.45):
        # Compat con (texto) o (texto, ctx)
        return "Sin contexto suficiente para responder.", {}

try:
    from memory.memory import search as memory_search, write as memory_write  # aliases expuestos
except Exception:  # pragma: no cover
    def memory_search(q: str, top_k: int = 3, project_id: Optional[str] = None,
                      user: Optional[str] = None, tags: Optional[List[str]] = None):
        return []
    def memory_write(text: str, **meta):
        return {"ok": True, "id": "mem-dummy", "meta": meta}

try:
    from root.verifier import verify, attempt_self_correction
except Exception:  # pragma: no cover
    def verify(output_preview: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        # “Aprobar” por defecto si no hay verificador
        return {"ok": True, "autocorrected": False, "issues": [], "suggestions": [], "trace": [], "retry_plan": []}
    def attempt_self_correction(text: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": text, "used_model": False, "trace": ["[no adapter]"]}

# --------- Config ---------
@dataclass
class AgentConfig:
    min_score: float = 0.45
    top_k: int = 5
    allow_autocorrect: bool = True
    include_mem_hits: bool = True

# --------- Util ---------
def _unpack_rag_answer(ans) -> Tuple[str, Dict[str, Any]]:
    """Acepta ans = texto | (texto, ctx) y lo normaliza."""
    if isinstance(ans, tuple) and len(ans) >= 1:
        txt = ans[0]
        meta = ans[1] if len(ans) > 1 and isinstance(ans[1], dict) else {}
        return str(txt), meta
    return str(ans), {}

# --------- API principal ---------
def agent_turn(
    query: str,
    user: Optional[str] = None,
    project_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    cfg: AgentConfig = AgentConfig(),
) -> Dict[str, Any]:
    """
    Ejecuta un turno del agente sobre `query`.
    Devuelve dict con: output, stats, context (hits), verified.
    """
    t0 = perf_counter()

    # 1) RAG: búsqueda
    hits = rag_search(query, top_k=cfg.top_k, min_score=cfg.min_score) or []
    cite_block = rag_citations(hits) if hits else ""

    # 2) Opcional: memoria (hits recientes/rel.)
    mem_hits = memory_search(query, top_k=3, project_id=project_id, user=user, tags=tags) if cfg.include_mem_hits else []

    # 3) Respuesta preliminar vía RAG
    ans_txt, rag_ctx = _unpack_rag_answer(rag_answer(query, top_k=cfg.top_k, min_score=cfg.min_score))

    # Si RAG no trae nada, dilo explícitamente
    if not hits and not ans_txt.strip():
        ans_txt = "No encuentro contexto suficiente. Si puedes, dame más detalle o intenta con otra formulación."

    # 4) Verificación + autocorrección
    ctx = {"rag_hits": hits, "min_score": cfg.min_score, "top_k": cfg.top_k, "mem_hits": mem_hits}
    ver = verify(ans_txt, ctx)
    final_text = ans_txt
    if not ver.get("ok", True) and cfg.allow_autocorrect:
        ac = attempt_self_correction(ans_txt, ctx)
        if isinstance(ac, dict) and ac.get("output"):
            final_text = ac["output"]
            ver = {**ver, "autocorrected": True, "used_model": ac.get("used_model", ver.get("used_model", False))}
        else:
            ver = {**ver, "autocorrected": False}

    # 5) Ensamblado final
    latency_ms = int((perf_counter() - t0) * 1000)
    stats = {
        "latency_ms": latency_ms,
        "rag_hits": len(hits),
        "mem_hits": len(mem_hits),
        "min_score": cfg.min_score,
        "top_k": cfg.top_k,
    }
    context = {"rag_hits": hits, "mem_hits": mem_hits, "rag_ctx": rag_ctx, "citations": cite_block}

    return {
        "ok": True,
        "output": final_text,
        "preview": ans_txt,
        "stats": stats,
        "context": context,
        "verified": ver,
    }

# --------- CLI rápido ---------
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "define episodios de memoria y su estructura"
    res = agent_turn(q)
    print("Respuesta:\n", res["output"])
    print("\n[stats]", res["stats"])
    if res["context"].get("citations"):
        print("\nFuentes:\n", res["context"]["citations"])
