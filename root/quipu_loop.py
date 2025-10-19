# -*- coding: utf-8 -*-
"""
QuipuLoop con:
- Memoria (write/search)
- RAG (umbral + MMR + citas)
- Verificador (reglas + autocorrec. con LLM opcional)
- Guardrails (check input/output)
- Adaptador LLM opcional (p. ej., llama.cpp) vía importlib

Ejemplo:
  python -m root.quipu_loop --query "define episodios de memoria y su estructura" --min-score 0.45 --top-k 5
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Protocol, cast
import argparse
import importlib
import time

# ---- settings robusto ----
try:
    from .settings import RAG_TOPK_DEFAULT, RAG_SCORE_THRESHOLD, GUARD_CONF
except Exception:  # pragma: no cover
    try:
        from settings import RAG_TOPK_DEFAULT, RAG_SCORE_THRESHOLD, GUARD_CONF  # type: ignore
    except Exception:
        from root.settings import RAG_TOPK_DEFAULT, RAG_SCORE_THRESHOLD, GUARD_CONF  # type: ignore

# ---- memoria ----
from memory.memory import search as memory_search, write as memory_write

# ---- RAG ----
from rag.retriever import search as rag_search, citations as rag_citations

# ---- verifier ----
try:
    from .verifier import verify, attempt_self_correction, retry_plan
except Exception:  # pragma: no cover
    try:
        from verifier import verify, attempt_self_correction, retry_plan  # type: ignore
    except Exception:
        def verify(output: str, context: Dict[str, Any]) -> Dict[str, Any]:
            return {"passed": True, "issues": [], "suggestions": [], "task_type": "general"}
        def attempt_self_correction(draft: str, context: Dict[str, Any]) -> Dict[str, Any]:
            return {"output": draft, "used_model": False, "trace": ["verifier missing"]}
        def retry_plan(context: Dict[str, Any], verify_result: Dict[str, Any]) -> List[Dict[str, Any]]:
            return []

# ---- guardrails ----
try:
    from .guardrails import check_input, check_output
except Exception:  # pragma: no cover
    try:
        from guardrails import check_input, check_output  # type: ignore
    except Exception:
        def check_input(q: str, max_len: int = 4000):  # type: ignore
            return {"ok": len(q) <= max_len, "issues": [] if len(q) <= max_len else ["query demasiado larga"]}
        def check_output(a: str, max_len: int = 8000):  # type: ignore
            return {"ok": len(a) <= max_len, "issues": [] if len(a) <= max_len else ["respuesta demasiado larga"]}

# ---- adaptador opcional ----
class LLMAdapter(Protocol):
    MODEL_NAME: str
    def generate(self, prompt: str, max_tokens: int = ..., temperature: float = ...) -> str: ...

def _load_adapter() -> Optional[LLMAdapter]:
    try:
        mod = importlib.import_module("llama_cpp_adapter")
        return cast(LLMAdapter, mod)
    except Exception:
        return None

# ---- helpers ----
def _format_prompt(query: str, mem_hits: List[Dict[str, Any]], rag_hits: List[Dict[str, Any]]) -> str:
    secs: List[str] = []
    if mem_hits:
        lines = []
        for h in mem_hits:
            t = h.get("preview") or h.get("text") or ""
            t = t.splitlines()[0][:200]
            if t:
                lines.append(f"- {t}")
        if lines:
            secs.append("## MEMORIA (recientes/relevantes)\n" + "\n".join(lines))
    if rag_hits:
        lines = []
        for h in rag_hits:
            score = h.get("score", 0.0)
            t = (h.get("text") or "").replace("\n", " ")
            lines.append(f"- ({score:.3f}) {t}")
        if lines:
            secs.append("## CONTEXTO RAG\n" + "\n".join(lines))
    secs.append("## CONSULTA\n" + query.strip())
    secs.append(
        "## INSTRUCCIONES\n"
        "Responde de forma breve, directa y basada en MEMORIA y CONTEXTO RAG. "
        "No inventes fuentes; si no hay contexto suficiente, dilo explícitamente."
    )
    return "\n\n".join(secs)

def _inference(prompt: str, adapter: Optional[LLMAdapter], *, max_tokens: int = 512, temperature: float = 0.7) -> str:
    if adapter and hasattr(adapter, "generate"):
        try:
            return adapter.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature)  # type: ignore[arg-type]
        except Exception:
            pass
    preview = prompt if len(prompt) <= 1800 else (prompt[:1800] + " …")
    return "Respuesta (preview basada en contexto):\n\n" + preview

# ---- núcleo ----
@dataclass
class LoopStats:
    latency_ms: int
    rag_hits: int
    mem_hits: int
    min_score: float
    top_k: int

class QuipuLoop:
    def __init__(self, project_id: str = "default", *, min_score: float = RAG_SCORE_THRESHOLD, top_k: int = RAG_TOPK_DEFAULT) -> None:
        self.project_id = project_id
        self.min_score = float(min_score)
        self.top_k = int(top_k)
        self.adapter = _load_adapter()

    def run(self, query: str) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # Guardrails (entrada)
        ci = check_input(query, max_len=int(GUARD_CONF.get("max_prompt_chars", 4000)))
        if not ci["ok"]:
            blocked = " | ".join(ci["issues"])
            return {
                "ok": False,
                "output": f"Entrada bloqueada por guardrails: {blocked}",
                "stats": {"latency_ms": 0, "rag_hits": 0, "mem_hits": 0, "min_score": self.min_score, "top_k": self.top_k},
                "verified": {"ok": False, "autocorrected": False, "issues": ci["issues"], "suggestions": [], "trace": ["guardrails:input"], "retry_plan": []},
            }

        # 1) Memoria
        mem_hits = memory_search(query, topk=5, project_id=self.project_id)

        # 2) RAG
        rag_hits = rag_search(query, top_k=self.top_k, min_score=self.min_score)

        # 3) Generación
        prompt = _format_prompt(query, mem_hits, rag_hits)
        answer = _inference(prompt, self.adapter)

        # 4) Citas
        cites = rag_citations(rag_hits)
        if cites:
            answer = f"{answer}\n\nFuentes:\n{cites}"

        # Guardrails (salida)
        co = check_output(answer, max_len=int(GUARD_CONF.get("max_answer_chars", 8000)))
        if not co["ok"]:
            answer = "[Salida redacted por guardrails] " + " | ".join(co["issues"])

        # 5) Verificación + autocorrección
        ctx = {"rag_hits": rag_hits, "mem_hits": mem_hits, "query": query, "min_score": self.min_score}
        ver1 = verify(answer, ctx)
        autocorrected = False
        trace: List[str] = []
        if not ver1.get("passed", False):
            ac = attempt_self_correction(answer, ctx)
            answer = ac.get("output", answer)
            autocorrected = True
            trace = list(ac.get("trace", []))
        ver2 = verify(answer, ctx)
        rplan = retry_plan(ctx, ver2)

        # 6) Persistir episodio
        memory_write(
            f"Q: {query}\nA: {answer}",
            user="system",
            project_id=self.project_id,
            tags=["answer", "rag" if rag_hits else "no_rag"],
            importance=0.4,
            meta={
                "used_model": getattr(self.adapter, "MODEL_NAME", "preview/fallback") if self.adapter else "preview/fallback",
                "rag_hits": len(rag_hits),
                "min_score": self.min_score,
                "verified_ok": bool(ver2.get("passed")),
                "autocorrected": autocorrected,
            },
        )

        stats = LoopStats(
            latency_ms=int((time.perf_counter() - t0) * 1000),
            rag_hits=len(rag_hits),
            mem_hits=len(mem_hits),
            min_score=self.min_score,
            top_k=self.top_k,
        )
        return {
            "ok": True,
            "output": answer,
            "stats": vars(stats),
            "context": {"rag_hits": rag_hits, "mem_hits": mem_hits},
            "verified": {
                "ok": bool(ver2.get("passed")),
                "autocorrected": autocorrected,
                "issues": ver2.get("issues"),
                "suggestions": ver2.get("suggestions"),
                "trace": trace,
                "retry_plan": rplan,
            },
        }

# ---- CLI ----
def _cli() -> None:
    ap = argparse.ArgumentParser("root.quipu_loop")
    ap.add_argument("-q", "--query", type=str, required=True)
    ap.add_argument("--project-id", type=str, default="default")
    ap.add_argument("--top-k", type=int, default=RAG_TOPK_DEFAULT)
    ap.add_argument("--min-score", type=float, default=RAG_SCORE_THRESHOLD)
    args = ap.parse_args()

    loop = QuipuLoop(project_id=args.project_id, top_k=args.top_k, min_score=args.min_score)
    out = loop.run(args.query)
    print(out.get("output", ""))
    print("\n[stats]", out.get("stats", {}))
    print("[verified]", out.get("verified", {}))

if __name__ == "__main__":
    _cli()

__all__ = ["QuipuLoop"]
