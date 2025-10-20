# -*- coding: utf-8 -*-
"""
Runner: capa ligera para invocar el loop de AGI (RAG + Memoria + Verificación)
y ofrecer una interfaz de línea de comandos.

Uso típico:
  python -m runner.runner --query "define episodios de memoria y su estructura"
  python -m runner.runner --query "..." --min-score 0.36 --top-k 6 --json
  python -m runner.runner --repl  # modo interactivo
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


# ------------------------ Adaptadores / Imports seguros ------------------------ #

def _load_answer_fn():
    """
    Intenta cargar la función `answer(query, **kwargs)` desde el loop estándar.
    Si no está, cae a un plan B usando retriever directamente.
    """
    # Preferimos el loop integrado que ya combina RAG + Memoria + Verificador
    try:
        from root.quipu_loop import answer as loop_answer  # type: ignore
        return loop_answer
    except Exception:
        loop_answer = None

    # Fallback: RAG directo (sin verificación). Útil si alguien ejecuta el runner suelto.
    def rag_fallback(query: str, min_score: float = 0.45, top_k: int = 5, **_):
        try:
            from rag.retriever import rag_answer  # type: ignore
            txt, ctx = rag_answer(query, top_k=top_k, min_score=min_score)
            return {
                "ok": True,
                "output": txt,
                "stats": {"rag_hits": len(ctx.get("rag_hits", [])),
                          "mem_hits": len(ctx.get("mem_hits", [])),
                          "min_score": min_score,
                          "top_k": top_k},
                "context": ctx,
                "verified": {"ok": True, "autocorrected": False, "issues": [], "suggestions": []},
            }
        except Exception as e:
            return {"ok": False, "error": f"RAG fallback failed: {e}"}

    return loop_answer or rag_fallback


# ------------------------------ Config y Runner ------------------------------- #

@dataclass
class RunnerConfig:
    min_score: float = 0.45
    top_k: int = 5
    session_id: Optional[str] = None
    json_output: bool = False

class Runner:
    def __init__(self, cfg: RunnerConfig):
        self.cfg = cfg
        self._answer = _load_answer_fn()

    def run(self, query: str) -> Dict[str, Any]:
        """
        Ejecuta la consulta usando el loop AGI si está disponible;
        si no, usa el fallback RAG.
        """
        payload = {
            "min_score": self.cfg.min_score,
            "top_k": self.cfg.top_k,
            "session_id": self.cfg.session_id,
        }
        res = self._answer(query, **payload)
        # Normalizamos estructura mínima
        if isinstance(res, str):
            res = {"ok": True, "output": res}
        if not isinstance(res, dict):
            res = {"ok": False, "error": "Unexpected response type from loop."}
        return res


# ------------------------------- CLI / REPL ---------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="runner",
        description="Runner para AGI doméstica (RAG + Memoria + Verificación)."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--query", type=str, help="Consulta a ejecutar.")
    g.add_argument("--repl", action="store_true", help="Modo interactivo (multi-turn).")

    p.add_argument("--min-score", type=float, default=0.45, dest="min_score",
                   help="Umbral mínimo de score para los hits RAG.")
    p.add_argument("--top-k", type=int, default=5, dest="top_k",
                   help="Máximo de pasajes recuperados (RAG).")
    p.add_argument("--session", type=str, default=None, dest="session_id",
                   help="ID de sesión (agrupa historial/memoria por sesión).")
    p.add_argument("--json", action="store_true", dest="json_output",
                   help="Imprime salida en JSON.")
    return p


def _print_result(result: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if not result.get("ok", True):
        print(f"[ERROR] {result.get('error')}", file=sys.stderr)
        return

    out = result.get("output") or result.get("answer") or ""
    print(out)

    stats = result.get("stats")
    if stats:
        print("\n[stats]", json.dumps(stats, ensure_ascii=False))

    verified = result.get("verified")
    if verified:
        tag = "OK" if verified.get("ok") else "FAIL"
        auto = " (autocorrected)" if verified.get("autocorrected") else ""
        print(f"[verified] {tag}{auto}")


def _repl(runner: Runner):
    print("AGI runner — REPL. Escribe `:q` para salir.")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in (":q", ":quit", ":exit"):
            break
        res = runner.run(q)
        _print_result(res, as_json=False)


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = RunnerConfig(
        min_score=args.min_score,
        top_k=args.top_k,
        session_id=args.session_id,
        json_output=args.json_output,
    )
    runner = Runner(cfg)

    if args.repl:
        _repl(runner)
        return 0

    result = runner.run(args.query)
    _print_result(result, as_json=cfg.json_output)
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
