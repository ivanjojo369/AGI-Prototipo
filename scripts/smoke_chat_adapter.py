#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/smoke_chat_adapter.py

Prueba rápida de chat con:
- llama-cpp-python (modelo GGUF local)
- RAG opcional (si existe rag.retriever.search)
- Verificación y autocorrección opcional (si existe verifier)

Ejemplo:
  $env:PYTHONPATH = (Get-Location).Path
  python scripts/smoke_chat_adapter.py ^
    --model models\\Qwen2.5-3B-Instruct-Q4_K_M.gguf ^
    --query "Define episodios de memoria y su estructura" ^
    --use-rag --verify

Requisitos:
  pip install "llama-cpp-python>=0.2.0"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ------------------------- Utilidades ------------------------- #

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def sanitize_text(s: str, limit: int = 400) -> str:
    """Quita saltos/espacios raros para que sea seguro usar en f-strings."""
    s = (s or "")
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    if limit and len(s) > limit:
        s = s[:limit] + "..."
    return s


# ------------------------- RAG opcional ------------------------- #

def build_rag_context(query: str, top_k: int, min_score: float) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Si existe rag.retriever.search, devuelve un bloque de contexto RAG y lista de hits.
    Si no, devuelve ("", []).
    """
    try:
        rag = importlib.import_module("rag.retriever")
    except Exception:
        return "", []

    search = getattr(rag, "search", None)
    if not callable(search):
        return "", []

    hits = search(query, top_k=top_k, min_score=min_score) or []
    if not hits:
        return "", []

    lines = ["# CONTEXTO RAG", ""]
    for h in hits[:top_k]:
        text = sanitize_text(h.get("text", ""), 600)
        score = float(h.get("score", 0.0))
        meta = h.get("meta") or {}
        src = meta.get("path") or meta.get("source") or "local"
        # Evitar backslashes dentro de {} -> ya está sanitizado
        lines.append(f"- ({score:.3f}) {text}")
        lines.append(f"  fuente: {sanitize_text(str(src), 200)}")
    lines.append("")

    return "\n".join(lines), hits


# ------------------------- Verificación opcional ------------------------- #

def run_verifier(answer: str) -> Dict[str, Any]:
    """
    Si existe verifier.verify / verifier.attempt_self_correction, los usa.
    Si no, devuelve un 'ok' True por defecto.
    """
    try:
        ver = importlib.import_module("verifier")
    except Exception:
        return {"ok": True, "autocorrected": False, "output": answer, "issues": [], "suggestions": []}

    out = {"ok": True, "autocorrected": False, "output": answer, "issues": [], "suggestions": []}
    try:
        verify_fn = getattr(ver, "verify", None)
        if callable(verify_fn):
            vr = verify_fn(answer) or {}
            out.update(vr)
    except Exception as ex:
        eprint(f"[WARN] verify() falló: {ex}")

    # autocorrección si no pasó
    if not out.get("ok", True):
        try:
            ac = getattr(ver, "attempt_self_correction", None)
            if callable(ac):
                fixed = ac(answer, {"reason": "autocorrect"}) or {}
                if "output" in fixed and fixed["output"]:
                    out["autocorrected"] = True
                    out["output"] = fixed["output"]
        except Exception as ex:
            eprint(f"[WARN] attempt_self_correction() falló: {ex}")

    return out


# ------------------------- Modelo llama-cpp ------------------------- #

def load_llama(
    model_path: str,
    n_ctx: int = 4096,
    n_gpu_layers: int = 0,
    n_threads: Optional[int] = None,
):
    """
    Carga Llama de llama-cpp-python. Lanza error amigable si no está instalado.
    """
    try:
        from llama_cpp import Llama
    except Exception:
        eprint("[ERROR] No se encontró llama-cpp-python.")
        eprint("        Instala con:  pip install \"llama-cpp-python>=0.2.0\"")
        raise

    kwargs = dict(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    if n_threads is not None and n_threads > 0:
        kwargs["n_threads"] = n_threads

    return Llama(**kwargs)


def chat_completion(
    llm,
    messages: List[Dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """
    Usa la plantilla de chat del modelo (si existe).
    """
    # llama-cpp >= 0.2.0
    res = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=False,
    )
    try:
        return (res["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return json.dumps(res, ensure_ascii=False)


# ------------------------- Prompt ------------------------- #

def build_messages(query: str, rag_block: str, system_hint: Optional[str]) -> List[Dict[str, str]]:
    """
    Crea el histórico de mensajes para modelos Instruct (ChatML/llama-cpp).
    Si hay RAG, se incluye como contexto en el mensaje del system.
    """
    sys_lines: List[str] = []
    if system_hint:
        sys_lines.append(system_hint.strip())

    if rag_block:
        sys_lines.append("Responde usando únicamente el CONTEXTO RAG si aplica.")
        sys_lines.append(rag_block.strip())

    system_prompt = "\n\n".join(sys_lines).strip() if sys_lines else ""

    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": query})
    return msgs


# ------------------------- CLI principal ------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Smoke Chat Adapter (llama-cpp + RAG + Verifier opcional)")
    ap.add_argument("--model", type=str, required=True, help="Ruta al modelo GGUF (ej. models\\Qwen2.5-3B-Instruct-Q4_K_M.gguf)")
    ap.add_argument("--query", type=str, required=True, help="Pregunta del usuario")
    ap.add_argument("--use-rag", action="store_true", help="Habilita RAG si existe rag.retriever")
    ap.add_argument("--top-k", type=int, default=5, help="Top-k RAG (por defecto 5)")
    ap.add_argument("--min-score", type=float, default=0.45, help="Umbral min de score para RAG (por defecto 0.45)")
    ap.add_argument("--verify", action="store_true", help="Verificar y autocorregir con verifier si existe")
    ap.add_argument("--max-tokens", type=int, default=256, help="Máximo de tokens de salida")
    ap.add_argument("--temperature", type=float, default=0.7, help="Temperatura de decodificación")
    ap.add_argument("--top-p", type=float, default=0.95, help="Top-p de decodificación")
    ap.add_argument("--n-ctx", type=int, default=4096, help="Context window")
    ap.add_argument("--n-gpu-layers", type=int, default=0, help="Capas en GPU (0=CPU)")
    ap.add_argument("--n-threads", type=int, default=0, help="Hilos CPU (0=auto)")
    ap.add_argument("--system", type=str, default="Responde de forma breve, precisa y basada en el contexto.",
                    help="Instrucción de sistema")

    args = ap.parse_args()

    # 1) RAG opcional
    rag_block = ""
    rag_hits = []
    if args.use_rag:
        rag_block, rag_hits = build_rag_context(args.query, args.top_k, args.min_score)

    # 2) Modelo
    t0 = time.time()
    llm = load_llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=(args.n_threads if args.n_threads > 0 else None),
    )
    load_ms = int((time.time() - t0) * 1000)

    # 3) Mensajes
    messages = build_messages(args.query, rag_block, args.system)

    # 4) Infiere
    t1 = time.time()
    answer = chat_completion(
        llm=llm,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    infer_ms = int((time.time() - t1) * 1000)

    # 5) Verificación opcional
    verified = {"ok": True, "autocorrected": False, "output": answer, "issues": []}
    if args.verify:
        verified = run_verifier(answer)

    # 6) Muestra
    print("\n== Smoke Chat Adapter ==\n")
    print(">> Query:")
    print(args.query)
    print("\n>> RAG Context:" if rag_block else "\n>> RAG Context: (sin RAG o sin hits)")
    if rag_block:
        print(rag_block)

    print(">> Respuesta (modelo):")
    print(answer)
    if args.verify:
        print("\n>> Verificación:")
        print(json.dumps(verified, ensure_ascii=False, indent=2))

    # 7) Stats
    stats = {
        "rag_hits": len(rag_hits),
        "min_score": args.min_score,
        "top_k": args.top_k,
        "load_ms": load_ms,
        "infer_ms": infer_ms,
    }
    print("\n>> Stats:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        eprint("\n[INTERRUPTED]")
        sys.exit(130)
