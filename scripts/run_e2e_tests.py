# -*- coding: utf-8 -*-
"""
Runner de pruebas end-to-end (sin pytest).
- Reindexa carpetas del proyecto (fresh + append).
- Prueba Memoria (write/search/prune/reindex).
- Prueba RAG (search con umbral).
- Ejecuta el loop (QuipuLoop) y valida verificación/autocorrección.

Uso:
  python run_e2e_test.py
  python run_e2e_test.py --min-score 0.45 --top-k 5 --paths root memory rag executive docs
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

# --------- imports robustos a distintos layouts ----------
try:
    from root.quipu_loop import QuipuLoop
except Exception:
    # si no se pudo por paquete, intentamos import plano
    from quipu_loop import QuipuLoop  # type: ignore

try:
    from root.settings import RAG_SCORE_THRESHOLD, RAG_TOPK_DEFAULT
except Exception:
    try:
        from settings import RAG_SCORE_THRESHOLD, RAG_TOPK_DEFAULT  # type: ignore
    except Exception:
        RAG_SCORE_THRESHOLD, RAG_TOPK_DEFAULT = 0.35, 5

from memory.memory import write as mem_write, search as mem_search, prune as mem_prune, reindex as mem_reindex
from rag.retriever import search as rag_search


def _exists_dir(d: str | Path) -> bool:
    return Path(d).exists() and Path(d).is_dir()


def reindex(paths: List[str], *, fresh_first: bool = True) -> bool:
    ok_all = True
    if not paths:
        return True
    first_done = False
    for p in paths:
        if not _exists_dir(p):
            continue
        mode = "fresh" if (fresh_first and not first_done) else "append"
        first_done = True or first_done
        cmd = [sys.executable, "-m", "scripts.index_folder", "--path", p, "--ext", ".py,.md,.txt", "--mode", mode]
        print("  →", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print("[index_folder] ERROR:", proc.stderr.strip())
            ok_all = False
        else:
            print(proc.stdout.strip())
    return ok_all


def test_memory() -> bool:
    print("• Test Memoria: write/search/prune/reindex")
    r1 = mem_write("nota e2e: preferencias RAG vs heurísticas", user="tester", project_id="e2e", tags=["e2e"])
    if not r1.get("ok"):
        print("  ✗ write falló")
        return False
    hits = mem_search("preferencias RAG", topk=3, project_id="e2e")
    if not hits:
        print("  ✗ search no devolvió resultados")
        return False
    pr = mem_prune()
    if not pr.get("ok"):
        print("  ✗ prune falló")
        return False
    r2 = mem_reindex(project_id="e2e")
    if not r2.get("ok"):
        print("  ✗ reindex falló")
        return False
    print("  ✓ Memoria OK")
    return True


def test_rag(min_score: float) -> bool:
    print("• Test RAG: search con umbral")
    probes = [
        "memoria episodica",
        "episodios de memoria",
        "settings del proyecto",
        "retriever rag",
        "loop quipu",
    ]
    for q in probes:
        hits = rag_search(q, top_k=5, min_score=min_score)
        if hits:
            print(f"  ✓ RAG OK con consulta: '{q}' → {len(hits)} hit(s), min_score={min_score}")
            return True
    print("  ✗ RAG sin resultados (ajusta min_score o reindexa otras carpetas)")
    return False


def test_loop(min_score: float, top_k: int) -> bool:
    print("• Test Loop: ejecución y verificación")
    loop = QuipuLoop(project_id="e2e", min_score=min_score, top_k=top_k)
    out = loop.run("define episodios de memoria y su estructura")
    ok = bool(out.get("ok")) and isinstance(out.get("output"), str)
    verified = out.get("verified", {})
    print("  salida:", (out.get("output") or "")[:200].replace("\n", " "), "…")
    print("  stats:", out.get("stats"))
    print("  verified:", verified)
    return ok and ("ok" in verified)


def main():
    ap = argparse.ArgumentParser("run_e2e_test")
    ap.add_argument("--min-score", type=float, default=RAG_SCORE_THRESHOLD)
    ap.add_argument("--top-k", type=int, default=RAG_TOPK_DEFAULT)
    ap.add_argument("--paths", nargs="*", default=["root", "memory", "rag", "executive", "docs"])
    args = ap.parse_args()

    print("== Reindexar ==")
    idx_ok = reindex(args.paths, fresh_first=True)

    print("\n== Pruebas ==")
    m_ok = test_memory()
    r_ok = test_rag(args.min_score)
    l_ok = test_loop(args.min_score, args.top_k)

    summary = {
        "index_ok": idx_ok,
        "memory_ok": m_ok,
        "rag_ok": r_ok,
        "loop_ok": l_ok,
    }
    print("\n== RESUMEN ==")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # exit code
    sys.exit(0 if all(summary.values()) else 1)


if __name__ == "__main__":
    main()
