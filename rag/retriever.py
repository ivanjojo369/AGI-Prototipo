# -*- coding: utf-8 -*-
"""
RAG Retriever actualizado
- Carga índice en data/semantic_store.json (lista de dicts con text/meta/vec).
- Compatibilidad con índice legado dict + (opcional) CORPUS_FILE.
- Umbral de score + re-ranking MMR.
- Embeddings compartidos con Memoria (n-gram stub).
- Helpers: search(), citations(), rag_answer(), repair(), CLI.
"""
from __future__ import annotations
import json, os
from typing import List, Dict, Any, Tuple, Optional

from root.settings import (
    SEMANTIC_STORE_JSON, EMBED_DIM,
    RAG_TOPK_DEFAULT, RAG_MAX_CHUNK_LEN, RAG_SCORE_THRESHOLD, RAG_MMR_LAMBDA
)

# Fallback para settings legado (si existiera)
try:
    from root.settings import CORPUS_FILE as _LEGACY_CORPUS_FILE  # type: ignore
except Exception:
    _LEGACY_CORPUS_FILE = None  # pragma: no cover

from memory.episodic_memory import embed_text  # mismo stub que memoria


# ---------------- Utilidades internas ----------------

def _load_index() -> List[Dict[str, Any]]:
    if not os.path.exists(SEMANTIC_STORE_JSON):
        return []
    try:
        with open(SEMANTIC_STORE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    if isinstance(data, list):
        fixed = []
        for i, it in enumerate(data):
            if not isinstance(it, dict):
                continue
            txt = it.get("text", "")
            if not txt:
                continue
            fixed.append({
                "id": it.get("id", f"doc_{i}"),
                "text": txt,
                "meta": it.get("meta") or {},
                "vec": it.get("vec") or [],
            })
        return fixed

    if isinstance(data, dict):
        if not _LEGACY_CORPUS_FILE or not os.path.exists(_LEGACY_CORPUS_FILE):
            return []
        id2text: Dict[str, str] = {}
        try:
            with open(_LEGACY_CORPUS_FILE, "r", encoding="utf-8") as cf:
                for raw in cf:
                    s = raw.strip()
                    if not s:
                        continue
                    try:
                        rec = json.loads(s)
                        did = rec.get("id")
                        if did:
                            id2text[did] = rec.get("text", "")
                    except Exception:
                        continue
        except Exception:
            return []

        fixed = []
        for did, info in data.items():
            txt = id2text.get(did, "")
            if not txt:
                continue
            fixed.append({
                "id": did,
                "text": txt,
                "meta": {"source": "legacy"},
                "vec": info.get("vec") or [],
            })
        return fixed

    return []


def _cos(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def _mmr(qv: List[float], docs: List[Tuple[float, Dict[str, Any]]], k: int, lamb: float) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    selected_vecs: List[List[float]] = []
    candidates = docs[:]
    while candidates and len(selected) < k:
        best_idx = 0
        best_val = -1e9
        for i, (sim_q, it) in enumerate(candidates):
            if not selected:
                val = sim_q
            else:
                max_sim = max(_cos(it.get("vec", []), sv) for sv in selected_vecs) if selected_vecs else 0.0
                val = lamb * sim_q - (1.0 - lamb) * max_sim
            if val > best_val:
                best_val = val
                best_idx = i
        sim_q, chosen = candidates.pop(best_idx)
        selected.append(chosen)
        selected_vecs.append(chosen.get("vec", []))
    return selected


# ---------------- API pública ----------------

def search(query: str,
           top_k: int = RAG_TOPK_DEFAULT,
           min_score: float = RAG_SCORE_THRESHOLD,
           mmr_lambda: float = RAG_MMR_LAMBDA,
           max_chunk_len: int = RAG_MAX_CHUNK_LEN) -> List[Dict[str, Any]]:
    index = _load_index()
    if not index:
        return []

    qv = embed_text(query or "")
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for it in index:
        vec = it.get("vec") or embed_text(it.get("text", ""))
        sc = _cos(qv, vec)
        if sc < float(min_score):
            continue
        txt = it.get("text", "")
        if len(txt) > int(max_chunk_len):
            txt = txt[:int(max_chunk_len)] + " …[truncado]"
        scored.append((sc, {**it, "text": txt, "score": float(round(sc, 6))}))

    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = _mmr(qv, scored, k=max(1, int(top_k)), lamb=float(mmr_lambda)) if scored else []

    for r in reranked:
        r.pop("vec", None)
    return reranked


def citations(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ""
    lines = []
    for i, h in enumerate(hits, 1):
        meta = h.get("meta") or {}
        path = meta.get("path", meta.get("source", ""))
        score = h.get("score", 0.0)
        lines.append(f"[{i}] {path} (score={score:.3f})")
    return "\n".join(lines)


def rag_answer(query: str,
               top_k: int = RAG_TOPK_DEFAULT,
               min_score: float = RAG_SCORE_THRESHOLD):
    """
    Construye una respuesta “preview” con fragmentos y devuelve también las citas.
    (Corregido: evita backslashes dentro de la expresión de f-strings.)
    """
    hits = search(query=query, top_k=top_k, min_score=min_score)
    if not hits:
        text = ("No se encontró contexto relevante. Indexa con "
                "`python -m scripts.index_folder --path ./docs --ext .md,.py --mode fresh`.")
        return text, []

    snips: List[str] = []
    for h in hits:
        _score = h.get("score", 0.0)
        _txt = (h.get("text") or "").replace("\n", " ")
        snips.append(f"- ({_score:.3f}) {_txt}")

    text = "Contexto RAG:\n" + "\n".join(snips)
    return text, hits


def repair() -> Dict[str, Any]:
    if not os.path.exists(SEMANTIC_STORE_JSON):
        return {"ok": True, "fixed": 0, "reason": "no-index"}
    try:
        with open(SEMANTIC_STORE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"ok": False, "error": f"load-error: {e}"}

    fixed: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for it in data:
            if isinstance(it, dict) and "text" in it:
                fixed.append({
                    "id": it.get("id"),
                    "text": it.get("text", ""),
                    "meta": it.get("meta") or {},
                    "vec": it.get("vec") or [],
                })
    elif isinstance(data, dict):
        if not _LEGACY_CORPUS_FILE or not os.path.exists(_LEGACY_CORPUS_FILE):
            return {"ok": False, "error": "legacy-index-without-corpus"}
        id2text: Dict[str, str] = {}
        with open(_LEGACY_CORPUS_FILE, "r", encoding="utf-8") as cf:
            for raw in cf:
                s = raw.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                    did = rec.get("id")
                    if did:
                        id2text[did] = rec.get("text", "")
                except Exception:
                    continue
        for did, info in data.items():
            txt = id2text.get(did, "")
            if not txt:
                continue
            fixed.append({"id": did, "text": txt, "meta": {"source": "legacy"}, "vec": info.get("vec") or []})
    else:
        return {"ok": False, "error": "unknown-index-format"}

    with open(SEMANTIC_STORE_JSON, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)
    return {"ok": True, "fixed": len(fixed)}


# ---------------- CLI ----------------

def _cli_search(args):
    hits = search(args.query, args.topk, args.min_score, args.mmr_lambda)
    print(json.dumps(
        [{"id": h.get("id"), "score": h.get("score"), "meta": h.get("meta"),
          "preview": (h.get("text") or "")[:140]} for h in hits],
        ensure_ascii=False, indent=2
    ))

def _cli_repair(_args):
    print(json.dumps(repair(), ensure_ascii=False, indent=2))

def main(argv: Optional[List[str]] = None):
    import argparse
    p = argparse.ArgumentParser("rag.retriever")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("search")
    s1.add_argument("-q", "--query", required=True)
    s1.add_argument("-k", "--topk", type=int, default=RAG_TOPK_DEFAULT)
    s1.add_argument("--min-score", type=float, default=RAG_SCORE_THRESHOLD)
    s1.add_argument("--mmr-lambda", type=float, default=RAG_MMR_LAMBDA)
    s1.set_defaults(func=_cli_search)

    s2 = sub.add_parser("repair")
    s2.set_defaults(func=_cli_repair)

    args = p.parse_args(argv)
    args.func(args)

__all__ = ["search", "citations", "rag_answer", "repair"]

if __name__ == "__main__":
    main()
