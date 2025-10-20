# -*- coding: utf-8 -*-
"""
Normalización básica de texto para indexación RAG (sin dependencias externas).
"""
from __future__ import annotations
import re
from typing import List, Tuple, Set

_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_CODE_FENCE = re.compile(r"```.*?```", re.S)
_HTML = re.compile(r"<[^>]+>")
_MULTI_WS = re.compile(r"\s+")

def clean_markdown(text: str) -> str:
    if not text:
        return ""
    t = text
    # quitar bloques de código
    t = _CODE_FENCE.sub(" ", t)
    # convertir [texto](url) -> texto
    t = _MD_LINK.sub(r"\1", t)
    # quitar tags HTML
    t = _HTML.sub(" ", t)
    return squash_spaces(t)

def squash_spaces(text: str) -> str:
    return _MULTI_WS.sub(" ", (text or "")).strip()

def _shingles(tokens: List[str], k: int = 8) -> Set[Tuple[str, ...]]:
    if len(tokens) < k:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i+k]) for i in range(len(tokens)-k+1)}

def dedupe_chunks(chunks: List[str], jaccard_threshold: float = 0.92) -> List[int]:
    """
    Devuelve los índices de chunks a conservar tras deduplicación por Jaccard (shingles).
    O(n^2) simple pero suficiente para repos medianos.
    """
    keep: List[int] = []
    seen: List[Set[Tuple[str, ...]]] = []
    for i, ch in enumerate(chunks):
        toks = (ch or "").lower().split()
        sh = _shingles(toks, k=8)
        is_dup = False
        for other in seen:
            inter = len(sh & other)
            uni = len(sh | other) or 1
            j = inter / uni
            if j >= jaccard_threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)
            seen.append(sh)
    return keep
