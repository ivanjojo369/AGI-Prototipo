# root/settings.py
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJ_DIR = ROOT_DIR.parent

DATA_DIR = PROJ_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Stores separados: memoria semántica y RAG documental nunca comparten archivo.
SEMANTIC_STORE_JSON = os.getenv(
    "SEMANTIC_STORE_JSON",
    str(DATA_DIR / "semantic_store.json"),
)
RAG_STORE_JSON = os.getenv(
    "RAG_STORE_JSON",
    str(DATA_DIR / "rag_store.json"),
)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default


EMBED_DIM = _env_int("EMBED_DIM", 1536)
RAG_TOPK_DEFAULT = _env_int("RAG_TOPK_DEFAULT", 5)
RAG_MAX_CHUNK_LEN = _env_int("RAG_MAX_CHUNK_LEN", 1200)
RAG_SCORE_THRESHOLD = _env_float("RAG_SCORE_THRESHOLD", 0.10)
RAG_MMR_LAMBDA = _env_float("RAG_MMR_LAMBDA", 0.70)

CHUNK_SIZE = _env_int("RAG_CHUNK_SIZE", 900)
CHUNK_OVERLAP = _env_int("RAG_CHUNK_OVERLAP", 120)
