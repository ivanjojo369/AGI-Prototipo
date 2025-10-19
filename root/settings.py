# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import os
API_KEY = os.getenv("AGI_API_KEY", "")  # si vacío, no exige API-Key (modo dev)

# --- Rutas base (root/) ---
BASE_DIR = Path(__file__).resolve().parent

# --- Datos / Memoria ---
DATA_DIR          = BASE_DIR / "data"
MEMORY_DIR        = DATA_DIR / "memory"
SESSIONS_DIR      = MEMORY_DIR / "sessions"
NOTES_DIR         = MEMORY_DIR / "notes"
MEMORY_BACKUP_DIR = MEMORY_DIR / "_backup"
JOBS_DIR          = DATA_DIR / "jobs"

# Archivos principales
EPISODES_JSONL        = MEMORY_DIR / "episodes.jsonl"
SEMANTIC_MEMORY_JSON  = MEMORY_DIR / "semantic_memory.json"
SEMANTIC_STORE_JSON   = DATA_DIR / "semantic_store.json"    # índice de RAG (chunks)
VECTOR_META_FILE      = MEMORY_DIR / "vector_meta.json"
JOBS_JSONL            = JOBS_DIR / "jobs.jsonl"

# Crear carpetas si no existen
for p in (DATA_DIR, MEMORY_DIR, SESSIONS_DIR, NOTES_DIR, MEMORY_BACKUP_DIR, JOBS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# --- Embedding stub / compatibilidad ---
EMBED_DIM = 384  # n-gram stub; usa el mismo en retriever y memoria

# --- Memoria de largo plazo ---
MEMORY_CONF = dict(
    recent_k=6,
    semantic_top_k=6,
    episodic_max_items=20000,
    episodic_ttl_days=365,
    semantic_max_items=40000,
    semantic_ttl_days=730,
)

# --- RAG ---
RAG_TOPK_DEFAULT    = 6       # cuántos chunks devolver
RAG_MAX_CHUNK_LEN   = 1600    # truncado de seguridad por chunk
RAG_SCORE_THRESHOLD = 0.45    # umbral mínimo recomendado para hits
RAG_MMR_LAMBDA      = 0.55    # 0..1 (diversidad vs relevancia) en re-ranking MMR

# --- Guardrails/loop ---
STEP_SOFT_TIMEOUT_SECS = 5
MIN_ANSWER_LEN         = 40
LOOP_MAX_STEPS         = 8
REPLAN_MAX_ATTEMPTS    = 1
ALLOWLIST_TOOLS = {"qa_rag": True, "write_note": True}

# --- Verifier / Autocorrección ---
VERIFY_CONF = dict(
    min_chars=80,
    must_end_with_punctuation=True,
    banned_phrases=[
        "no puedo", "no es posible", "como modelo de lenguaje", "lo siento",
    ],
    require_citation_on_rag=True,
    min_rag_citations=1,
    min_rag_score=RAG_SCORE_THRESHOLD,  # usa el mismo umbral que el retriever
)

MODEL_CONF = dict(
    max_tokens=512,
    temperature=0.7,
)

# --- Guardrails / límites ---
GUARD_CONF = dict(
    max_prompt_chars=4000,
    max_answer_chars=8000,
)
