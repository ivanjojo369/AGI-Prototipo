# memory/embeddings_local.py
from __future__ import annotations
import os
from typing import List
try:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
except Exception as e:
    SentenceTransformer = None

_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_model = None

def ensure_model():
    global _model
    if _model is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers no está instalado.")
        _model = SentenceTransformer(_MODEL_NAME)
    return _model

def embed(texts: List[str]) -> List[List[float]]:
    m = ensure_model()
    # normalize_embeddings=True produce vectores unitarios (útil en cosine sim)
    return m.encode(texts, normalize_embeddings=True).tolist()
