# memory/embeddings_local.py
from __future__ import annotations

import os
from typing import List

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_model = None


def ensure_model():
    """Carga el embedding model una vez por proceso.

    Primero intenta usar exclusivamente la caché local. Esto evita las muchas
    peticiones HEAD a Hugging Face que antes aparecían en cada arranque de CLI.
    Si el modelo aún no existe en caché, se permite una descarga normal.
    """
    global _model
    if _model is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers no está instalado.")
        try:
            _model = SentenceTransformer(
                _MODEL_NAME,
                local_files_only=True,
            )
        except Exception:
            _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(texts: List[str]) -> List[List[float]]:
    model = ensure_model()
    return model.encode(texts, normalize_embeddings=True).tolist()
