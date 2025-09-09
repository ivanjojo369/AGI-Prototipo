# memory/vector_store_faiss.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# pip install faiss-cpu
import faiss  # type: ignore

from .embeddings_local import embed

DATA_DIR   = Path(os.getenv("DATA_DIR", "data")) / "memory"
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH  = DATA_DIR / "faiss_meta.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)

class FaissStore:
    def __init__(self, dim: int = 384):  # MiniLM-L6-v2 -> 384
        self.dim = dim
        self.index = None
        self.meta: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if INDEX_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
        else:
            self.index = faiss.IndexFlatIP(self.dim)  # dot product (con embeddings normalizados ≈ cosine)
        if META_PATH.exists():
            self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    def _save(self):
        faiss.write_index(self.index, str(INDEX_PATH))
        META_PATH.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_texts(self, texts: List[str], metas: List[Dict[str, Any]] | None = None):
        if not texts: return
        vecs = embed(texts)
        import numpy as np  # lazy
        xb = np.array(vecs, dtype="float32")
        self.index.add(xb)
        for i, t in enumerate(texts):
            self.meta.append({"text": t, **((metas or [{}])[i] if metas else {})})
        self._save()

    def search(self, q: str, k: int = 3) -> List[Dict[str, Any]]:
        if not q: return []
        import numpy as np
        qv = np.array(embed([q]), dtype="float32")
        D, I = self.index.search(qv, k)
        out = []
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            if idx == -1: continue
            m = self.meta[idx] if idx < len(self.meta) else {}
            out.append({"text": m.get("text",""), "score": float(score), **m})
        return out
