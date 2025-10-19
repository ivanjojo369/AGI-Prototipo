# memory/vector_store_faiss.py
from __future__ import annotations
import os, json, tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from itertools import zip_longest

import numpy as np  # numpy a nivel módulo
try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError(
        "FAISS no está disponible. Instala 'faiss-cpu' o 'faiss-gpu'."
    ) from e

from .embeddings_local import embed

DATA_DIR   = Path(os.getenv("DATA_DIR", "data")) / "memory"
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH  = DATA_DIR / "faiss_meta.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def _atomic_write_index(index, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(tmp))
    Path(str(tmp)).replace(path)

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    # Evita divisiones por cero
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

class FaissStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = None
        self.meta: List[Dict[str, Any]] = []
        self._load()

    def _new_index(self) -> Any:
        # Producto punto + vectores normalizados ≈ cosine
        return faiss.IndexFlatIP(self.dim)

    def _load(self) -> None:
        if INDEX_PATH.exists():
            idx = faiss.read_index(str(INDEX_PATH))
            # Verifica dimension
            d = idx.d if hasattr(idx, "d") else getattr(idx, "ntotal", None)
            if d is not None and d != self.dim:
                # Si hay desajuste, reconstruye un índice vacío con la nueva dim
                idx = self._new_index()
            self.index = idx
        else:
            self.index = self._new_index()

        if META_PATH.exists():
            try:
                self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
                if not isinstance(self.meta, list):
                    self.meta = []
            except Exception:
                self.meta = []

        # Si el índice afirma tener más vectores que self.meta, evita lecturas fuera de rango
        if hasattr(self.index, "ntotal"):
            ntotal = int(self.index.ntotal)  # type: ignore[attr-defined]
            if ntotal < len(self.meta):
                self.meta = self.meta[:ntotal]

    def _save(self) -> None:
        _atomic_write_index(self.index, INDEX_PATH)
        _atomic_write_text(META_PATH, json.dumps(self.meta, ensure_ascii=False, indent=2))

    def add_texts(self, texts: List[str], metas: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        if not texts:
            return []
        vecs = embed(texts)
        xb = np.asarray(vecs, dtype="float32")
        xb = _l2_normalize(xb)
        self.index.add(xb)

        # Alinea metadatos de forma segura
        new_ids: List[int] = []
        base = len(self.meta)
        for i, (t, m) in enumerate(zip_longest(texts, metas or [], fillvalue={})):
            self.meta.append({"text": t, **(m or {})})
            new_ids.append(base + i)

        self._save()
        return new_ids

    def search(self, q: str, k: int = 3) -> List[Dict[str, Any]]:
        if not q:
            return []
        # Cortocircuito: índice vacío
        ntotal = int(self.index.ntotal)  # type: ignore[attr-defined]
        if ntotal == 0:
            return []

        qv = np.asarray(embed([q]), dtype="float32")
        qv = _l2_normalize(qv)

        D, I = self.index.search(qv, k)
        out: List[Dict[str, Any]] = []
        idxs = I[0].tolist()
        scores = D[0].tolist()

        for idx, score in zip(idxs, scores):
            if idx == -1:
                continue
            m = self.meta[idx] if idx < len(self.meta) else {}
            out.append({
                "text": m.get("text", ""),
                "score": float(score),
                "id": idx,
                **m
            })
        return out

    def get_status(self) -> Dict[str, Any]:
        return {
            "size": int(self.index.ntotal),  # type: ignore[attr-defined]
            "dim": self.dim,
            "meta_len": len(self.meta),
            "ok": True
        }
