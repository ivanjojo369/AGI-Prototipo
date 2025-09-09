from __future__ import annotations
import math, os, re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from .vector_store import SimpleVectorStore

_WORD = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", re.UNICODE)

def _tokens(s: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(s or "")]

class SemanticMemory:
    """
    Memoria semántica mínima:
    - Almacén JSON (SimpleVectorStore)
    - TF-IDF + coseno en memoria
    """

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.store_path = os.path.join(base_dir, "semantic_store.json")
        os.makedirs(base_dir, exist_ok=True)
        self.store = SimpleVectorStore(self.store_path)

        # índices en memoria
        self._doc_tf: Dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._idf: Dict[str, float] = {}
        self._doc_vec: Dict[str, Dict[str, float]] = {}

        self._rebuild_index()

    # ---------- utilidades ----------
    def _rebuild_index(self) -> None:
        """Reconstruye TF, DF, IDF y vectores de todos los docs."""
        self._doc_tf.clear()
        self._df.clear()
        self._idf.clear()
        self._doc_vec.clear()

        docs = self.store.items()
        # TF por doc y DF global
        for d in docs:
            doc_id = d["id"]
            toks = _tokens(d.get("text", ""))
            tf = Counter(toks)
            self._doc_tf[doc_id] = tf
            for t in tf.keys():
                self._df[t] += 1

        n_docs = max(1, len(docs))
        for t, df in self._df.items():
            # idf suave
            self._idf[t] = math.log((1 + n_docs) / (1 + df)) + 1.0

        for d in docs:
            doc_id = d["id"]
            self._doc_vec[doc_id] = self._tfidf_norm(self._doc_tf[doc_id])

    def _tfidf_norm(self, tf: Counter[str]) -> Dict[str, float]:
        vec: Dict[str, float] = {}
        for t, f in tf.items():
            vec[t] = (f * self._idf.get(t, 0.0))
        # normaliza
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        for t in list(vec.keys()):
            vec[t] /= norm
        return vec

    def _cos(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        # producto interno sobre la intersección
        # asume vectores ya normalizados
        if len(a) > len(b):
            a, b = b, a
        return sum(a.get(t, 0.0) * b.get(t, 0.0) for t in a.keys())

    # ---------- API pública ----------
    def upsert(self, payload: Dict[str, Any]) -> str:
        doc_id = self.store.upsert(payload)
        # actualizar índices de ese doc
        d = self.store.get(doc_id)
        tf = Counter(_tokens(d.get("text", "")))
        self._doc_tf[doc_id] = tf

        # actualizar DF e IDF de forma simple (recomputo global por seguridad)
        self._rebuild_index()
        return doc_id

    def search(self, query: str, top_k: int = 5, include_embeddings: bool = False) -> List[Dict[str, Any]]:
        q_vec = self._tfidf_norm(Counter(_tokens(query)))
        if not q_vec:
            return []

        results: List[Tuple[str, float]] = []
        for doc_id, d_vec in self._doc_vec.items():
            s = self._cos(q_vec, d_vec)
            if s > 0.0:
                results.append((doc_id, float(s)))

        results.sort(key=lambda x: x[1], reverse=True)
        hits = []
        for doc_id, score in results[: max(1, top_k)]:
            doc = self.store.get(doc_id) or {}
            hit = {"doc_id": doc_id, "score": float(score), "text": doc.get("text", "")}
            if include_embeddings:
                hit["embedding"] = self._doc_vec.get(doc_id)
            hits.append(hit)
        return hits

    # compat: a veces otros módulos esperan "retrieve" o "query"
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return self.search(query, top_k=k)

    def query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return self.search(query, top_k=k)

    def write(self, text: str, meta: Dict[str, Any] | None = None) -> str:
        """Atajo: inserta texto suelto generando id."""
        i = str(self.store.count + 1)
        return self.upsert({"id": i, "text": text, "tags": [], "meta": meta or {}})

    # métricas/depuración
    def count(self) -> int:
        return self.store.count

    # para scripts de depuración: rutas conocidas
    @property
    def paths(self) -> Dict[str, str]:
        return {"store_path": self.store_path, "base_dir": self.base_dir}
