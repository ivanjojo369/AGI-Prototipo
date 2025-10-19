# -*- coding: utf-8 -*-
"""
RAG mínimo con TF-IDF ligero (sin embeddings).
Expose funciones de alto nivel que usa el router: `rag_query` y `RAG_DEFAULT_DIR`.
"""
from __future__ import annotations

import os, json, math, re, time, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

# métricas opcionales
try:
    from prometheus_client import Counter, Histogram
    RAG_Q = Counter("quipu_rag_queries_total", "Total consultas RAG")
    RAG_L = Histogram("quipu_rag_latency_seconds", "Latencia RAG")
except Exception:  # pragma: no cover
    RAG_Q = RAG_L = None

CORPUS_DIR  = Path(os.getenv("RAG_CORPUS_DIR", "rag_corpus")).resolve()
INDEX_PATH  = Path(os.getenv("RAG_INDEX_PATH", "rag/index.json")).resolve()
ALLOWED_EXT = {e.strip().lower() for e in os.getenv("RAG_EXTS", ".txt,.md,.mdx").split(",")}
SNIPPET_LEN = int(os.getenv("RAG_SNIPPET_CHARS", "320"))

# Export por compat con router
RAG_DEFAULT_DIR = str(CORPUS_DIR)

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-záéíóúñA-ZÁÉÍÓÚÑ0-9]+", text.lower())

def _tf(words: List[str]) -> Dict[str, float]:
    tf: Dict[str, float] = {}
    for w in words:
        tf[w] = tf.get(w, 0.0) + 1.0
    n = float(len(words)) or 1.0
    for k in tf:
        tf[k] /= n
    return tf

def _corpus_signature(folder: Path) -> str:
    parts = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            st = p.stat()
            parts.append(f"{p}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()

class RagService:
    def __init__(self, corpus: Path = CORPUS_DIR, index: Path = INDEX_PATH):
        self.corpus = corpus
        self.index  = index
        self.data: Dict[str, Any] = {"docs": [], "idf": {}, "N": 1, "sig": ""}
        self.load_or_build()

    def load_or_build(self):
        sig = _corpus_signature(self.corpus) if self.corpus.exists() else ""
        if self.index.exists():
            try:
                tmp = json.loads(self.index.read_text(encoding="utf-8"))
                if tmp.get("sig") == sig:
                    self.data = tmp
                    return
            except Exception:
                pass
        self.build(sig)

    def build(self, sig: str):
        docs = []
        self.corpus.mkdir(parents=True, exist_ok=True)
        for p in self.corpus.rglob("*"):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
                text = p.read_text(encoding="utf-8", errors="replace")
                tokens = _tokenize(text)
                docs.append({"path": str(p), "text": text, "tf": _tf(tokens)})
        # idf
        df: Dict[str, int] = {}
        for d in docs:
            for w in d["tf"].keys():
                df[w] = df.get(w, 0) + 1
        N = len(docs) or 1
        idf = {w: math.log((N + 1) / (c + 1)) + 1.0 for w, c in df.items()}
        self.data = {"docs": docs, "idf": idf, "N": N, "sig": sig}
        self.index.parent.mkdir(parents=True, exist_ok=True)
        self.index.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")

    def query(self, q: str, top_k: int = 3, snippet_chars: int = SNIPPET_LEN) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()
        try:
            qw = _tokenize(q)
            if not qw:
                # top por longitud como fallback
                docs = sorted(self.data["docs"], key=lambda d: -len(d["text"]))[:top_k]
                return [{"score": 0.0, "doc": d["path"], "snippet": d["text"][:snippet_chars]} for d in docs]
            qtf: Dict[str, float] = {}
            for w in qw: qtf[w] = qtf.get(w, 0.0) + 1.0
            for w in qtf: qtf[w] /= len(qw)
            idf = self.data.get("idf", {})
            scores: List[Tuple[float, int]] = []
            for i, d in enumerate(self.data["docs"]):
                tf = d["tf"]; s = 0.0
                for w, t in qtf.items(): s += t * tf.get(w, 0.0) * idf.get(w, 1.0)
                if s > 0: scores.append((s, i))
            scores.sort(reverse=True)
            hits = []
            for s, i in scores[:top_k]:
                doc = self.data["docs"][i]
                txt = doc["text"]
                # snippet alrededor del primer término encontrado
                pos = -1
                for w in qw:
                    pos = txt.lower().find(w)
                    if pos >= 0: break
                if pos < 0: pos = 0
                start = max(0, pos - snippet_chars // 2)
                snip = txt[start:start + snippet_chars]
                hits.append({"score": float(s), "doc": doc["path"], "snippet": snip})
            return hits
        finally:
            if RAG_Q: RAG_Q.inc()
            if RAG_L: RAG_L.observe(time.perf_counter() - t0)

# Instancia global por defecto
RAG = RagService()

def rag_query(
    query: str,
    corpus_dir: str | None = None,
    k: int = 5,
    min_score: float = 0.0,
    max_chars: int = SNIPPET_LEN,
):
    """
    Wrapper de alto nivel que usa el servicio global RAG.
    - Permite cambiar dinámicamente el corpus pasándole `corpus_dir`.
    - Filtra por `min_score` y limita el snippet con `max_chars`.
    """
    global RAG
    if corpus_dir:
        corpus = Path(corpus_dir).resolve()
        # Si cambió el corpus, reconstruye la instancia
        if corpus != RAG.corpus:
            RAG = RagService(corpus=corpus, index=INDEX_PATH)
    hits = RAG.query(query, top_k=k, snippet_chars=max_chars)
    if min_score > 0:
        hits = [h for h in hits if h.get("score", 0.0) >= float(min_score)]
    return hits
