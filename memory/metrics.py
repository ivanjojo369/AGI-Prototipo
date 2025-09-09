# memory/metrics.py
# Métricas para Memoria Semántica "pro": retrieval@k, used_fact y logging a CSV.

from __future__ import annotations
import os, csv, time, math, hashlib, logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterable

import numpy as np

# Intentamos usar el mismo encoder local que el vector store
try:
    from memory.embeddings_local import EmbeddingsLocal  # tu clase local
except Exception:  # fallback defensivo
    EmbeddingsLocal = None


# ---------------------------- Utilidades numéricas ----------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    va = float(np.linalg.norm(a))
    vb = float(np.linalg.norm(b))
    if va == 0.0 or vb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (va * vb))

def _jaccard_chars(a: str, b: str) -> float:
    # Conjunto de trigramas de caracteres (robusto a pequeñas variaciones)
    def tri(s: str) -> set:
        s = " ".join(s.lower().split())
        return {s[i:i+3] for i in range(len(s)-2)} if len(s) >= 3 else {s}
    A, B = tri(a), tri(b)
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))

def _smart_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:10]


# --------------------------- Estructuras de tracking --------------------------

@dataclass
class RetrievalItem:
    doc_id: str
    score: float
    text: str
    emb: Optional[np.ndarray] = None

@dataclass
class MemoryTrace:
    goal: str
    retrieved: List[RetrievalItem] = field(default_factory=list)
    ts_start: float = field(default_factory=time.time)

    def record_retrieval(
        self,
        ids: Iterable[str],
        texts: Iterable[str],
        scores: Iterable[float],
        embs: Optional[Iterable[np.ndarray]] = None,
    ) -> None:
        self.retrieved.clear()
        if embs is None:
            embs = [None] * len(list(ids))  # se recalcula abajo; lista vacía rompe zip
            # Ojo: rehacemos los iterables
            ids = list(ids); texts = list(texts); scores = list(scores)
        else:
            ids = list(ids); texts = list(texts); scores = list(scores); embs = list(embs)
        for i, t, s, e in zip(ids, texts, scores, embs):
            self.retrieved.append(RetrievalItem(doc_id=str(i), text=str(t), score=float(s), emb=e))


# ----------------------------- Cálculo de métricas ---------------------------

class _Encoder:
    """Wrapper para encoder local; si no existe, hace un TF-IDF muy básico."""
    def __init__(self):
        self._enc = None
        if EmbeddingsLocal is not None:
            try:
                self._enc = EmbeddingsLocal()
            except Exception as e:
                logging.warning("EmbeddingsLocal no disponible: %s", e)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._enc is not None:
            return self._enc.encode(texts)  # debe devolver np.ndarray [n, d]
        # Fallback muy simple: bolsa de trigramas de caracteres binaria
        vocab = {}
        seqs = []
        for tx in texts:
            s = " ".join(tx.lower().split())
            tris = [s[i:i+3] for i in range(max(0, len(s)-2))]
            seqs.append(tris)
            for t in tris:
                if t not in vocab:
                    vocab[t] = len(vocab)
        mat = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for r, tris in enumerate(seqs):
            for t in tris:
                mat[r, vocab[t]] = 1.0
        # normaliza filas
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        return mat / norms


def retrieval_at_k(retrieved_ids: List[str], gold_ids: Optional[List[str]], k: int = 5) -> Dict[str, Any]:
    """
    Devuelve {'hit': bool, 'rank': int|None, 'k': k}.
    Si no hay gold_ids, devuelve hit=None, rank=None (se registra como 'n/a').
    """
    if not gold_ids:
        return {"hit": None, "rank": None, "k": int(k)}
    topk = list(retrieved_ids)[:k]
    for idx, rid in enumerate(topk, start=1):
        if rid in gold_ids:
            return {"hit": True, "rank": idx, "k": int(k)}
    return {"hit": False, "rank": None, "k": int(k)}


def infer_used_fact(
    answer: str,
    retrieved: List[RetrievalItem],
    *,
    encoder: Optional[_Encoder] = None,
    alpha: float = 0.7,
    sim_threshold: float = 0.48,
    jac_threshold: float = 0.08,
) -> Dict[str, Any]:
    """
    Calcula si la respuesta usó algún hecho recuperado, combinando:
    - similitud coseno (mismo encoder para respuesta y docs)
    - solapamiento Jaccard de trigramas
    Codifica SIEMPRE [answer]+docs juntos para mantener la misma dimensionalidad.
    """
    if not retrieved:
        return {"used_fact": False, "support_score": 0.0, "best_doc_id": None}

    if encoder is None:
        encoder = _Encoder()

    doc_texts = [it.text for it in retrieved]
    try:
        mat = encoder.encode([answer] + doc_texts)  # misma base/vocab
    except Exception:
        # fallback defensivo: sin embeddings, desactiva coseno
        mat = None

    if mat is not None and isinstance(mat, np.ndarray) and len(mat) == len(doc_texts) + 1:
        ans_vec = mat[0]
        doc_vecs = mat[1:]
        # normaliza/guarda en items para futuras llamadas
        for it, v in zip(retrieved, doc_vecs):
            it.emb = v
    else:
        # si no pudimos vectorizar, usa vectores nulos (cos=0)
        ans_vec = None

    best = {"score": -1.0, "doc_id": None, "cos": 0.0, "jac": 0.0}
    for it in retrieved:
        cos = _cosine(ans_vec, it.emb) if (ans_vec is not None and it.emb is not None) else 0.0
        jac = _jaccard_chars(answer, it.text)
        score = alpha * cos + (1.0 - alpha) * jac
        if score > best["score"]:
            best = {"score": score, "doc_id": it.doc_id, "cos": cos, "jac": jac}

    used = (best["score"] >= sim_threshold) and (best["jac"] >= jac_threshold)
    return {
        "used_fact": bool(used),
        "support_score": float(best["score"]),
        "best_doc_id": best["doc_id"],
        "best_cos": float(best["cos"]),
        "best_jac": float(best["jac"]),
    }


def compute_memory_metrics(
    goal: str,
    answer: str,
    trace: MemoryTrace,
    *,
    k: int = 5,
    gold_doc_ids: Optional[List[str]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Calcula métricas clave y devuelve un dict listo para loguear o mostrar."""
    retrieved_ids = [r.doc_id for r in trace.retrieved]
    retk = retrieval_at_k(retrieved_ids, gold_doc_ids, k=k)
    used = infer_used_fact(answer, trace.retrieved)

    return {
        "run_id": run_id or _smart_hash(f"{goal}-{time.time():.0f}"),
        "goal_hash": _smart_hash(goal),
        "answer_len": len(answer),
        "retrieved": len(trace.retrieved),
        "retrieval_hit@k": retk["hit"],
        "retrieval_rank": retk["rank"],
        "retrieval_k": retk["k"],
        "used_fact": used["used_fact"],
        "support_score": round(used["support_score"], 4),
        "support_doc_id": used["best_doc_id"],
        "support_cos": round(used["best_cos"], 4),
        "support_jac": round(used["best_jac"], 4),
        "ts": int(time.time()),
    }


# ------------------------------ Logging a CSV ---------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

CSV_HEADERS = [
    "run_id","ts","goal_hash","answer_len",
    "retrieved","retrieval_k","retrieval_hit@k","retrieval_rank",
    "used_fact","support_score","support_cos","support_jac","support_doc_id"
]

def append_metrics_csv(out_dir: str, row: Dict[str, Any]) -> str:
    """Escribe/append a logs/benchmarks/run_YYYYMMDD.csv; devuelve ruta del archivo."""
    _ensure_dir(out_dir)
    day = time.strftime("%Y%m%d")
    path = os.path.join(out_dir, f"memory_metrics_{day}.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if new:
            w.writeheader()
        w.writerow({
            "run_id": row.get("run_id"),
            "ts": row.get("ts"),
            "goal_hash": row.get("goal_hash"),
            "answer_len": row.get("answer_len"),
            "retrieved": row.get("retrieved"),
            "retrieval_k": row.get("retrieval_k"),
            "retrieval_hit@k": row.get("retrieval_hit@k"),
            "retrieval_rank": row.get("retrieval_rank"),
            "used_fact": row.get("used_fact"),
            "support_score": row.get("support_score"),
            "support_cos": row.get("support_cos"),
            "support_jac": row.get("support_jac"),
            "support_doc_id": row.get("support_doc_id"),
        })
    return path


# --------------------------- Helper de integración ---------------------------

def compute_and_log_memory_metrics(
    run_id: str,
    goal: str,
    answer: str,
    trace: MemoryTrace,
    *,
    k: int = 5,
    gold_doc_ids: Optional[List[str]] = None,
    out_dir: str = "logs/benchmarks",
) -> Dict[str, Any]:
    """
    Calcula y guarda métricas de memoria en CSV. Devuelve el dict de métricas.
    """
    metrics = compute_memory_metrics(goal, answer, trace, k=k, gold_doc_ids=gold_doc_ids, run_id=run_id)
    csv_path = append_metrics_csv(out_dir, metrics)
    logging.info("Memoria/metrics → %s | used_fact=%s support=%.3f hit@%d=%s rank=%s",
                 os.path.basename(csv_path),
                 metrics["used_fact"], metrics["support_score"],
                 metrics["retrieval_k"], metrics["retrieval_hit@k"],
                 str(metrics["retrieval_rank"]))
    return metrics
