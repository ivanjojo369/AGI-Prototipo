# memory/metrics.py
# Métricas para Memoria Semántica “pro”
# Conserva todo lo existente (encoder local, retrieval@k, used_fact, CSV con ts)
# y añade: recall@k, mrr@k, ndcg@k (+ columnas nuevas en el CSV).

from __future__ import annotations
import os, csv, time, math, hashlib, logging, re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterable

import numpy as np

# Intentamos usar el mismo encoder local que el vector store
try:
    from memory.embeddings_local import EmbeddingsLocal  # tu clase local
except Exception:  # fallback defensivo
    EmbeddingsLocal = None


# ---------------------------- Utilidades numéricas ----------------------------

def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
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
        # Aseguramos listas para poder iterar más de una vez
        ids = list(ids); texts = list(texts); scores = list(scores)
        if embs is None:
            embs = [None] * len(ids)
        else:
            embs = list(embs)
        for i, t, s, e in zip(ids, texts, scores, embs):
            self.retrieved.append(RetrievalItem(doc_id=str(i), text=str(t), score=float(s), emb=e))


# ----------------------------- Cálculo de métricas ---------------------------

class _Encoder:
    """Wrapper para encoder local; si no existe, hace un TF-IDF/BoW simple de trigramas."""
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
        # Fallback muy simple: bolsa de trigramas binaria normalizada por fila
        vocab: Dict[str, int] = {}
        seqs: List[List[str]] = []
        for tx in texts:
            s = " ".join(tx.lower().split())
            tris = [s[i:i+3] for i in range(max(0, len(s)-2))]
            seqs.append(tris)
            for t in tris:
                if t not in vocab:
                    vocab[t] = len(vocab)
        if not vocab:
            return np.zeros((len(texts), 1), dtype=np.float32)
        mat = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for r, tris in enumerate(seqs):
            for t in tris:
                mat[r, vocab[t]] = 1.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        return mat / norms


def retrieval_at_k(retrieved_ids: List[str], gold_ids: Optional[List[str]], k: int = 5) -> Dict[str, Any]:
    """
    Devuelve {'hit': bool|None, 'rank': int|None, 'k': k}.
    Si no hay gold_ids, devuelve hit=None, rank=None (se registra como 'n/a').
    """
    if not gold_ids:
        return {"hit": None, "rank": None, "k": int(k)}
    topk = list(retrieved_ids)[:k]
    for idx, rid in enumerate(topk, start=1):
        if rid in set(gold_ids):
            return {"hit": True, "rank": idx, "k": int(k)}
    return {"hit": False, "rank": None, "k": int(k)}


# ---- “Usó el hecho”: mezcla coseno + Jaccard; heurística robusta a respuestas textuales

_NUM_RE = re.compile(r"\b\d{3,4}\b")  # útil para años tipo 1781

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
    Además, si hay números (años) en la respuesta y en el doc relevante, ayuda a la decisión.
    """
    if not retrieved:
        return {"used_fact": False, "support_score": 0.0, "best_doc_id": None, "best_cos": 0.0, "best_jac": 0.0}

    if encoder is None:
        encoder = _Encoder()

    doc_texts = [it.text for it in retrieved]
    try:
        mat = encoder.encode([answer] + doc_texts)  # misma base/vocab
    except Exception:
        mat = None

    ans_vec: Optional[np.ndarray] = None
    if isinstance(mat, np.ndarray) and len(mat) == len(doc_texts) + 1:
        ans_vec = mat[0]
        doc_vecs = mat[1:]
        for it, v in zip(retrieved, doc_vecs):
            it.emb = v

    best = {"score": -1.0, "doc_id": None, "cos": 0.0, "jac": 0.0}
    for it in retrieved:
        cos = _cosine(ans_vec, it.emb)
        jac = _jaccard_chars(answer, it.text)
        score = alpha * cos + (1.0 - alpha) * jac
        if score > best["score"]:
            best = {"score": score, "doc_id": it.doc_id, "cos": cos, "jac": jac}

    # Señal extra: coincidencia de números (años) entre respuesta y doc top
    ans_nums = set(_NUM_RE.findall(answer or ""))
    doc_nums = set(_NUM_RE.findall(retrieved[0].text or "")) if retrieved else set()
    numeric_boost = 0.05 if (ans_nums & doc_nums) else 0.0
    boosted = best["score"] + numeric_boost

    used = (boosted >= sim_threshold) and (best["jac"] >= jac_threshold)
    return {
        "used_fact": bool(used),
        "support_score": float(boosted),
        "best_doc_id": best["doc_id"],
        "best_cos": float(best["cos"]),
        "best_jac": float(best["jac"]),
    }


# ---- Métricas clásicas de ranking

def _topk_ids(items: List[RetrievalItem], k: int) -> List[str]:
    return [it.doc_id for it in items[:max(1, int(k))]]

def _recall_at_k(topk_ids: List[str], gold_ids: Iterable[str]) -> float:
    gold = set(gold_ids or [])
    if not gold:
        return 0.0
    hit = len(set(topk_ids) & gold)
    return hit / float(len(gold))

def _mrr_at_k(topk_ids: List[str], gold_ids: Iterable[str]) -> float:
    gold = set(gold_ids or [])
    for idx, did in enumerate(topk_ids, start=1):
        if did in gold:
            return 1.0 / float(idx)
    return 0.0

def _ndcg_at_k(topk_ids: List[str], gold_ids: Iterable[str]) -> float:
    gold = set(gold_ids or [])
    if not gold:
        return 0.0
    dcg = 0.0
    for i, did in enumerate(topk_ids, start=1):
        rel = 1.0 if did in gold else 0.0
        if rel > 0.0:
            dcg += rel / math.log2(i + 1.0)  # i=1 -> log2(2)=1
    ideal_hits = min(len(gold), len(topk_ids))
    idcg = sum(1.0 / math.log2(i + 1.0) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


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

    # Métricas de “hit” y rango
    retk = retrieval_at_k(retrieved_ids, gold_doc_ids, k=k)

    # Métrica de “usó el hecho”
    used = infer_used_fact(answer, trace.retrieved)

    # Métricas de ranking clásicas
    topk_ids = _topk_ids(trace.retrieved, k)
    recall_k = _recall_at_k(topk_ids, gold_doc_ids or [])
    mrr_k = _mrr_at_k(topk_ids, gold_doc_ids or [])
    ndcg_k = _ndcg_at_k(topk_ids, gold_doc_ids or [])

    # Salida compatible + extendida
    return {
        # IDs y hashes útiles
        "run_id": run_id or _smart_hash(f"{goal}-{time.time():.0f}"),
        "goal_hash": _smart_hash(goal),

        # Medidas generales
        "answer_len": len(answer),
        "retrieved": len(trace.retrieved),

        # Métricas de recuperación existentes
        "retrieval_hit@k": retk["hit"],
        "retrieval_rank": retk["rank"],
        "retrieval_k": retk["k"],

        # NUEVAS métricas de ranking
        "recall@k": float(recall_k),
        "mrr@k": float(mrr_k),
        "ndcg@k": float(ndcg_k),

        # Señal de uso de hecho (compatibilidad)
        "used_fact": used["used_fact"],
        "support_score": round(used["support_score"], 4),
        "support_doc_id": used["best_doc_id"],
        "support_cos": round(used["best_cos"], 4),
        "support_jac": round(used["best_jac"], 4),

        # Para inspección/debug
        "gold_doc_ids": list(gold_doc_ids or []),
        "topk_doc_ids": topk_ids,
        "topk_scores": [float(it.score) for it in trace.retrieved[:max(1, int(k))]],

        # Timestamps
        "ts": int(time.time()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ------------------------------ Logging a CSV ---------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

CSV_HEADERS = [
    # Compatibilidad previa
    "run_id","ts","goal_hash","answer_len","retrieved",
    "retrieval_k","retrieval_hit@k","retrieval_rank",
    "used_fact","support_score","support_cos","support_jac","support_doc_id",
    # Extensiones nuevas
    "timestamp","recall@k","mrr@k","ndcg@k",
    "gold_doc_ids","topk_doc_ids","topk_scores",
]

def append_metrics_csv(out_dir: str, row: Dict[str, Any]) -> str:
    """Escribe/append a logs/benchmarks/memory_metrics_YYYYMMDD.csv; devuelve ruta del archivo."""
    _ensure_dir(out_dir)
    day = time.strftime("%Y%m%d")
    path = os.path.join(out_dir, f"memory_metrics_{day}.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if new:
            w.writeheader()
        w.writerow({
            # Base/compat
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
            # Nuevas
            "timestamp": row.get("timestamp"),
            "recall@k": row.get("recall@k"),
            "mrr@k": row.get("mrr@k"),
            "ndcg@k": row.get("ndcg@k"),
            "gold_doc_ids": "|".join(row.get("gold_doc_ids", []) or []),
            "topk_doc_ids": "|".join(row.get("topk_doc_ids", []) or []),
            "topk_scores": "|".join(str(x) for x in (row.get("topk_scores") or [])),
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
    metrics = compute_memory_metrics(
        goal=goal, answer=answer, trace=trace, k=k, gold_doc_ids=gold_doc_ids, run_id=run_id
    )
    csv_path = append_metrics_csv(out_dir, metrics)
    logging.info(
        "Memoria/metrics → %s | used_fact=%s support=%.3f hit@%d=%s rank=%s r@k=%.3f mrr@k=%.3f ndcg@k=%.3f",
        os.path.basename(csv_path),
        metrics["used_fact"], metrics["support_score"],
        metrics["retrieval_k"], metrics["retrieval_hit@k"],
        str(metrics["retrieval_rank"]),
        metrics["recall@k"], metrics["mrr@k"], metrics["ndcg@k"]
    )
    # Devolvemos también la ruta del CSV por conveniencia
    out = dict(metrics)
    out["csv_path"] = csv_path
    return out
