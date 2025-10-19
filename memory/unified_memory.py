# unified_memory.py
# Fase 7: Memoria vectorial con FAISS/NumPy + Olvido útil (TTL, decay, LRU) + resumen denso opcional.
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import hashlib
import statistics
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

# ---------------------------------------------------------------------------
# FAISS opcional con fallback NumPy (coseno)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - dependerá del entorno
    import faiss  # type: ignore
    _HAVE_FAISS = True
except Exception:  # pragma: no cover
    faiss = None
    _HAVE_FAISS = False


class _NPIndex:
    """Índice mínimo NumPy (coseno) para fallback cuando FAISS no está."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._mat: Optional[np.ndarray] = None  # (N, dim) float32

    @property
    def ntotal(self) -> int:
        return 0 if self._mat is None else int(self._mat.shape[0])

    def add(self, vecs: np.ndarray) -> None:
        vecs = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms
        if self._mat is None:
            self._mat = vecs
        else:
            self._mat = np.vstack([self._mat, vecs])

    def search(self, q: np.ndarray, top_k: int):
        if self._mat is None or self._mat.shape[0] == 0:
            D = np.full((1, top_k), float("inf"), dtype=np.float32)
            I = np.full((1, top_k), -1, dtype=np.int32)
            return D, I
        q = np.asarray(q, dtype=np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        sims = q @ self._mat.T  # (1, N)
        take = min(top_k, self._mat.shape[0])
        idx = np.argpartition(-sims[0], take - 1)[:take]
        idx = idx[np.argsort(-sims[0, idx])]
        D = (1.0 - sims[0, idx]).astype(np.float32).reshape(1, -1)  # 1 - sim = "dist"
        I = idx.astype(np.int32).reshape(1, -1)
        return D, I


SECONDS_DAY = 86400.0

@dataclass
class _CompatItem:
    key: str
    text: str
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    score: float = 1.0


class UnifiedMemory:
    """
    Memoria vectorial con FAISS (CPU/GPU opcional) o fallback NumPy y 'olvido útil'.

    - Determinista: embed_text(text) usa RNG con seed del hash del texto (útil para CI).
    - Persistente: vector_memories.json + faiss_index.bin (o reindexa desde memories).
    - Seguro/portable: bloqueos en mutaciones y tolerancia a errores de I/O.
    - Olvido útil: TTL, decaimiento (half-life), LRU y opción de 'resumen denso' por callback.
    """

    # -------------------- Init / IO / índice --------------------
    def __init__(
        self,
        memory_dir: Optional[str] = "memory_store",
        vector_dim: int = 1536,
        use_gpu: bool = False,
        *,
        # Olvido útil
        ttl_days: int = 0,                 # 0 = desactivado
        half_life_days: int = 14,          # decaimiento exponencial
        max_items: int = 5000,             # recorte LRU si se supera
        summarize_after_days: Optional[int] = None,  # si no es None, permite resumen denso
        summarizer: Optional[Callable[[List[str]], str]] = None,  # función externa
        # Compat con variantes anteriores:
        data_dir: Optional[str] = None,
        base_dir: Optional[str] = None,
        **_: Any,
    ):
        if data_dir:
            memory_dir = data_dir
        if base_dir and not memory_dir:
            memory_dir = str(base_dir)

        self.logger = logging.getLogger("UnifiedMemory")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)

        self.memory_dir = str(memory_dir or "memory_store")
        self.vector_dim = int(vector_dim)
        self.use_gpu = bool(use_gpu)

        # Olvido útil
        self.ttl_days = int(ttl_days)
        self.half_life_days = int(half_life_days)
        self.max_items = int(max_items)
        self.summarize_after_days = summarize_after_days
        self.summarizer = summarizer

        self.lock = threading.Lock()
        os.makedirs(self.memory_dir, exist_ok=True)
        self.index_file = os.path.join(self.memory_dir, "faiss_index.bin")
        self.vector_file = os.path.join(self.memory_dir, "vector_memories.json")

        # Cada item:
        #   {"mem_id","text","metadata","result_quality","confidence","trace_id",
        #    "created_at","last_access","score"}
        self.memories: List[Dict[str, Any]] = []
        self.items: Dict[str, _CompatItem] = {}  # API compat para tests/herramientas
        self.index: Any = None  # faiss.Index | _NPIndex

        self.counters: Dict[str, int] = {"adds_total": 0, "search_total": 0}
        self._lat_add_ms: List[float] = []
        self._lat_search_ms: List[float] = []
        self._last_trace_id: Optional[str] = None
        self._last_mem_id: Optional[str] = None

        self._load_memories()
        self._init_index()

    # Carga memories de disco y migra campos faltantes
    def _load_memories(self) -> None:
        try:
            if os.path.exists(self.vector_file):
                with open(self.vector_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.memories = data if isinstance(data, list) else []
                changed = False
                now = time.time()
                for m in self.memories:
                    if "mem_id" not in m:
                        m["mem_id"] = f"mem://{uuid4()}"; changed = True
                    m.setdefault("metadata", {})
                    m.setdefault("result_quality", "unknown")
                    m.setdefault("confidence", 0.5)
                    m.setdefault("trace_id", str(uuid4()))
                    m.setdefault("created_at", now)
                    m.setdefault("last_access", now)
                    m.setdefault("score", 1.0)
                    # poblar estructura compat
                    self.items[m["mem_id"]] = _CompatItem(
                        key=m["mem_id"],
                        text=m.get("text", ""),
                        metadata=m.get("metadata", {}),
                        created_at=m.get("created_at", now),
                        last_access=m.get("last_access", now),
                        score=float(m.get("score", 1.0)),
                    )
                if changed:
                    with open(self.vector_file, "w", encoding="utf-8") as f:
                        json.dump(self.memories, f, ensure_ascii=False, indent=2)
                self.logger.info(f"[UM] Cargadas {len(self.memories)} memorias.")
            else:
                self.memories = []
        except Exception as e:
            self.logger.error(f"[UM] Error cargando memorias: {e}")
            self.memories = []

    def _new_index_faiss(self):
        if not _HAVE_FAISS:  # pragma: no cover
            return _NPIndex(self.vector_dim)
        idx = faiss.IndexFlatL2(self.vector_dim)  # type: ignore
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()  # type: ignore
                idx = faiss.index_cpu_to_gpu(res, 0, idx)  # type: ignore
            except Exception as e:
                self.logger.warning(f"[UM] GPU no disponible para FAISS: {e}")
        return idx

    def _init_index(self) -> None:
        """Crea/carga índice y garantiza que #vectores == #memories."""
        try:
            if _HAVE_FAISS and os.path.exists(self.index_file):
                idx = faiss.read_index(self.index_file)  # type: ignore
                if int(idx.d) != self.vector_dim:
                    self.logger.warning("[UM] Dim mismatch en índice; se reconstruye.")
                    self.index = self._new_index_faiss()
                    self._rebuild_index_from_memories()
                else:
                    self.index = idx
                    if self.use_gpu:
                        try:
                            res = faiss.StandardGpuResources()  # type: ignore
                            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)  # type: ignore
                        except Exception as e:
                            self.logger.warning(f"[UM] GPU no disponible para FAISS: {e}")
            else:
                self.index = self._new_index_faiss() if _HAVE_FAISS else _NPIndex(self.vector_dim)
                self._rebuild_index_from_memories()
        except Exception as e:
            self.logger.error(f"[UM] Error inicializando índice: {e}")
            self.index = _NPIndex(self.vector_dim)
            self._rebuild_index_from_memories()

        try:
            nt = int(getattr(self.index, "ntotal", 0))
        except Exception:
            nt = 0
        if nt != len(self.memories):
            self.logger.info("[UM] Realineando índice con memories…")
            self._rebuild_index_from_memories()

    # Reconstruye el índice desde self.memories
    def _rebuild_index_from_memories(self) -> None:
        with self.lock:
            self.index = self._new_index_faiss() if _HAVE_FAISS else _NPIndex(self.vector_dim)
            if not self.memories:
                return
            batch: List[np.ndarray] = []
            for m in self.memories:
                vec = self.embed_text(m.get("text", ""))
                batch.append(vec)
                if len(batch) >= 1024:
                    self.index.add(np.asarray(batch, dtype=np.float32))  # type: ignore
                    batch.clear()
            if batch:
                self.index.add(np.asarray(batch, dtype=np.float32))  # type: ignore

    def save_to_disk(self) -> None:
        try:
            with open(self.vector_file, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
            if _HAVE_FAISS and self.index is not None:
                try:
                    idx_cpu = self.index
                    if hasattr(faiss, "index_gpu_to_cpu"):  # type: ignore
                        try:
                            idx_cpu = faiss.index_gpu_to_cpu(self.index)  # type: ignore
                        except Exception:
                            idx_cpu = self.index
                    faiss.write_index(idx_cpu, self.index_file)  # type: ignore
                except Exception as e:  # pragma: no cover
                    self.logger.warning(f"[UM] No se pudo escribir índice FAISS: {e}")
        except Exception as e:
            self.logger.error(f"[UM] Error guardando: {e}")

    # -------------------- Embeddings / Métricas --------------------
    def embed_text(self, text: str) -> np.ndarray:
        # Determinista por hash; suficiente para pruebas/CI
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(self.vector_dim, dtype=np.float32)

    @staticmethod
    def _avg(x: List[float]) -> Optional[float]:
        try:
            return float(statistics.fmean(x)) if x else None
        except Exception:
            return None

    @staticmethod
    def _p95(x: List[float]) -> Optional[float]:
        if not x:
            return None
        s = sorted(x)
        k = max(0, math.ceil(0.95 * len(s)) - 1)
        return float(s[k])

    # -------------------- Altas / Búsquedas --------------------
    def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        result_quality: str = "unknown",
        confidence: float = 0.5,
        trace_id: Optional[str] = None,
    ) -> str:
        """Inserta UNA memoria y devuelve su mem_id."""
        t0 = perf_counter()
        try:
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()
            if not text:
                return ""
            emb = self.embed_text(text)
            now = time.time()
            tid = trace_id or str(uuid4())
            mid = f"mem://{uuid4()}"

            with self.lock:
                self.index.add(np.array([emb], dtype=np.float32))  # type: ignore
                row = {
                    "mem_id": mid,
                    "text": text,
                    "metadata": metadata or {},
                    "result_quality": str(result_quality),
                    "confidence": float(confidence),
                    "trace_id": tid,
                    "created_at": now,
                    "last_access": now,
                    "score": 1.0,
                }
                self.memories.append(row)
                # compat map
                self.items[mid] = _CompatItem(key=mid, text=text, metadata=row["metadata"],
                                              created_at=now, last_access=now, score=1.0)
                self.counters["adds_total"] += 1
                self._last_trace_id = tid
                self._last_mem_id = mid

            self.save_to_disk()
            return mid
        except Exception as e:
            self.logger.error(f"[UM] Error agregando memoria: {e}")
            return ""
        finally:
            self._lat_add_ms.append((perf_counter() - t0) * 1000.0)

    def add_memories_bulk(self, items: List[Dict[str, Any]]) -> int:
        t0 = perf_counter()
        added = 0
        try:
            vecs, rows = [], []
            now = time.time()
            for it in items or []:
                text = str(it.get("text", "")).strip()
                if not text:
                    continue
                emb = self.embed_text(text)
                vecs.append(emb)
                row = {
                    "mem_id": it.get("mem_id") or f"mem://{uuid4()}",
                    "text": text,
                    "metadata": it.get("metadata") or {},
                    "result_quality": str(it.get("result_quality", "unknown")),
                    "confidence": float(it.get("confidence", 0.5)),
                    "trace_id": it.get("trace_id") or str(uuid4()),
                    "created_at": now,
                    "last_access": now,
                    "score": 1.0,
                }
                rows.append(row)
            if not rows:
                return 0
            with self.lock:
                self.index.add(np.asarray(vecs, dtype=np.float32))  # type: ignore
                self.memories.extend(rows)
                for r in rows:
                    self.items[r["mem_id"]] = _CompatItem(key=r["mem_id"], text=r["text"],
                                                          metadata=r["metadata"], created_at=r["created_at"],
                                                          last_access=r["last_access"], score=1.0)
                added = len(rows)
                self.counters["adds_total"] += added
                self._last_mem_id = rows[-1]["mem_id"]; self._last_trace_id = rows[-1]["trace_id"]
            self.save_to_disk()
            return added
        except Exception as e:
            self.logger.error(f"[UM] Error en add_memories_bulk: {e}")
            return 0
        finally:
            self._lat_add_ms.append((perf_counter() - t0) * 1000.0)

    def search_memory(self, query_vector, top_k: int = 5) -> List[Dict[str, Any]]:
        t0 = perf_counter()
        try:
            if self.index is None:
                return []
            if isinstance(query_vector, dict):
                query_vector = query_vector.get("input", "")
            if isinstance(query_vector, str):
                query_vector = self.embed_text(query_vector)
            q = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            D, I = self.index.search(q, int(top_k))  # type: ignore

            out: List[Dict[str, Any]] = []
            if I is None or D is None:
                return out
            now = time.time()
            for idx, dist in zip(I[0], D[0]):
                if idx != -1 and 0 <= idx < len(self.memories):
                    m = self.memories[idx]
                    # refrescar 'last_access' y subir ligeramente 'score'
                    m["last_access"] = now
                    m["score"] = float(m.get("score", 1.0)) + 0.05
                    # sync compat
                    if m["mem_id"] in self.items:
                        self.items[m["mem_id"]].last_access = now
                        self.items[m["mem_id"]].score = m["score"]
                    out.append(
                        {
                            "mem_id": m["mem_id"],
                            "citation_id": m["mem_id"],
                            "text": m["text"],
                            "metadata": m.get("metadata", {}),
                            "distance": float(dist),
                            "result_quality": m.get("result_quality", "unknown"),
                            "confidence": float(m.get("confidence", 0.5)),
                            "trace_id": m.get("trace_id"),
                        }
                    )
            return out
        except Exception as e:
            self.logger.error(f"[UM] Error search_memory: {e}")
            return []
        finally:
            self.counters["search_total"] += 1
            self._lat_search_ms.append((perf_counter() - t0) * 1000.0)

    # -------------------- API compatible / helpers --------------------
    def upsert(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Compat: genera id por hash si no se provee y añade como nueva memoria."""
        text = (text or "").strip()
        if not text:
            return ""
        key = (metadata or {}).get("id") if metadata and "id" in metadata else hashlib.sha1(text.encode("utf-8")).hexdigest()
        # Para mantener alineado el índice, tratamos 'upsert' como add si no existe:
        if key in self.items:
            # Si ya existe, añadimos una nueva versión y mantenemos ambas (historia).
            pass
        return self.add_memory(text, metadata=metadata)

    def add_interaction(self, user_text: str, assistant_text: Optional[str] = None, **labels):
        self.add_memory(user_text, metadata={"type": "interaction", "role": "user"}, **labels)
        if assistant_text:
            self.add_memory(assistant_text, metadata={"type": "interaction", "role": "assistant"}, **labels)

    def get_context(self, limit: int = 10) -> List[Tuple[str, str]]:
        items: List[Tuple[str, str]] = []
        for m in reversed(self.memories):
            md = m.get("metadata", {})
            if md.get("type") == "interaction":
                items.append((md.get("role", "user"), m.get("text", "")))
                if len(items) >= limit:
                    break
        items.reverse()
        return items

    def retrieve_relevant_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.search_memory(query, top_k=top_k)
        out: List[Dict[str, Any]] = []
        for r in results:
            score = 1.0 / (1.0 + float(r.get("distance", 0.0)))
            out.append(
                {
                    "mem_id": r["mem_id"],
                    "citation_id": r["citation_id"],
                    "text": r["text"],
                    "score": score,
                    "metadata": r.get("metadata", {}),
                    "source": "vector",
                    "result_quality": r.get("result_quality", "unknown"),
                    "confidence": float(r.get("confidence", 0.5)),
                    "trace_id": r.get("trace_id"),
                }
            )
        return out

    # -------------------- Olvido útil / mantenimiento --------------------
    def _decay_factor(self, delta_seconds: float) -> float:
        if self.half_life_days <= 0:
            return 1.0
        return 0.5 ** (delta_seconds / (self.half_life_days * SECONDS_DAY))

    def decay_scores(self):
        if self.half_life_days <= 0:
            return
        now = time.time()
        for m in self.memories:
            dt = now - float(m.get("last_access", now))
            factor = max(0.01, self._decay_factor(dt))
            m["score"] = float(m.get("score", 1.0)) * factor
            if m["mem_id"] in self.items:
                self.items[m["mem_id"]].score = m["score"]

    def purge_ttl(self):
        if self.ttl_days <= 0:
            return
        ttl = self.ttl_days * SECONDS_DAY
        now = time.time()
        before = len(self.memories)
        self.memories = [m for m in self.memories if (now - float(m.get("created_at", now))) <= ttl]
        # sync compat
        keep_ids = {m["mem_id"] for m in self.memories}
        self.items = {k: v for k, v in self.items.items() if k in keep_ids}
        if len(self.memories) != before:
            self._rebuild_index_from_memories()
            self.save_to_disk()

    def purge_lru(self):
        if self.max_items <= 0 or len(self.memories) <= self.max_items:
            return
        ranked = sorted(self.memories, key=lambda x: (float(x.get("score", 1.0)), -float(x.get("last_access", 0.0))))
        overflow = len(self.memories) - self.max_items
        drop = set(m["mem_id"] for m in ranked[:overflow])
        self.memories = [m for m in self.memories if m["mem_id"] not in drop]
        # sync compat
        self.items = {k: v for k, v in self.items.items() if k not in drop}
        self._rebuild_index_from_memories()
        self.save_to_disk()

    def summarize_older(self, older_than_days: int = 30, batch_size: int = 20) -> Optional[str]:
        if not self.summarizer:
            return None
        now = time.time()
        cands = [m for m in self.memories if (now - float(m.get("created_at", now))) > older_than_days * SECONDS_DAY]
        if not cands:
            return None
        texts = [m["text"] for m in sorted(cands, key=lambda x: x.get("created_at", 0))[:batch_size]]
        summary = self.summarizer(texts)
        sid = self.add_memory(f"[RESUMEN DENSO]\n{summary}", metadata={"kind": "summary", "older_than_days": older_than_days})
        # elimina originales resumidos
        drop = set(m["mem_id"] for m in cands[:batch_size])
        self.memories = [m for m in self.memories if m["mem_id"] not in drop]
        self.items = {k: v for k, v in self.items.items() if k not in drop}
        self._rebuild_index_from_memories()
        self.save_to_disk()
        return sid

    def maintenance(self):
        """Ejecuta mantenimiento estándar: decay → TTL → LRU → (resumen opcional)."""
        self.decay_scores()
        self.purge_ttl()
        self.purge_lru()
        if self.summarize_after_days is not None:
            try:
                self.summarize_older(self.summarize_after_days)
            except Exception:
                pass
        self.save_to_disk()

    # -------------------- Administrativas / otras --------------------
    def delete_memories(self, mem_ids: List[str]) -> int:
        if not mem_ids:
            return 0
        mem_set = set(mem_ids)
        before = len(self.memories)
        self.memories = [m for m in self.memories if m.get("mem_id") not in mem_set]
        self.items = {k: v for k, v in self.items.items() if k not in mem_set}
        removed = before - len(self.memories)
        if removed > 0:
            self._rebuild_index_from_memories()
            self.save_to_disk()
        return removed

    def compact_duplicates(self) -> int:
        seen, out = set(), []
        for m in self.memories:
            key = m.get("mem_id") or f"txt:{m.get('text','')}"
            if key in seen:
                continue
            seen.add(key)
            out.append(m)
        removed = len(self.memories) - len(out)
        if removed:
            self.memories = out
            self.items = {m["mem_id"]: self.items.get(m["mem_id"], _CompatItem(m["mem_id"], m["text"], m.get("metadata", {}))) for m in out}
            self._rebuild_index_from_memories()
            self.save_to_disk()
        return removed

    def reindex(self) -> None:
        self._rebuild_index_from_memories()
        self.save_to_disk()

    # -------------------- Estado / utilidades --------------------
    def get_status(self) -> Dict[str, Any]:
        try:
            index_ntotal = int(getattr(self.index, "ntotal", 0))
        except Exception:
            index_ntotal = 0

        reflections = sum(1 for m in self.memories if m.get("metadata", {}).get("type") == "reflection")
        events = sum(1 for m in self.memories if m.get("metadata", {}).get("type") == "event")
        interactions = sum(1 for m in self.memories if m.get("metadata", {}).get("type") == "interaction")

        return {
            "ok": True,
            "status": "ready",
            "memory_dir": self.memory_dir,
            "memories": len(self.memories),
            "index_size": index_ntotal,
            "vector_dim": self.vector_dim,
            "use_gpu": bool(self.use_gpu),
            "last_trace_id": self._last_trace_id,
            "last_mem_id": self._last_mem_id,
            "reflections_stored": reflections,
            "events_stored": events,
            "interactions_stored": interactions,
            "counters": dict(self.counters),
            "latency_ms": {
                "add_avg": self._avg(self._lat_add_ms),
                "add_p95": self._p95(self._lat_add_ms),
                "search_avg": self._avg(self._lat_search_ms),
                "search_p95": self._p95(self._lat_search_ms),
            },
            "forgetting": {
                "ttl_days": self.ttl_days,
                "half_life_days": self.half_life_days,
                "max_items": self.max_items,
                "summarize_after_days": self.summarize_after_days,
            },
        }

    def get_memory_count(self) -> int:
        return len(self.memories)

    def export_memories(self) -> List[Dict[str, Any]]:
        return self.memories
