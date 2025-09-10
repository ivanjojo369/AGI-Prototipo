# memory/unified_memory.py
import os
import json
import faiss
import numpy as np
import logging
import threading
import math
import statistics
from uuid import uuid4
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple


class UnifiedMemory:
    """
    Memoria vectorial simple respaldada por FAISS.

    Compatibilidad con tus tests:
    - __init__ acepta alias: data_dir / base_dir además de memory_dir.
    - Interacciones:  add_interaction(), get_context(limit) -> List[Tuple[role, content]]
    - Reflexiones:    store_reflection(title, content=None),
                      retrieve_recent_reflections(limit) -> List[dict] (incluye 'contenido')
    - Eventos:        store_event(...), get_recent_events(limit) -> List[dict] (incluye 'contenido')
    - Vector:         add_to_vector_memory(text), search_vector_memory(query),
                      vector_search(query, top_k), retrieve_relevant_memories(query, top_k)
    - Estado:         get_status() -> incluye contadores de adds/búsquedas, latencias y tamaños
    - Utilidades:     get_memory_count(), export_memories(), save_to_disk()

    Novedades Fase 1:
    - Cada recuerdo queda etiquetado con: result_quality, confidence, trace_id.
    - get_status() publica contadores y latencias (avg/p95) de add/search.
    """

    # --------------------------- Init / Paths ---------------------------
    def __init__(
        self,
        memory_dir: Optional[str] = "memory_store",
        vector_dim: int = 1536,
        use_gpu: bool = False,
        *,
        data_dir: Optional[str] = None,
        base_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        # Aliases aceptados por los tests
        if data_dir:
            memory_dir = data_dir
        if base_dir and not memory_dir:
            memory_dir = str(base_dir)

        # Logger seguro
        self.logger = logging.getLogger("UnifiedMemory")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)

        self.memory_dir = str(memory_dir or "memory_store")
        self.vector_dim = vector_dim
        self.use_gpu = use_gpu
        self.lock = threading.Lock()

        os.makedirs(self.memory_dir, exist_ok=True)
        self.index_file = os.path.join(self.memory_dir, "faiss_index.bin")
        self.vector_file = os.path.join(self.memory_dir, "vector_memories.json")

        # cada item: {"text": str, "metadata": dict,
        #             "result_quality": str, "confidence": float, "trace_id": str}
        self.memories: List[Dict[str, Any]] = []
        self.index = None

        # Métricas/contadores de operación
        self.counters: Dict[str, int] = {
            "adds_total": 0,
            "search_total": 0,
        }
        self._lat_add_ms: List[float] = []
        self._lat_search_ms: List[float] = []
        self._last_trace_id: Optional[str] = None

        self._load_memories()
        self._init_faiss_index()

    # --------------------------- FAISS ---------------------------
    def _init_faiss_index(self):
        try:
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
                self.logger.info("[UM] Índice FAISS cargado correctamente.")
            else:
                self.index = faiss.IndexFlatL2(self.vector_dim)
                self.logger.info("[UM] Nuevo índice FAISS creado.")

            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    self.logger.info("[UM] FAISS usando GPU.")
                except Exception as e:
                    self.logger.warning(f"[UM] No se pudo usar GPU para FAISS: {e}")
        except Exception as e:
            self.logger.error(f"[UM] Error inicializando FAISS: {e}")
            self.index = faiss.IndexFlatL2(self.vector_dim)

    # --------------------------- IO ---------------------------
    def _load_memories(self):
        try:
            if os.path.exists(self.vector_file):
                with open(self.vector_file, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)
                # Migración suave: añade campos Fase 1 si faltan
                for m in self.memories:
                    m.setdefault("metadata", {})
                    m.setdefault("result_quality", "unknown")
                    m.setdefault("confidence", 0.5)
                    m.setdefault("trace_id", str(uuid4()))
                self.logger.info(f"[UM] {len(self.memories)} memorias cargadas desde disco.")
            else:
                self.memories = []
        except Exception as e:
            self.logger.error(f"[UM] Error cargando memorias: {e}")
            self.memories = []

    def save_to_disk(self):
        try:
            with open(self.vector_file, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
            if self.index is not None:
                try:
                    # si estuviera en GPU, lo pasamos a CPU antes de guardar
                    self.index = faiss.index_gpu_to_cpu(self.index)
                except Exception:
                    pass
                faiss.write_index(self.index, self.index_file)
            self.logger.info("[UM] Memorias y FAISS guardados en disco.")
        except Exception as e:
            self.logger.error(f"[UM] Error guardando memorias: {e}")

    # --------------------------- Embeddings ---------------------------
    def embed_text(self, text: str) -> np.ndarray:
        """
        Stub determinístico (semilla por hash) para dar estabilidad a los tests.
        Sustitúyelo por tu modelo real si lo necesitas.
        """
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(self.vector_dim, dtype=np.float32)

    # --------------------------- Helpers de latencia ---------------------------
    @staticmethod
    def _avg(x: List[float]) -> Optional[float]:
        return float(statistics.fmean(x)) if x else None

    @staticmethod
    def _p95(x: List[float]) -> Optional[float]:
        if not x:
            return None
        s = sorted(x)
        k = max(0, math.ceil(0.95 * len(s)) - 1)
        return float(s[k])

    # --------------------------- Altas / Búsquedas ---------------------------
    def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        result_quality: str = "unknown",
        confidence: float = 0.5,
        trace_id: Optional[str] = None,
    ) -> int:
        """
        Inserta una memoria vectorial etiquetada.
        """
        t0 = perf_counter()
        try:
            if not isinstance(text, str):
                text = str(text)
            emb = self.embed_text(text)
            tid = trace_id or str(uuid4())
            with self.lock:
                self.index.add(np.array([emb], dtype=np.float32))
                self.memories.append(
                    {
                        "text": text,
                        "metadata": metadata or {},
                        "result_quality": str(result_quality),
                        "confidence": float(confidence),
                        "trace_id": tid,
                    }
                )
                idx = len(self.memories) - 1
                self.counters["adds_total"] += 1
                self._last_trace_id = tid
            self.save_to_disk()
            self.logger.info(f"[UM] Memoria agregada: {text[:50]}")
            return idx
        except Exception as e:
            self.logger.error(f"[UM] Error agregando memoria: {e}")
            return -1
        finally:
            self._lat_add_ms.append((perf_counter() - t0) * 1000.0)

    def search_memory(self, query_vector, top_k: int = 5) -> List[Dict[str, Any]]:
        t0 = perf_counter()
        try:
            if isinstance(query_vector, dict):
                query_vector = query_vector.get("input", "")
            if isinstance(query_vector, str):
                query_vector = self.embed_text(query_vector)
            query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            distances, indices = self.index.search(query_vector, top_k)
            out: List[Dict[str, Any]] = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(self.memories):
                    m = self.memories[idx]
                    out.append(
                        {
                            "text": m["text"],
                            "metadata": m.get("metadata", {}),
                            "distance": float(dist),
                            # etiquetas Fase 1:
                            "result_quality": m.get("result_quality", "unknown"),
                            "confidence": float(m.get("confidence", 0.5)),
                            "trace_id": m.get("trace_id"),
                        }
                    )
            return out
        except Exception as e:
            self.logger.error(f"[UM] Error en search_memory: {e}")
            return []
        finally:
            self.counters["search_total"] += 1
            self._lat_search_ms.append((perf_counter() - t0) * 1000.0)

    # --------------------------- API de compatibilidad ---------------------------
    # Interacciones
    def add_interaction(self, user_text: str, assistant_text: Optional[str] = None, **labels):
        """
        labels puede incluir: result_quality, confidence, trace_id
        """
        self.add_memory(user_text, metadata={"type": "interaction", "role": "user"}, **labels)
        if assistant_text:
            self.add_memory(assistant_text, metadata={"type": "interaction", "role": "assistant"}, **labels)

    def get_context(self, limit: int = 10) -> List[Tuple[str, str]]:
        """
        Devuelve las últimas 'limit' interacciones como lista de TUPLAS
        (role, content), para que los tests puedan hacer context[0][1].
        """
        items: List[Tuple[str, str]] = []
        for m in reversed(self.memories):
            md = m.get("metadata", {})
            if md.get("type") == "interaction":
                items.append((md.get("role", "user"), m.get("text", "")))
                if len(items) >= limit:
                    break
        items.reverse()
        return items

    # Reflexiones
    def store_reflection(self, title: str, content: Optional[Any] = None, **labels) -> int:
        text = f"{title}: {content}" if content is not None else str(title)
        return self.add_memory(text, metadata={"type": "reflection", "title": title}, **labels)

    def retrieve_recent_reflections(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Devuelve reflexiones recientes (más nueva primero) como dicts.
        Incluye alias esperados por los tests: 'contenido' y 'titulo'.
        """
        out: List[Dict[str, Any]] = []
        for m in reversed(self.memories):
            md = m.get("metadata", {})
            if md.get("type") == "reflection":
                item = {
                    "title": md.get("title", ""),
                    "titulo": md.get("title", ""),      # alias
                    "text": m.get("text", ""),
                    "contenido": m.get("text", ""),     # alias requerido
                    "metadata": md,
                    # devolvemos también etiquetas
                    "result_quality": m.get("result_quality", "unknown"),
                    "confidence": float(m.get("confidence", 0.5)),
                    "trace_id": m.get("trace_id"),
                }
                out.append(item)
                if len(out) >= limit:
                    break
        return out

    # Eventos
    def store_event(self, *args, **kwargs) -> int:
        """
        Acepta:
          - store_event('tipo', 'texto', **labels)
          - store_event({'text': '...', 'metadata': {...}}, **labels)
          - store_event(text='...', metadata={...}, **labels)

        Donde **labels puede incluir: result_quality, confidence, trace_id
        """
        labels = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in ("result_quality", "confidence", "trace_id")}

        if len(args) >= 2 and isinstance(args[0], str):
            event_type, text = args[0], args[1]
            meta = {"type": "event", "event_type": event_type}
            return self.add_memory(str(text), metadata=meta, **labels)

        if len(args) == 1 and isinstance(args[0], dict):
            event = args[0]
            meta = {"type": "event", **(event.get("metadata", {}) if isinstance(event, dict) else {})}
            text = event.get("text") if isinstance(event, dict) else str(event)
            return self.add_memory(text or "event", metadata=meta, **labels)

        text = kwargs.get("text", "event")
        meta = {"type": "event", **kwargs.get("metadata", {})}
        return self.add_memory(str(text), metadata=meta, **labels)

    def get_recent_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Devuelve eventos recientes (más nuevo primero) como diccionarios.
        Incluye alias esperados por los tests: 'contenido' y 'tipo'.
        """
        out: List[Dict[str, Any]] = []
        for m in reversed(self.memories):
            md = m.get("metadata", {})
            if md.get("type") == "event":
                etype = md.get("event_type") or md.get("type") or "event"
                item = {
                    "event_type": etype,
                    "tipo": etype,                      # alias
                    "text": m.get("text", ""),
                    "contenido": m.get("text", ""),     # alias requerido
                    "metadata": md,
                    # etiquetas
                    "result_quality": m.get("result_quality", "unknown"),
                    "confidence": float(m.get("confidence", 0.5)),
                    "trace_id": m.get("trace_id"),
                }
                out.append(item)
                if len(out) >= limit:
                    break
        return out

    # Vector (aliases usados por tests)
    def add_to_vector_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None, **labels) -> int:
        return self.add_memory(text, metadata=metadata, **labels)

    def search_vector_memory(self, query: str, top_k: int = 5):
        return self.vector_search(query, top_k=top_k)

    def vector_search(self, query: str, top_k: int = 5):
        return self.search_memory(query, top_k=top_k)

    def retrieve_relevant_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.search_memory(query, top_k=top_k)
        out: List[Dict[str, Any]] = []
        for r in results:
            # distancia L2 -> score en [0,1) aproximado
            score = 1.0 / (1.0 + float(r.get("distance", 0.0)))
            enriched = {
                "text": r["text"],
                "score": score,
                "metadata": r.get("metadata", {}),
                "source": "vector",
                # arrastramos etiquetas
                "result_quality": r.get("result_quality", "unknown"),
                "confidence": float(r.get("confidence", 0.5)),
                "trace_id": r.get("trace_id"),
            }
            out.append(enriched)
        return out

    # Estado
    def get_status(self) -> Dict[str, Any]:
        """
        Resumen para tests de integración. Incluye contadores específicos
        y métricas de latencia.
        """
        try:
            index_ntotal = int(getattr(self.index, "ntotal", 0))
        except Exception:
            index_ntotal = 0

        reflections = sum(1 for m in self.memories if m.get("metadata", {}).get("type") == "reflection")
        events = sum(1 for m in self.memories if m.get("metadata", {}).get("type") == "event")
        interactions = sum(1 for m in self.memories if m.get("metadata", {}).get("type") == "interaction")

        return {
            "status": "ready",
            "memory_dir": self.memory_dir,
            "memories": len(self.memories),
            "index_size": index_ntotal,
            "vector_dim": self.vector_dim,
            "use_gpu": bool(self.use_gpu),

            # etiquetas últimas usadas
            "last_trace_id": self._last_trace_id,

            # claves que piden tests / Fase 1:
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
        }

    # --------------------------- Utilidades ---------------------------
    def get_memory_count(self) -> int:
        return len(self.memories)

    def export_memories(self) -> List[Dict[str, Any]]:
        return self.memories
