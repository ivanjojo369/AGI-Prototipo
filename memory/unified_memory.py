# memory/unified_memory.py
import os
import json
import re
import time
import numpy as np
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # Memory 2.x funciona sin FAISS

try:
    from .embeddings_local import embed as semantic_embed
except Exception:
    semantic_embed = None

try:
    from .semantic_memory import SemanticMemory
except Exception:
    SemanticMemory = None

try:
    from .relational_memory import RelationalMemory
except Exception:
    RelationalMemory = None

try:
    from .associative_memory import AssociativeMemory
except Exception:
    AssociativeMemory = None

try:
    from .advanced_memory import AdvancedMemory
except Exception:
    AdvancedMemory = None

try:
    from .episodic_memory import (
        extract_fields as episodic_extract_fields,
        looks_like_memory_request as episodic_looks_like_memory_request,
        looks_like_recall_request as episodic_looks_like_recall_request,
        memory_search as episodic_search,
        memory_write as episodic_write,
    )
except Exception:
    episodic_extract_fields = None
    episodic_looks_like_memory_request = None
    episodic_looks_like_recall_request = None
    episodic_search = None
    episodic_write = None


class UnifiedMemory:
    """
    Memoria vectorial simple respaldada por FAISS.

    Compatibilidad con tus tests:
    - __init__ acepta alias: data_dir / base_dir además de memory_dir.
    - Interacciones: add_interaction(), get_context(limit) -> List[Tuple[role, content]]
    - Reflexiones:  store_reflection(title, content=None),
                    retrieve_recent_reflections(limit) -> List[dict] (incluye 'contenido')
    - Eventos:      store_event(...), get_recent_events(limit) -> List[dict] (incluye 'contenido')
    - Vector:       add_to_vector_memory(text), search_vector_memory(query),
                    vector_search(query, top_k), retrieve_relevant_memories(query, top_k)
    - Estado:       get_status() -> incluye 'reflections_stored', 'events_stored', 'interactions_stored'
    - Utilidades:   get_memory_count(), export_memories(), save_to_disk()
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
        # Evita duplicar cada mensaje cuando el logger raíz también está activo.
        self.logger.propagate = False

        self.memory_dir = str(memory_dir or "memory_store")
        self.vector_dim = vector_dim
        self.use_gpu = use_gpu
        self.lock = threading.Lock()

        os.makedirs(self.memory_dir, exist_ok=True)
        self.index_file = os.path.join(self.memory_dir, "faiss_index.bin")
        self.vector_file = os.path.join(self.memory_dir, "vector_memories.json")

        self.memories: List[Dict[str, Any]] = []  # cada item: {"text": str, "metadata": dict}
        self.index = None

        self._load_memories()
        self._init_faiss_index()

        # Ruta compartida por las capas semántica/relacional/asociativa/avanzada.
        # Se define incluso si SemanticMemory no puede inicializarse, para que
        # las capas opcionales fallen de forma limpia y nunca por una variable
        # local sin asignar.
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        semantic_base_dir = str(
            kwargs.get(
                "semantic_base_dir",
                os.path.join(project_root, "data"),
            )
        )

        # Memory 2.5 — memoria semántica consolidada.
        self.semantic_memory = None

        if SemanticMemory is not None:
            try:
                self.semantic_memory = SemanticMemory(
                    base_dir=semantic_base_dir
                )

                episodes_path = os.path.join(
                    semantic_base_dir,
                    "episodes.jsonl",
                )

                migration = (
                    self.semantic_memory
                    .consolidate_from_jsonl(
                        episodes_path
                    )
                )

                self.logger.info(
                    "[UM] Semantic Memory 2.3.x lista: %s",
                    migration,
                )
            except Exception as exc:
                self.semantic_memory = None
                self.logger.warning(
                    "[UM] Semantic Memory 2.4 no disponible: %s",
                    exc,
                )

        # Memory 2.5 — grafo relacional/temporal.
        self.relational_memory = None

        if (
            RelationalMemory is not None
            and self.semantic_memory is not None
        ):
            try:
                self.relational_memory = RelationalMemory(
                    base_dir=semantic_base_dir
                )
                relation_sync = self.relational_memory.sync_from_semantic(
                    self.semantic_memory
                )
                self.logger.info(
                    "[UM] Relational Memory 2.4 lista: %s",
                    relation_sync,
                )
            except Exception as exc:
                self.relational_memory = None
                self.logger.warning(
                    "[UM] Relational Memory 2.4 no disponible: %s",
                    exc,
                )

        # Memory 2.5 — memoria asociativa/contextual.
        self.associative_memory = None

        if (
            AssociativeMemory is not None
            and self.semantic_memory is not None
        ):
            try:
                self.associative_memory = AssociativeMemory(
                    base_dir=semantic_base_dir
                )
                associative_sync = (
                    self.associative_memory
                    .sync_from_semantic(
                        self.semantic_memory,
                        self.relational_memory,
                    )
                )
                self.logger.info(
                    "[UM] Associative Memory 2.5.1 lista: %s",
                    associative_sync,
                )
            except Exception as exc:
                self.associative_memory = None
                self.logger.warning(
                    "[UM] Associative Memory 2.5.1 no disponible: %s",
                    exc,
                )

        # Memory 2.9 — capa avanzada consolidada:
        # saliencia, conflicto/procedencia, procedimientos,
        # objetivos prospectivos, esquemas y metamemoria.
        self.advanced_memory = None

        if AdvancedMemory is not None:
            try:
                self.advanced_memory = AdvancedMemory(
                    base_dir=semantic_base_dir
                )
                advanced_sync = self.advanced_memory.sync(
                    self.semantic_memory,
                    self.relational_memory,
                    self.associative_memory,
                )
                self.logger.info(
                    "[UM] Advanced Memory 2.9.1 lista: %s",
                    advanced_sync,
                )
            except Exception as exc:
                self.advanced_memory = None
                self.logger.warning(
                    "[UM] Advanced Memory 2.9.1 no disponible: %s",
                    exc,
                )

    # --------------------------- FAISS ---------------------------
    def _init_faiss_index(self):
        # FAISS pasa a ser opcional. La memoria conversacional básica
        # debe funcionar aunque faiss-cpu/faiss-gpu no estén instalados.
        if faiss is None:
            self.index = None
            self.logger.warning(
                "[UM] FAISS no está instalado. "
                "Memory 2.9.1 usará capas episódica, semántica, relacional, asociativa y avanzada si están disponibles; si no, léxica + recencia."
            )
            return

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
                    self.logger.warning(
                        f"[UM] No se pudo usar GPU para FAISS: {e}"
                    )
        except Exception as e:
            self.logger.error(f"[UM] Error inicializando FAISS: {e}")
            try:
                self.index = faiss.IndexFlatL2(self.vector_dim)
            except Exception:
                self.index = None

    # --------------------------- IO ---------------------------
    def _load_memories(self):
        try:
            if os.path.exists(self.vector_file):
                with open(self.vector_file, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)
                self.logger.info(f"[UM] {len(self.memories)} memorias cargadas desde disco.")
            else:
                self.memories = []
        except Exception as e:
            self.logger.error(f"[UM] Error cargando memorias: {e}")
            self.memories = []

    def save_to_disk(self):
        try:
            with open(self.vector_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.memories,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            if self.index is not None and faiss is not None:
                try:
                    # Si estuviera en GPU, lo pasamos a CPU antes de guardar.
                    cpu_index = self.index
                    try:
                        cpu_index = faiss.index_gpu_to_cpu(self.index)
                    except Exception:
                        pass
                    faiss.write_index(cpu_index, self.index_file)
                except Exception as e:
                    self.logger.warning(
                        f"[UM] No se pudo guardar el índice FAISS: {e}"
                    )

            self.logger.info("[UM] Memorias guardadas en disco.")
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

    # --------------------------- Altas / Búsquedas ---------------------------
    def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        try:
            if not isinstance(text, str):
                text = str(text)

            md = dict(metadata or {})
            md.setdefault("ts", time.time())

            with self.lock:
                # El almacenamiento conversacional no depende de FAISS.
                if self.index is not None:
                    try:
                        emb = self.embed_text(text)
                        self.index.add(
                            np.array([emb], dtype=np.float32)
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"[UM] No se pudo indexar en FAISS: {e}"
                        )

                self.memories.append(
                    {
                        "text": text,
                        "metadata": md,
                    }
                )
                idx = len(self.memories) - 1

            self.save_to_disk()
            self.logger.info(
                f"[UM] Memoria agregada: {text[:50]}"
            )
            return idx

        except Exception as e:
            self.logger.error(f"[UM] Error agregando memoria: {e}")
            return -1

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return [
            tok
            for tok in re.split(
                r"[^\wáéíóúñü]+",
                (text or "").lower(),
            )
            if tok
        ]

    def _lexical_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        q_tokens = set(self._tokens(query))

        scored: List[Tuple[float, int, Dict[str, Any]]] = []

        for idx, memory in enumerate(self.memories):
            text_value = str(memory.get("text", ""))
            md = dict(memory.get("metadata", {}))

            tokens = set(self._tokens(text_value))
            overlap = len(q_tokens & tokens)

            # Recencia suave: los elementos recientes ganan un pequeño bonus,
            # pero un recuerdo antiguo con palabras relevantes puede superar
            # a uno reciente irrelevante.
            recency = (idx + 1) / max(1, len(self.memories))
            score = float(overlap) * 4.0 + recency

            if overlap > 0:
                scored.append(
                    (
                        score,
                        idx,
                        {
                            "text": text_value,
                            "metadata": md,
                            # Compatibilidad con consumidores que esperan distance.
                            "distance": 1.0 / (1.0 + score),
                        },
                    )
                )

        scored.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        return [
            item[2]
            for item in scored[: max(1, int(top_k or 5))]
        ]

    def search_memory(
        self,
        query_vector,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        try:
            query_text: Optional[str] = None

            if isinstance(query_vector, dict):
                query_vector = query_vector.get("input", "")

            if isinstance(query_vector, str):
                query_text = query_vector

            if self.index is None or faiss is None:
                if query_text is not None:
                    return self._lexical_search(
                        query_text,
                        top_k=top_k,
                    )
                return []

            if isinstance(query_vector, str):
                query_vector = self.embed_text(query_vector)

            query_vector = np.array(
                query_vector,
                dtype=np.float32,
            ).reshape(1, -1)

            distances, indices = self.index.search(
                query_vector,
                top_k,
            )

            out: List[Dict[str, Any]] = []

            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(self.memories):
                    out.append(
                        {
                            "text": self.memories[idx]["text"],
                            "metadata": self.memories[idx].get(
                                "metadata",
                                {},
                            ),
                            "distance": float(dist),
                        }
                    )

            return out

        except Exception as e:
            self.logger.error(f"[UM] Error en search_memory: {e}")

            if isinstance(query_vector, str):
                return self._lexical_search(
                    query_vector,
                    top_k=top_k,
                )

            return []

    # --------------------------- API de compatibilidad ---------------------------
    # API directa usada por QuipuLoop ---------------------------------
    @staticmethod
    def _query_requests_current_fact(
        query: str,
    ) -> bool:
        """
        Detecta consultas que piden explícitamente el valor vigente/actual.

        En esos casos Semantic Memory debe tener prioridad sobre episodios
        históricos/superseded.
        """
        text = (query or "").strip().lower()

        if re.search(
            r"\b(?:actual|actualmente|vigente|ahora|hoy)\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True

        return any(
            marker in text
            for marker in (
                "más reciente",
                "mas reciente",
                "último valor",
                "ultimo valor",
                "valor vigente",
                "dato vigente",
            )
        )

    @staticmethod
    def is_contextual_followup(query: str) -> bool:
        """
        Política de continuidad conversacional 1.0.2.

        True solo cuando el turno actual depende claramente de contexto previo.
        Un saludo o una pregunta autosuficiente NO debe heredar por recencia
        las últimas interacciones.
        """
        text = " ".join((query or "").strip().lower().split())
        if not text:
            return False

        # Cambios explícitos de tema: empezar limpio.
        independent_shift_markers = (
            "ahora hablemos de",
            "hablemos de",
            "pasemos a",
            "cambiando de tema",
            "otro tema",
            "y si hablamos de",
        )
        if any(marker in text for marker in independent_shift_markers):
            return False

        # Saludos simples no dependen del turno anterior.
        if re.fullmatch(
            r"[¿¡!,.\s]*(?:hola|buenas|buenos días|buenos dias|"
            r"buenas tardes|buenas noches|hey|qué tal|que tal)"
            r"(?:[,!\s]+(?:cómo estás|como estas|qué tal|que tal))?"
            r"[¿?¡!,.\s]*",
            text,
            flags=re.IGNORECASE,
        ):
            return False

        explicit_markers = (
            "lo anterior",
            "lo de antes",
            "lo que dijiste",
            "lo que mencionaste",
            "que te mencioné",
            "que te mencione",
            "que te dije",
            "te comenté",
            "te comente",
            "te mencioné",
            "te mencione",
            "el proyecto que te",
            "como dijiste",
            "como mencionaste",
            "con toda la información anterior",
            "con toda la informacion anterior",
            "con toda la información que te he dado",
            "con toda la informacion que te he dado",
            "con lo anterior",
            "a partir de eso",
            "en ese caso",
            "esa parte",
            "ese punto",
            "esto último",
            "esto ultimo",
            "continúa",
            "continua",
            "sigue con",
            "retomemos",
            "retomando",
            "regresando a",
            "volviendo a",
            "revisa lo que dijiste",
            "corrige lo anterior",
        )
        if any(marker in text for marker in explicit_markers):
            return True

        # Pronombres/anclajes que requieren un antecedente.
        if re.search(
            r"\b(?:eso|esa|ese|esto|aquello|anterior|antes)\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True

        # "Y si..." suele aportar una condición nueva al hilo anterior.
        # "Y si hablamos de..." ya fue excluido arriba como cambio de tema.
        if re.match(
            r"^[¿¡!,.\s]*y\s+si\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True

        # Otras continuaciones breves: "¿y por qué?", "entonces...", etc.
        tokens = re.findall(r"[a-záéíóúñü0-9%]+", text)
        if (
            len(tokens) <= 16
            and re.match(
                r"^[¿¡!,.\s]*(?:y|entonces|pero|además|ademas)\b",
                text,
                flags=re.IGNORECASE,
            )
        ):
            return True

        return False

    @staticmethod
    def _query_requests_multiple_episodes(query: str) -> bool:
        """Detecta consultas realmente multi-episodio."""
        lowered = (query or "").lower()
        explicit_markers = (
            "todas las memorias",
            "todos los recuerdos",
            "todos los episodios",
            "todas las memorias episódicas",
            "varias memorias",
            "varios recuerdos",
            "varios episodios",
            "memorias nuevas",
            "recuerdos nuevos",
        )
        if any(marker in lowered for marker in explicit_markers):
            return True
        if re.search(
            r"\b(?:las|los)\s+(?:dos|tres|cuatro|cinco|2|3|4|5)\s+"
            r"(?:memorias?|recuerdos?|episodios?)\b",
            lowered,
        ):
            return True
        if any(verb in lowered for verb in ("reúne", "reune", "combina", "agrupa")):
            if re.search(r"\b(?:memorias?|recuerdos?|episodios?)\b", lowered):
                return True
        return False


    @staticmethod
    def _episode_to_recall(item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convierte un episodio recuperado al formato interno de Quipu.

        Memory 2.4 prioriza los campos estructurados sobre el texto bruto.
        El score y episode_key se conservan como metadatos internos, pero el
        contenido que verá el modelo contiene únicamente hechos.
        """
        fields = {
            str(key): str(value)
            for key, value in dict(item.get("fields") or {}).items()
            if str(key).strip()
            and str(value).strip()
            and not re.match(
                r"^(?:responde|respondeme|respuesta|contesta|contestame|reply|answer)",
                str(key).strip(),
                flags=re.IGNORECASE,
            )
        }

        if fields:
            content = "\n".join(
                f"{key} = {value}"
                for key, value in fields.items()
            )
        else:
            content = str(item.get("text") or "")

        return {
            "role": "memory",
            "content": content,
            "ts": item.get("ts", ""),
            "source": "episode",
            "episode_key": item.get("episode_key") or "episodio",
            "score": float(item.get("score", 0.0)),
            "fields": fields,
        }


    def _recall_episodes(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if episodic_search is None:
            return []

        try:
            raw = episodic_search(
                query,
                topk=max(3, min(6, limit)),
            )
        except Exception as exc:
            self.logger.warning(
                f"[UM] Búsqueda episódica no disponible: {exc}"
            )
            return []

        if not raw:
            return []

        # Si la consulta pide un solo recuerdo, damos prioridad al mejor
        # episodio y evitamos inundar el contexto con episodios vecinos.
        if not self._query_requests_multiple_episodes(query):
            best = raw[0]
            best_score = float(best.get("score", 0.0))
            second_score = (
                float(raw[1].get("score", 0.0))
                if len(raw) > 1
                else -1.0
            )

            # Incluso si la diferencia es pequeña, devolver un episodio
            # completo es preferible a mezclar fragmentos de varios.
            selected = [best]

            # Solo añade un segundo si tiene casi la misma puntuación y la
            # consulta contiene dos nombres/temas explícitos.
            named_hits = sum(
                1
                for item in raw[:3]
                if str(item.get("episode_key") or "").lower()
                in (query or "").lower()
            )
            if (
                named_hits >= 2
                and len(raw) > 1
                and second_score >= best_score - 0.08
            ):
                selected.append(raw[1])

            return [
                self._episode_to_recall(item)
                for item in selected
            ]

        # Consultas agregadoras: permite varios episodios completos.
        return [
            self._episode_to_recall(item)
            for item in raw[: min(5, limit)]
        ]

    @staticmethod
    def _semantic_to_recall(
        subject: str,
        facts: List[Dict[str, Any]],
        score: float,
    ) -> Dict[str, Any]:
        fields = {
            str(item.get("predicate")):
                str(item.get("value"))
            for item in facts
            if item.get("predicate")
            and item.get("value")
            not in (None, "")
        }

        content = "\n".join(
            f"{key} = {value}"
            for key, value
            in fields.items()
        )

        return {
            "role": "memory",
            "content": content,
            "source": "semantic",
            "semantic_subject": subject,
            "score": float(score),
            "fields": fields,
            "facts": facts,
        }

    def _recall_semantic(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if self.semantic_memory is None:
            return []

        try:
            hits = (
                self.semantic_memory.search(
                    query,
                    top_k=max(
                        8,
                        min(
                            20,
                            limit * 2,
                        ),
                    ),
                )
            )
        except Exception as exc:
            self.logger.warning(
                "[UM] Búsqueda Semantic Memory 2.4 no disponible: %s",
                exc,
            )
            return []

        structured = [
            hit
            for hit in hits
            if hit.get("kind")
            == "semantic_fact"
            and hit.get("subject")
            and float(
                hit.get("score")
                or 0.0
            ) >= 0.24
        ]

        if not structured:
            return []

        query_lower = (
            query or ""
        ).lower()

        named_subjects: List[str] = []

        for hit in structured:
            subject = str(
                hit.get("subject")
                or ""
            ).strip()

            if (
                subject
                and subject.lower()
                in query_lower
                and subject
                not in named_subjects
            ):
                named_subjects.append(
                    subject
                )

        if named_subjects:
            selected_subjects = (
                named_subjects[:3]
            )
        else:
            subject_scores: Dict[
                str,
                float,
            ] = {}

            for hit in structured:
                subject = str(
                    hit.get("subject")
                    or ""
                )
                score = float(
                    hit.get("score")
                    or 0.0
                )

                subject_scores[
                    subject
                ] = max(
                    subject_scores.get(
                        subject,
                        0.0,
                    ),
                    score,
                )

            selected_subjects = [
                subject
                for subject, _
                in sorted(
                    subject_scores.items(),
                    key=lambda item:
                        item[1],
                    reverse=True,
                )[:2]
            ]

        recalls: List[
            Dict[str, Any]
        ] = []

        for subject in (
            selected_subjects
        ):
            try:
                facts = (
                    self.semantic_memory
                    .facts_for_subject(
                        subject
                    )
                )
            except Exception:
                facts = []

            if not facts:
                continue

            best_score = max(
                (
                    float(
                        hit.get("score")
                        or 0.0
                    )
                    for hit in structured
                    if str(
                        hit.get("subject")
                        or ""
                    )
                    == subject
                ),
                default=0.0,
            )

            if (
                best_score < 0.30
                and subject.lower()
                not in query_lower
            ):
                continue

            recalls.append(
                self._semantic_to_recall(
                    subject,
                    facts,
                    best_score,
                )
            )

        return recalls[
            : max(
                1,
                min(3, limit),
            )
        ]

    def add(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Guarda un turno conversacional.

        Memory 2.9.1:
        - todos los turnos siguen en vector_memories.json;
        - peticiones explícitas de memoria se guardan además como
          episodios completos en data/episodes.jsonl.
        """
        role = str(role or "user").strip().lower()
        content = str(content or "").strip()

        if not content:
            return -1

        interaction_metadata = dict(metadata or {})
        interaction_metadata.update(
            {
                "type": "interaction",
                "role": role,
            }
        )
        interaction_metadata.setdefault("ts", time.time())

        idx = self.add_memory(
            content,
            metadata=interaction_metadata,
        )

        if (
            role == "user"
            and episodic_write is not None
            and episodic_looks_like_memory_request is not None
        ):
            try:
                should_store = bool(
                    episodic_looks_like_memory_request(content)
                )
                is_recall = bool(
                    episodic_looks_like_recall_request(content)
                ) if episodic_looks_like_recall_request is not None else False

                fields = (
                    episodic_extract_fields(content)
                    if episodic_extract_fields is not None
                    else {}
                )

                # Memory 2.9.1:
                # Solo se crea episodio si hay intención explícita de guardar,
                # NO es una consulta de recuperación y existen al menos
                # dos pares clave=valor.
                if should_store and not is_recall and len(fields) >= 2:
                    episode_result = episodic_write(
                        content,
                        tags=["explicit_memory"],
                        importance=0.95,
                        meta={
                            "fields": fields,
                            "source": "QuipuLoop",
                            "role": "user",
                        },
                    )

                    if (
                        self.semantic_memory is not None
                        and isinstance(
                            episode_result,
                            dict,
                        )
                        and episode_result.get("ok")
                    ):
                        try:
                            self.semantic_memory.consolidate_episode(
                                episode_result
                            )

                            if self.relational_memory is not None:
                                self.relational_memory.sync_from_semantic(
                                    self.semantic_memory
                                )

                            if self.associative_memory is not None:
                                self.associative_memory.sync_from_semantic(
                                    self.semantic_memory,
                                    self.relational_memory,
                                )
                        except Exception as semantic_exc:
                            self.logger.warning(
                                "[UM] No se pudo consolidar el episodio "
                                "en Semantic Memory 2.9.1: %s",
                                semantic_exc,
                            )
            except Exception as exc:
                self.logger.warning(
                    f"[UM] No se pudo guardar episodio: {exc}"
                )

        if (
            role == "user"
            and self.advanced_memory is not None
        ):
            try:
                advanced_fields = (
                    episodic_extract_fields(
                        content
                    )
                    if episodic_extract_fields
                    is not None
                    else None
                )

                self.advanced_memory.ingest_user(
                    content,
                    fields=advanced_fields,
                )
                self.advanced_memory.sync(
                    self.semantic_memory,
                    self.relational_memory,
                    self.associative_memory,
                )
            except Exception as exc:
                self.logger.warning(
                    "[UM] Advanced Memory 2.9.1 no pudo ingerir/sincronizar: %s",
                    exc,
                )

        return idx


    @staticmethod
    def _cosine_from_normalized(
        a: List[float],
        b: List[float],
    ) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return float(sum(x * y for x, y in zip(a, b)))

    def _semantic_rank(
        self,
        query: str,
        interactions: List[Tuple[int, Dict[str, Any]]],
    ) -> List[Tuple[float, int, Dict[str, Any]]]:
        """
        Ordena recuerdos por similitud semántica usando embeddings locales.

        No requiere FAISS. Si sentence-transformers no está disponible,
        devuelve [] y recall() usa el fallback léxico.
        """
        if semantic_embed is None or not interactions:
            return []

        texts = [
            str(memory.get("text", ""))
            for _, memory in interactions
        ]

        try:
            vectors = semantic_embed([query] + texts)
        except Exception as e:
            self.logger.warning(
                f"[UM] Embeddings semánticos no disponibles: {e}"
            )
            return []

        if not vectors or len(vectors) != len(texts) + 1:
            return []

        query_vec = vectors[0]
        ranked: List[Tuple[float, int, Dict[str, Any]]] = []
        total = max(1, len(interactions))

        for pos, ((idx, memory), mem_vec) in enumerate(
            zip(interactions, vectors[1:])
        ):
            semantic_score = self._cosine_from_normalized(
                query_vec,
                mem_vec,
            )
            recency = (pos + 1) / total
            score = semantic_score * 10.0 + recency * 0.25
            ranked.append((score, idx, memory))

        ranked.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        return ranked

    def _finalize_recalls(
        self,
        recalls: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        selected = list(
            recalls or []
        )[: max(1, int(limit or 10))]

        if (
            selected
            and self.advanced_memory is not None
        ):
            try:
                self.advanced_memory.note_recall(
                    selected
                )
            except Exception:
                pass

        return selected

    def _recall_advanced_memory(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if self.advanced_memory is None:
            return []

        try:
            self.advanced_memory.sync(
                self.semantic_memory,
                self.relational_memory,
                self.associative_memory,
            )
            return self.advanced_memory.recall(
                query,
                top_k=max(
                    1,
                    int(limit or 10),
                ),
            )
        except Exception as exc:
            self.logger.warning(
                "[UM] Recuperación Advanced Memory 2.9.1 no disponible: %s",
                exc,
            )
            return []

    def _recall_associative(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if (
            self.associative_memory is None
            or self.semantic_memory is None
        ):
            return []

        try:
            # La capa es materializada: puede reconstruirse sin modificar
            # las fuentes canónicas.
            self.associative_memory.sync_from_semantic(
                self.semantic_memory,
                self.relational_memory,
            )

            return self.associative_memory.recall(
                query,
                top_k=max(
                    2,
                    min(
                        3,
                        int(limit or 10),
                    ),
                ),
            )[: max(1, int(limit or 10))]
        except Exception as exc:
            self.logger.warning(
                "[UM] Recuperación asociativa 2.5.1 no disponible: %s",
                exc,
            )
            return []

    def _recall_relational(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if (
            self.relational_memory is None
            or self.semantic_memory is None
        ):
            return []

        try:
            self.relational_memory.sync_from_semantic(
                self.semantic_memory
            )
            return self.relational_memory.recall(
                query,
                max_hops=3,
                max_edges=max(
                    4,
                    min(12, int(limit or 10)),
                ),
            )[: max(1, int(limit or 10))]
        except Exception as exc:
            self.logger.warning(
                "[UM] Recuperación relacional 2.4 no disponible: %s",
                exc,
            )
            return []

    def recall(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Memory 2.9.1:
        - petición explícita de recuerdo/episodio -> Episodic Memory;
        - pregunta factual estable -> Semantic Memory;
        - conversación reciente -> contexto auxiliar.

        Se conserva el aislamiento episódico aprobado en Memory 2.2.3.
        """
        limit = max(
            1,
            int(top_k or 10),
        )

        advanced_recalls = self._recall_advanced_memory(
            query,
            limit=limit,
        )

        if advanced_recalls:
            return self._finalize_recalls(
                advanced_recalls,
                limit,
            )

        associative_recalls = self._recall_associative(
            query,
            limit=limit,
        )

        if associative_recalls:
            return self._finalize_recalls(
                associative_recalls,
                limit,
            )

        relational_recalls = self._recall_relational(
            query,
            limit=limit,
        )

        if relational_recalls:
            return self._finalize_recalls(
                relational_recalls,
                limit,
            )

        # Memory 3.0.2:
        # Si el usuario pide el valor ACTUAL/VIGENTE, consultamos primero
        # Semantic Memory. Así un episodio histórico (por ejemplo Rust)
        # no puede desplazar al valor consolidado actual (por ejemplo Python).
        current_fact_query = self._query_requests_current_fact(
            query
        )
        semantic_recalls: List[
            Dict[str, Any]
        ] = []

        query_lower = (
            query or ""
        ).lower()

        if current_fact_query:
            semantic_recalls = (
                self._recall_semantic(
                    query,
                    limit=limit,
                )
            )

            named_semantic_subject = any(
                str(
                    item.get(
                        "semantic_subject"
                    )
                    or ""
                ).strip()
                and str(
                    item.get(
                        "semantic_subject"
                    )
                    or ""
                ).strip().lower()
                in query_lower
                for item in semantic_recalls
            )

            if (
                semantic_recalls
                and named_semantic_subject
            ):
                return self._finalize_recalls(
                    semantic_recalls,
                    limit,
                )

        is_episodic_query = False

        if (
            episodic_looks_like_recall_request
            is not None
        ):
            try:
                is_episodic_query = bool(
                    episodic_looks_like_recall_request(
                        query
                    )
                )
            except Exception:
                is_episodic_query = False

        # Un episodio sigue siendo válido cuando:
        # - no se está pidiendo un valor actual; o
        # - no existe conocimiento semántico actual utilizable.
        if is_episodic_query:
            episodes = (
                self._recall_episodes(
                    query,
                    limit=limit,
                )
            )

            if episodes:
                return self._finalize_recalls(
                    episodes,
                    limit,
                )

        if not semantic_recalls:
            semantic_recalls = (
                self._recall_semantic(
                    query,
                    limit=limit,
                )
            )

        # Si la consulta nombra explícitamente un sujeto semántico
        # recuperado (Aster, Solaris, Observatorio Vela, etc.),
        # ese conocimiento consolidado es la fuente factual autoritativa.
        # No añadimos respuestas conversacionales antiguas que podrían
        # contener valores ya superados.
        named_semantic_subject = any(
            str(item.get("semantic_subject") or "").strip()
            and str(item.get("semantic_subject") or "").strip().lower()
            in query_lower
            for item in semantic_recalls
        )

        if semantic_recalls and named_semantic_subject:
            return self._finalize_recalls(
                semantic_recalls,
                limit,
            )

        interactions: List[
            Tuple[int, Dict[str, Any]]
        ] = []

        for idx, memory in enumerate(
            self.memories
        ):
            md = dict(
                memory.get("metadata")
                or {}
            )
            role = md.get("role")

            if (
                md.get("type")
                == "interaction"
                or role
                in ("user", "assistant")
            ):
                interactions.append(
                    (idx, memory)
                )

        if not interactions:
            return self._finalize_recalls(
                semantic_recalls,
                limit,
            )

        allow_recent_fallback = self.is_contextual_followup(
            query
        )

        q_tokens = set(
            self._tokens(query)
        )
        semantic_ranked = (
            self._semantic_rank(
                query,
                interactions,
            )
        )

        semantic_scores = {
            idx: score
            for score, idx, _
            in semantic_ranked
        }

        combined: List[
            Tuple[
                float,
                int,
                Dict[str, Any],
            ]
        ] = []

        for pos, (
            idx,
            memory,
        ) in enumerate(
            interactions
        ):
            content = str(
                memory.get("text")
                or ""
            )
            tokens = set(
                self._tokens(content)
            )
            overlap = len(
                q_tokens & tokens
            )
            recency = (
                (pos + 1)
                / max(
                    1,
                    len(interactions),
                )
            )

            lexical = (
                float(overlap) * 2.0
                + recency * 0.15
            )
            semantic = (
                semantic_scores.get(
                    idx,
                    0.0,
                )
            )
            score = (
                semantic + lexical
            )
            combined.append(
                (
                    score,
                    idx,
                    memory,
                )
            )

        combined.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        selected_indices: List[
            int
        ] = []

        interaction_budget = max(
            2,
            limit
            - len(
                semantic_recalls
            ),
        )
        relevant_budget = max(
            1,
            int(
                interaction_budget
                * 0.55
            ),
        )

        for _, idx, memory in (
            combined
        ):
            content = str(
                memory.get("text")
                or ""
            )
            md = dict(
                memory.get("metadata")
                or {}
            )

            if md.get("context_reuse") is False:
                continue

            overlap = len(
                q_tokens
                & set(
                    self._tokens(content)
                )
            )
            semantic_score = float(
                semantic_scores.get(
                    idx,
                    0.0,
                )
            )

            # Para consultas independientes no basta la recencia:
            # exigimos señal léxica o semántica real.
            if (
                not allow_recent_fallback
                and overlap <= 0
                and semantic_score < 4.5
            ):
                continue

            if (
                semantic_recalls
                and
                episodic_looks_like_memory_request
                is not None
            ):
                try:
                    if (
                        episodic_looks_like_memory_request(
                            content
                        )
                    ):
                        continue
                except Exception:
                    pass

            if re.fullmatch(
                r"[A-ZÁÉÍÓÚÑ0-9_-]+_GUARDAD[AO]",
                content.strip(),
                flags=re.IGNORECASE,
            ):
                continue

            if (
                idx
                not in selected_indices
            ):
                selected_indices.append(
                    idx
                )

            if (
                len(selected_indices)
                >= relevant_budget
            ):
                break

        if allow_recent_fallback:
            for idx, memory in reversed(
                interactions
            ):
                content = str(
                    memory.get("text")
                    or ""
                )
                md = dict(
                    memory.get("metadata")
                    or {}
                )

                if md.get("context_reuse") is False:
                    continue

                if (
                    semantic_recalls
                    and
                    episodic_looks_like_memory_request
                    is not None
                ):
                    try:
                        if (
                            episodic_looks_like_memory_request(
                                content
                            )
                        ):
                            continue
                    except Exception:
                        pass

                if re.fullmatch(
                    r"[A-ZÁÉÍÓÚÑ0-9_-]+_GUARDAD[AO]",
                    content.strip(),
                    flags=re.IGNORECASE,
                ):
                    continue

                if (
                    idx
                    not in selected_indices
                ):
                    selected_indices.append(
                        idx
                    )

                if (
                    len(selected_indices)
                    >= interaction_budget
                ):
                    break

        selected_indices = sorted(
            selected_indices[
                :interaction_budget
            ]
        )

        recent_context: List[
            Dict[str, Any]
        ] = []

        for idx in selected_indices:
            memory = (
                self.memories[idx]
            )
            md = dict(
                memory.get("metadata")
                or {}
            )
            content = str(
                memory.get("text")
                or ""
            )

            if len(content) > 1200:
                content = (
                    content[:1200]
                    + "…"
                )

            recent_context.append(
                {
                    "role":
                        md.get(
                            "role",
                            "user",
                        ),
                    "content":
                        content,
                    "ts":
                        md.get(
                            "ts",
                            "",
                        ),
                    "source":
                        "interaction",
                    "metadata":
                        md,
                    "context_reuse":
                        md.get(
                            "context_reuse",
                            True,
                        ),
                }
            )

        return self._finalize_recalls(
            semantic_recalls
            + recent_context,
            limit,
        )


    # Interacciones
    def add_interaction(self, user_text: str, assistant_text: Optional[str] = None):
        self.add_memory(user_text, metadata={"type": "interaction", "role": "user"})
        if assistant_text:
            self.add_memory(assistant_text, metadata={"type": "interaction", "role": "assistant"})

    def get_context_records(
        self,
        limit: int = 10,
        *,
        reusable_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Devuelve interacciones recientes con metadata.

        reusable_only=True excluye respuestas marcadas como no reutilizables
        (por ejemplo bloqueos metacognitivos).
        """
        items: List[Dict[str, Any]] = []

        for m in reversed(self.memories):
            md = dict(
                m.get("metadata")
                or {}
            )

            if md.get("type") != "interaction":
                continue

            if (
                reusable_only
                and md.get("context_reuse") is False
            ):
                continue

            items.append(
                {
                    "role": md.get("role", "user"),
                    "content": m.get("text", ""),
                    "metadata": md,
                }
            )

            if len(items) >= limit:
                break

        items.reverse()
        return items

    def get_context(self, limit: int = 10) -> List[Tuple[str, str]]:
        """
        Compatibilidad: devuelve las últimas 'limit' interacciones como
        lista de TUPLAS (role, content).
        """
        return [
            (
                str(item.get("role") or "user"),
                str(item.get("content") or ""),
            )
            for item in self.get_context_records(
                limit=limit,
                reusable_only=False,
            )
        ]

    # Reflexiones
    def store_reflection(self, title: str, content: Optional[Any] = None) -> int:
        text = f"{title}: {content}" if content is not None else str(title)
        return self.add_memory(text, metadata={"type": "reflection", "title": title})

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
                }
                out.append(item)
                if len(out) >= limit:
                    break
        return out

    # Eventos
    def store_event(self, *args, **kwargs) -> int:
        """
        Acepta:
          - store_event('tipo', 'texto')
          - store_event({'text': '...', 'metadata': {...}})
          - store_event(text='...', metadata={...})
        """
        if len(args) >= 2 and isinstance(args[0], str):
            event_type, text = args[0], args[1]
            meta = {"type": "event", "event_type": event_type}
            return self.add_memory(str(text), metadata=meta)

        if len(args) == 1 and isinstance(args[0], dict):
            event = args[0]
            meta = {"type": "event", **(event.get("metadata", {}) if isinstance(event, dict) else {})}
            text = event.get("text") if isinstance(event, dict) else str(event)
            return self.add_memory(text or "event", metadata=meta)

        text = kwargs.get("text", "event")
        meta = {"type": "event", **kwargs.get("metadata", {})}
        return self.add_memory(str(text), metadata=meta)

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
                }
                out.append(item)
                if len(out) >= limit:
                    break
        return out

    # Vector (aliases usados por tests)
    def add_to_vector_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        return self.add_memory(text, metadata=metadata)

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
            out.append({"text": r["text"], "score": score, "metadata": r.get("metadata", {}), "source": "vector"})
        return out

    # Estado
    def get_status(self) -> Dict[str, Any]:
        """
        Resumen para tests de integración. Incluye contadores específicos
        que los tests buscan (p. ej. 'reflections_stored').
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
            # claves que piden los tests:
            "reflections_stored": reflections,
            "events_stored": events,
            "interactions_stored": interactions,
        }

    # --------------------------- Utilidades ---------------------------
    def get_memory_count(self) -> int:
        return len(self.memories)

    def export_memories(self) -> List[Dict[str, Any]]:
        return self.memories
