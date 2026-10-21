from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .cognitive_state import CognitiveState, MemoryCandidate


@dataclass
class RelevanceDecision:
    accepted: bool
    score: float
    lexical_score: float
    entity_score: float
    tag_score: float
    temporal_score: float
    retrieval_score: float
    reason: str


class MemoryRelevanceGate:
    """
    Quipu Cognitive Control 1.0
    MemoryRelevanceGate v1.0.0

    Objetivo:
      - Evitar que recuerdos verdaderos pero irrelevantes secuestren la respuesta.
      - Separar "confianza del recuerdo" de "relevancia para esta consulta".
      - Mantener el comportamiento determinista y comprobable por tests.

    Regla central:
      Un retrieval_score alto NO basta para aceptar un recuerdo.
      Debe existir evidencia contextual: coincidencia léxica, de entidad o de tags.
    """

    VERSION = "1.0.1"

    DEFAULT_THRESHOLD = 0.34
    STRONG_ENTITY_THRESHOLD = 0.95
    MIN_CONTEXT_EVIDENCE = 0.08

    # Palabras muy frecuentes que aportan poca información temática.
    STOPWORDS = {
        "a", "al", "algo", "alguna", "alguno", "algunos", "ante", "antes",
        "aqui", "así", "asi", "bajo", "bien", "cada", "como", "con", "contra",
        "cual", "cuando", "cuanto", "cuanta", "cuantos", "cuantas", "de", "del",
        "desde", "donde", "dos", "el", "ella", "ellas", "ellos", "en", "entre",
        "era", "es", "esa", "ese", "eso", "esta", "este", "esto", "fue", "ha",
        "haber", "hasta", "hay", "la", "las", "le", "les", "lo", "los", "mas",
        "más", "me", "mi", "mis", "mucha", "mucho", "muy", "no", "nos", "o",
        "otra", "otro", "para", "pero", "por", "porque", "que", "qué", "quien",
        "se", "si", "sí", "sin", "sobre", "su", "sus", "te", "ti", "todo",
        "toda", "todos", "todas", "tu", "tus", "un", "una", "uno", "unos",
        "unas", "y", "ya",
        # Palabras de control conversacional: útiles para intención, no para tema.
        "recuerda", "recuerdas", "recordar", "memoria", "informacion",
        "información", "anterior", "anteriormente", "dijiste", "respuesta",
        "revisa", "revisar", "comprueba", "verifica", "corrige", "corregir",
    }

    GENERIC_MEMORY_TYPES = {
        "semantic", "episodic", "relational", "associative", "advanced",
        "prospective", "procedural", "unknown",
    }

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = float(threshold)

    def filter_state(self, state: CognitiveState) -> CognitiveState:
        """
        Evalúa state.memory_candidates y llena accepted_memories/rejected_memories.

        No borra candidatos: conserva trazabilidad para logs y diagnóstico.
        """
        if not isinstance(state, CognitiveState):
            raise TypeError("state debe ser CognitiveState.")

        # Evita duplicados si se invoca dos veces sobre el mismo turno.
        state.accepted_memories.clear()
        state.rejected_memories.clear()

        for candidate in state.memory_candidates:
            decision = self.evaluate(
                query=state.user_query,
                candidate=candidate,
                world_state=state.world_state,
                conversation_state=state.conversation_state,
            )

            candidate.relevance = decision.score
            candidate.entity_match = decision.entity_score
            candidate.temporal_match = decision.temporal_score

            candidate.metadata.setdefault("relevance_gate", {})
            candidate.metadata["relevance_gate"].update(
                {
                    "version": self.VERSION,
                    "score": decision.score,
                    "lexical_score": decision.lexical_score,
                    "entity_score": decision.entity_score,
                    "tag_score": decision.tag_score,
                    "temporal_score": decision.temporal_score,
                    "retrieval_score": decision.retrieval_score,
                    "accepted": decision.accepted,
                    "reason": decision.reason,
                }
            )

            if decision.accepted:
                state.accept_memory(candidate)
            else:
                state.reject_memory(candidate, decision.reason)

        return state

    def evaluate(
        self,
        query: str,
        candidate: MemoryCandidate,
        world_state: Optional[Dict[str, Any]] = None,
        conversation_state: Optional[Dict[str, Any]] = None,
    ) -> RelevanceDecision:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query debe ser un string no vacío.")
        if not isinstance(candidate, MemoryCandidate):
            raise TypeError("candidate debe ser MemoryCandidate.")

        content_text = self._stringify(candidate.content)

        query_tokens = self._key_tokens(query)
        memory_tokens = self._key_tokens(content_text)

        lexical = self._query_coverage(query_tokens, memory_tokens)

        query_entities = self._extract_entities(query)
        memory_entities = self._extract_entities(content_text)
        memory_entities |= self._metadata_entities(candidate.metadata)

        entity = self._entity_match(query_entities, memory_entities)

        query_tags = query_tokens | query_entities
        memory_tags = self._metadata_tags(candidate.metadata)
        tag_score = self._query_coverage(query_tags, memory_tags)

        temporal = self._temporal_match(query, candidate)

        retrieval_score = self._bounded_float(
            candidate.metadata.get(
                "retrieval_score",
                candidate.metadata.get("score", 0.0),
            )
        )

        # retrieval_score solo aporta un pequeño refuerzo.
        score = (
            0.55 * lexical
            + 0.25 * entity
            + 0.15 * tag_score
            + 0.03 * temporal
            + 0.02 * retrieval_score
        )

        # Boost moderado para entidades exactas.
        if entity >= self.STRONG_ENTITY_THRESHOLD:
            score = max(score, 0.72)

        # Penaliza memorias prospectivas cuando no hay evidencia de que el usuario
        # esté hablando de objetivos/planes/proyectos.
        if self._is_prospective(candidate) and not self._query_is_goal_related(query):
            score *= 0.72

        # El punto crucial: un score del retriever no puede aceptar por sí solo.
        context_evidence = max(lexical, entity, tag_score)

        if context_evidence < self.MIN_CONTEXT_EVIDENCE:
            return RelevanceDecision(
                accepted=False,
                score=round(score, 4),
                lexical_score=round(lexical, 4),
                entity_score=round(entity, 4),
                tag_score=round(tag_score, 4),
                temporal_score=round(temporal, 4),
                retrieval_score=round(retrieval_score, 4),
                reason="Sin evidencia contextual suficiente para la consulta actual.",
            )

        lexical_overlap_count = len(query_tokens & memory_tokens)
        tag_overlap_count = len(query_tags & memory_tags)

        # En consultas de largo alcance, cada recuerdo puede aportar solo una
        # pieza del problema. No exigimos que un único recuerdo cubra toda la
        # consulta: dos coincidencias temáticas independientes son evidencia
        # suficiente para dejarlo pasar a la siguiente etapa de control.
        partial_context_match = (
            max(lexical_overlap_count, tag_overlap_count) >= 2
            and score >= 0.12
        )

        accepted = score >= self.threshold or partial_context_match

        if accepted:
            if partial_context_match and score < self.threshold:
                reason = (
                    "Memoria parcialmente relevante: aporta una pieza contextual "
                    "útil para reconstrucción de largo alcance."
                )
            else:
                reason = "Memoria relevante para la consulta actual."
        else:
            reason = (
                f"Relevancia insuficiente: {score:.3f} < "
                f"umbral {self.threshold:.3f}."
            )

        return RelevanceDecision(
            accepted=accepted,
            score=round(score, 4),
            lexical_score=round(lexical, 4),
            entity_score=round(entity, 4),
            tag_score=round(tag_score, 4),
            temporal_score=round(temporal, 4),
            retrieval_score=round(retrieval_score, 4),
            reason=reason,
        )

    @classmethod
    def _key_tokens(cls, text: str) -> Set[str]:
        normalized = cls._normalize(text)
        raw = re.findall(r"[a-z0-9_]+", normalized)
        return {
            token
            for token in raw
            if len(token) >= 3 and token not in cls.STOPWORDS
        }

    @staticmethod
    def _query_coverage(query_tokens: Set[str], memory_tokens: Set[str]) -> float:
        """
        Qué parte del contenido relevante de la consulta está cubierta
        por el recuerdo. Se usa cobertura de la query, no Jaccard, para
        no castigar recuerdos largos que sí contienen el dato buscado.
        """
        if not query_tokens or not memory_tokens:
            return 0.0
        return len(query_tokens & memory_tokens) / len(query_tokens)

    @classmethod
    def _extract_entities(cls, text: str) -> Set[str]:
        """
        Extracción simple de entidades nominales.

        - Palabras con mayúscula interna/propia en el texto original.
        - Identificadores alfanuméricos con guiones/guion bajo.
        - Se normalizan para evitar que "Aster" coincida por substring
          con "asteroides".
        """
        entities: Set[str] = set()

        # Nombres propios / tokens capitalizados.
        for match in re.findall(r"\b[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚáéíóúÑñ0-9_-]{2,}\b", text):
            entities.add(cls._normalize(match))

        # Identificadores como J-42, N63, proyecto_x.
        for match in re.findall(r"\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b", text):
            entities.add(cls._normalize(match))

        return {e for e in entities if e and e not in cls.STOPWORDS}

    @classmethod
    def _metadata_entities(cls, metadata: Dict[str, Any]) -> Set[str]:
        values = metadata.get("entities", [])
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, (list, tuple, set)):
            return set()

        return {
            cls._normalize(str(v))
            for v in values
            if str(v).strip()
        }

    @classmethod
    def _metadata_tags(cls, metadata: Dict[str, Any]) -> Set[str]:
        values: List[Any] = []
        for key in ("tags", "topics", "keywords", "domain"):
            value = metadata.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, (list, tuple, set)):
                values.extend(value)

        tokens: Set[str] = set()
        for value in values:
            tokens |= cls._key_tokens(str(value))
        return tokens

    @classmethod
    def _entity_match(cls, query_entities: Set[str], memory_entities: Set[str]) -> float:
        if not query_entities or not memory_entities:
            return 0.0

        overlap = query_entities & memory_entities
        if not overlap:
            return 0.0

        return len(overlap) / len(query_entities)

    @classmethod
    def _temporal_match(cls, query: str, candidate: MemoryCandidate) -> float:
        q = cls._normalize(query)
        status = cls._normalize(str(candidate.metadata.get("status", "")))

        asks_current = any(
            marker in q
            for marker in (
                "actual", "actualmente", "vigente", "siguen vigentes",
                "ahora", "estado actual",
            )
        )
        asks_history = any(
            marker in q
            for marker in (
                "antes", "anterior", "historico", "histórico", "previo",
                "original", "inicial",
            )
        )

        if asks_current and status in {"current", "actual", "vigente", "active"}:
            return 1.0
        if asks_current and status in {"superseded", "historical", "historico", "old"}:
            return 0.0
        if asks_history and status in {"superseded", "historical", "historico", "old"}:
            return 1.0
        if asks_history and status in {"current", "actual", "vigente", "active"}:
            return 0.35

        # Sin señal temporal, no aporta ni penaliza mucho.
        return 0.5 if status else 0.0

    @classmethod
    def _is_prospective(cls, candidate: MemoryCandidate) -> bool:
        return cls._normalize(candidate.memory_type) == "prospective"

    @classmethod
    def _query_is_goal_related(cls, query: str) -> bool:
        q = cls._normalize(query)
        markers = (
            "objetivo", "meta", "plan", "pendiente", "siguiente paso",
            "proyecto", "migrar", "tarea", "hacer despues", "hacer luego",
        )
        return any(marker in q for marker in markers)

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return " ".join(f"{k} {v}" for k, v in value.items())
        if isinstance(value, (list, tuple, set)):
            return " ".join(str(v) for v in value)
        return str(value)

    @staticmethod
    def _bounded_float(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number))

    @staticmethod
    def _normalize(text: str) -> str:
        text = str(text).strip().lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"\s+", " ", text)
        return text
