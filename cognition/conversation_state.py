from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence
import re
import time
import unicodedata

from .cognitive_state import CognitiveState


VALID_TOPIC_STATUSES = {
    "active",
    "resolved",
    "blocked",
    "paused",
}

VALID_HYPOTHESIS_STATUSES = {
    "open",
    "supported",
    "weakened",
    "rejected",
    "resolved",
}


@dataclass
class TopicRecord:
    topic_id: str
    title: str
    status: str = "active"

    summary: Optional[str] = None
    block_reason: Optional[str] = None

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionRecord:
    decision_id: str
    text: str

    topic_id: Optional[str] = None
    accepted: bool = True
    source: str = "user"

    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PendingQuestion:
    question_id: str
    text: str

    topic_id: Optional[str] = None
    status: str = "pending"
    answer: Optional[str] = None

    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HypothesisRecord:
    hypothesis_id: str
    text: str

    topic_id: Optional[str] = None
    status: str = "open"
    confidence: float = 0.5

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationStateManager:
    """
    Quipu Cognitive Control 1.0
    ConversationStateManager v1.0.2

    Objetivo:
      - Mantener qué temas siguen activos, resueltos, bloqueados o pausados.
      - Registrar decisiones aceptadas por el usuario.
      - Mantener preguntas pendientes e hipótesis abiertas.
      - Evitar que Quipu repita planes completos que ya fueron aprobados.
      - Permitir reabrir explícitamente un tema resuelto.
      - Sincronizar todo con CognitiveState.conversation_state.

    PRINCIPIO:
      "Recordar el contenido" y "recordar el estado de la conversación"
      son problemas distintos.

    Este módulo se ocupa del segundo.

    CAMBIO 1.0.2:
      - Mantiene el fallback del tema ACTIVE más reciente de 1.0.1.
      - Añade anclas léxicas de tema para reactivar un tema previo.
      - Puede encontrar un tema BLOCKED/PAUSED relevante por contenido,
        incluso si hubo un turno intermedio no relacionado.
    """

    VERSION = "1.0.2"

    APPROVAL_MARKERS = (
        "esta bien",
        "está bien",
        "queda bien",
        "quedo bien",
        "quedó bien",
        "si queda",
        "sí queda",
        "me gusta",
        "aprobado",
        "de acuerdo",
        "perfecto",
        "asi queda",
        "así queda",
    )

    REOPEN_MARKERS = (
        "regresando a",
        "volviendo a",
        "retomemos",
        "retomando",
        "regresemos a",
        "volvamos a",
    )

    TOPIC_STOPWORDS = {
        "a", "al", "algo", "ante", "antes", "como", "con", "contra",
        "cual", "cuando", "de", "del", "desde", "donde", "el", "ella",
        "en", "entonces", "era", "es", "esa", "ese", "eso", "esta",
        "este", "esto", "hay", "la", "las", "lo", "los", "me", "mi",
        "para", "pero", "por", "que", "se", "si", "sin", "sobre",
        "su", "te", "tengo", "tiene", "tendra", "tendria", "tu", "un",
        "una", "y", "ya",
    }

    def __init__(self) -> None:
        self._topics: Dict[str, TopicRecord] = {}
        self._decisions: Dict[str, DecisionRecord] = {}
        self._pending: Dict[str, PendingQuestion] = {}
        self._hypotheses: Dict[str, HypothesisRecord] = {}

        self.current_topic_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Topics
    # ------------------------------------------------------------------

    def start_topic(
        self,
        topic_id: str,
        title: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TopicRecord:
        self._validate_id(topic_id, "topic_id")
        self._validate_text(title, "title")

        existing = self._topics.get(topic_id)
        if existing is not None:
            existing.status = "active"
            existing.updated_at = time.time()
            if existing.block_reason:
                existing.metadata["previous_block_reason"] = (
                    existing.block_reason
                )
            existing.block_reason = None
            if metadata:
                existing.metadata.update(metadata)
            self._merge_topic_anchors(
                existing,
                title,
            )
            self.current_topic_id = topic_id
            return existing

        topic = TopicRecord(
            topic_id=topic_id,
            title=title,
            status="active",
            metadata=dict(metadata or {}),
        )
        self._merge_topic_anchors(
            topic,
            title,
        )
        self._topics[topic_id] = topic
        self.current_topic_id = topic_id
        return topic

    def resolve_topic(
        self,
        topic_id: str,
        *,
        summary: Optional[str] = None,
    ) -> TopicRecord:
        topic = self._require_topic(topic_id)
        topic.status = "resolved"
        topic.summary = summary or topic.summary
        topic.block_reason = None
        topic.updated_at = time.time()

        if self.current_topic_id == topic_id:
            self.current_topic_id = self._fallback_current_topic(
                exclude=topic_id
            )

        return topic

    def block_topic(
        self,
        topic_id: str,
        reason: str,
    ) -> TopicRecord:
        topic = self._require_topic(topic_id)
        self._validate_text(reason, "reason")

        topic.status = "blocked"
        topic.block_reason = reason.strip()
        topic.updated_at = time.time()
        self._merge_topic_anchors(
            topic,
            reason,
        )

        if self.current_topic_id == topic_id:
            self.current_topic_id = self._fallback_current_topic(
                exclude=topic_id
            )

        return topic

    def pause_topic(self, topic_id: str) -> TopicRecord:
        topic = self._require_topic(topic_id)
        topic.status = "paused"
        topic.updated_at = time.time()

        if self.current_topic_id == topic_id:
            self.current_topic_id = self._fallback_current_topic(
                exclude=topic_id
            )

        return topic

    def reopen_topic(self, topic_id: str) -> TopicRecord:
        topic = self._require_topic(topic_id)
        if topic.block_reason:
            topic.metadata["previous_block_reason"] = (
                topic.block_reason
            )
        topic.status = "active"
        topic.block_reason = None
        topic.updated_at = time.time()
        self.current_topic_id = topic_id
        return topic

    def note_topic_turn(
        self,
        topic_id: str,
        user_text: str,
    ) -> TopicRecord:
        """
        Añade anclas del turno al tema sin convertir el contenido en memoria
        factual. Sirve únicamente para poder reidentificar el hilo después.
        """
        topic = self._require_topic(topic_id)
        self._validate_text(user_text, "user_text")

        self._merge_topic_anchors(
            topic,
            user_text,
        )
        topic.metadata["last_user_text"] = (
            user_text.strip()[:240]
        )
        topic.metadata["turn_count"] = int(
            topic.metadata.get(
                "turn_count",
                0,
            )
            or 0
        ) + 1
        topic.updated_at = time.time()
        return topic

    def find_relevant_topic(
        self,
        user_text: str,
        *,
        statuses: Optional[Sequence[str]] = None,
        exclude_topic_id: Optional[str] = None,
        min_score: float = 2.0,
    ) -> Optional[TopicRecord]:
        """
        Encuentra un tema previo por coincidencia léxica conservadora.

        No modifica el estado. El llamador decide si debe reabrirlo.
        """
        self._validate_text(user_text, "user_text")

        allowed = set(
            statuses
            or (
                "active",
                "blocked",
                "paused",
            )
        )
        unknown = allowed - VALID_TOPIC_STATUSES
        if unknown:
            raise ValueError(
                f"statuses inválidos: {sorted(unknown)}"
            )

        query_terms = set(
            self._topic_terms(
                user_text
            )
        )
        if not query_terms:
            return None

        ranked = []

        for topic in self._topics.values():
            if topic.topic_id == exclude_topic_id:
                continue
            if topic.status not in allowed:
                continue

            topic_terms = set(
                self._topic_terms(
                    " ".join(
                        [
                            topic.title or "",
                            topic.summary or "",
                            topic.block_reason or "",
                            " ".join(
                                str(item)
                                for item in (
                                    topic.metadata.get(
                                        "anchors",
                                        [],
                                    )
                                    or []
                                )
                            ),
                        ]
                    )
                )
            )

            overlap = query_terms & topic_terms
            if not overlap:
                continue

            block_terms = set(
                self._topic_terms(
                    topic.block_reason
                    or ""
                )
            )
            block_overlap = (
                query_terms
                & block_terms
            )

            numeric_overlap = {
                token
                for token in overlap
                if token.isdigit()
            }

            score = (
                float(len(overlap))
                + 0.75 * float(
                    len(block_overlap)
                )
                + 0.25 * float(
                    len(numeric_overlap)
                )
            )

            # Un tema bloqueado que coincide con el dato que faltaba recibe
            # una ligera prioridad para facilitar su reanudación.
            if (
                topic.status == "blocked"
                and block_overlap
            ):
                score += 0.35

            ranked.append(
                (
                    score,
                    topic.updated_at,
                    topic,
                )
            )

        if not ranked:
            return None

        ranked.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        best_score, _, best_topic = (
            ranked[0]
        )

        if best_score < float(
            min_score
        ):
            return None

        best_topic.metadata[
            "last_match_score"
        ] = round(
            best_score,
            4,
        )
        return best_topic

    def get_topic(self, topic_id: str) -> Optional[TopicRecord]:
        return self._topics.get(topic_id)

    def active_topics(self) -> List[TopicRecord]:
        return [
            topic
            for topic in self._topics.values()
            if topic.status == "active"
        ]

    def resolved_topics(self) -> List[TopicRecord]:
        return [
            topic
            for topic in self._topics.values()
            if topic.status == "resolved"
        ]

    def _fallback_current_topic(
        self,
        *,
        exclude: Optional[str] = None,
    ) -> Optional[str]:
        """
        Devuelve el tema ACTIVE más reciente que siga disponible.

        Se usa cuando el tema actual se resuelve, bloquea o pausa.
        Así Quipu no pierde el hilo principal si todavía existe otro
        tema activo.
        """
        candidates = [
            topic
            for topic in self._topics.values()
            if topic.status == "active"
            and topic.topic_id != exclude
        ]

        if not candidates:
            return None

        candidates.sort(
            key=lambda topic: topic.updated_at,
            reverse=True,
        )
        return candidates[0].topic_id

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def accept_decision(
        self,
        decision_id: str,
        text: str,
        *,
        topic_id: Optional[str] = None,
        source: str = "user",
        resolve_topic: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DecisionRecord:
        self._validate_id(decision_id, "decision_id")
        self._validate_text(text, "text")

        if topic_id is not None:
            self._require_topic(topic_id)

        decision = DecisionRecord(
            decision_id=decision_id,
            text=text.strip(),
            topic_id=topic_id,
            accepted=True,
            source=source,
            metadata=dict(metadata or {}),
        )
        self._decisions[decision_id] = decision

        if resolve_topic and topic_id is not None:
            self.resolve_topic(topic_id, summary=text.strip())

        return decision

    def revoke_decision(self, decision_id: str) -> DecisionRecord:
        decision = self._require_decision(decision_id)
        decision.accepted = False
        return decision

    def accepted_decisions(
        self,
        *,
        topic_id: Optional[str] = None,
    ) -> List[DecisionRecord]:
        records = [
            d
            for d in self._decisions.values()
            if d.accepted
        ]

        if topic_id is not None:
            records = [
                d for d in records
                if d.topic_id == topic_id
            ]

        return records

    # ------------------------------------------------------------------
    # Pending questions
    # ------------------------------------------------------------------

    def add_pending_question(
        self,
        question_id: str,
        text: str,
        *,
        topic_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PendingQuestion:
        self._validate_id(question_id, "question_id")
        self._validate_text(text, "text")

        if topic_id is not None:
            self._require_topic(topic_id)

        question = PendingQuestion(
            question_id=question_id,
            text=text.strip(),
            topic_id=topic_id,
            status="pending",
            metadata=dict(metadata or {}),
        )
        self._pending[question_id] = question
        return question

    def resolve_pending_question(
        self,
        question_id: str,
        answer: str,
    ) -> PendingQuestion:
        question = self._require_pending(question_id)
        self._validate_text(answer, "answer")

        question.status = "resolved"
        question.answer = answer.strip()
        question.resolved_at = time.time()
        return question

    def pending_questions(self) -> List[PendingQuestion]:
        return [
            q for q in self._pending.values()
            if q.status == "pending"
        ]

    # ------------------------------------------------------------------
    # Hypotheses
    # ------------------------------------------------------------------

    def add_hypothesis(
        self,
        hypothesis_id: str,
        text: str,
        *,
        topic_id: Optional[str] = None,
        confidence: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HypothesisRecord:
        self._validate_id(hypothesis_id, "hypothesis_id")
        self._validate_text(text, "text")

        if topic_id is not None:
            self._require_topic(topic_id)

        hypothesis = HypothesisRecord(
            hypothesis_id=hypothesis_id,
            text=text.strip(),
            topic_id=topic_id,
            status="open",
            confidence=self._bounded(confidence),
            metadata=dict(metadata or {}),
        )
        self._hypotheses[hypothesis_id] = hypothesis
        return hypothesis

    def update_hypothesis(
        self,
        hypothesis_id: str,
        *,
        status: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> HypothesisRecord:
        hypothesis = self._require_hypothesis(hypothesis_id)

        if status is not None:
            status_norm = status.strip().lower()
            if status_norm not in VALID_HYPOTHESIS_STATUSES:
                raise ValueError(
                    f"status inválido: {status!r}. "
                    f"Valores válidos: {sorted(VALID_HYPOTHESIS_STATUSES)}"
                )
            hypothesis.status = status_norm

        if confidence is not None:
            hypothesis.confidence = self._bounded(confidence)

        hypothesis.updated_at = time.time()
        return hypothesis

    def open_hypotheses(self) -> List[HypothesisRecord]:
        return [
            h for h in self._hypotheses.values()
            if h.status == "open"
        ]

    # ------------------------------------------------------------------
    # Conversational policy
    # ------------------------------------------------------------------

    def should_repeat_topic(
        self,
        topic_id: str,
        *,
        user_explicitly_requested_recap: bool = False,
    ) -> bool:
        """
        Regla anti-perseveración.

        - active   -> sí puede desarrollarse
        - resolved -> no repetir salvo que el usuario pida recap/revisión
        - paused   -> no repetir
        - blocked  -> no repetir solución como si estuviera resuelta
        """
        topic = self._require_topic(topic_id)

        if user_explicitly_requested_recap:
            return True

        return topic.status == "active"

    def response_directives(self) -> Dict[str, Any]:
        """
        Directivas compactas para ExecutiveController / generador.
        """
        resolved_ids = [
            t.topic_id
            for t in self.resolved_topics()
        ]
        blocked_ids = [
            t.topic_id
            for t in self._topics.values()
            if t.status == "blocked"
        ]
        active_ids = [
            t.topic_id
            for t in self.active_topics()
        ]

        return {
            "current_topic_id": self.current_topic_id,
            "active_topics": active_ids,
            "resolved_topics": resolved_ids,
            "blocked_topics": blocked_ids,
            "pending_questions": [
                q.question_id
                for q in self.pending_questions()
            ],
            "open_hypotheses": [
                h.hypothesis_id
                for h in self.open_hypotheses()
            ],
            "avoid_repeating_resolved_topics": True,
            "do_not_present_blocked_topics_as_resolved": True,
        }

    def apply_user_feedback(
        self,
        topic_id: str,
        user_text: str,
        *,
        accepted_decision_id: Optional[str] = None,
        accepted_decision_text: Optional[str] = None,
    ) -> str:
        """
        Regla determinista y conservadora para feedback simple.

        Devuelve:
          "resolved"
          "reopened"
          "no_change"

        No intenta interpretar feedback ambiguo.
        """
        topic = self._require_topic(topic_id)
        self._validate_text(user_text, "user_text")

        normalized = self._normalize(user_text)

        if any(marker in normalized for marker in self.REOPEN_MARKERS):
            self.reopen_topic(topic_id)
            return "reopened"

        if any(marker in normalized for marker in self.APPROVAL_MARKERS):
            if accepted_decision_id and accepted_decision_text:
                self.accept_decision(
                    accepted_decision_id,
                    accepted_decision_text,
                    topic_id=topic_id,
                    resolve_topic=True,
                )
            else:
                self.resolve_topic(topic_id)
            return "resolved"

        return "no_change"

    # ------------------------------------------------------------------
    # CognitiveState integration
    # ------------------------------------------------------------------

    def export_state(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "current_topic_id": self.current_topic_id,
            "topics": {
                key: value.snapshot()
                for key, value in self._topics.items()
            },
            "decisions": {
                key: value.snapshot()
                for key, value in self._decisions.items()
            },
            "pending_questions": {
                key: value.snapshot()
                for key, value in self._pending.items()
            },
            "hypotheses": {
                key: value.snapshot()
                for key, value in self._hypotheses.items()
            },
            "directives": self.response_directives(),
        }

    def sync_to_cognitive_state(
        self,
        state: CognitiveState,
    ) -> CognitiveState:
        if not isinstance(state, CognitiveState):
            raise TypeError("state debe ser CognitiveState.")

        state.conversation_state = self.export_state()

        state.turn_metadata.setdefault("conversation_state_manager", {})
        state.turn_metadata["conversation_state_manager"].update(
            {
                "version": self.VERSION,
                "current_topic_id": self.current_topic_id,
                "active_topic_count": len(self.active_topics()),
                "resolved_topic_count": len(self.resolved_topics()),
                "pending_question_count": len(self.pending_questions()),
                "open_hypothesis_count": len(self.open_hypotheses()),
            }
        )

        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_topic(self, topic_id: str) -> TopicRecord:
        self._validate_id(topic_id, "topic_id")
        try:
            return self._topics[topic_id]
        except KeyError as exc:
            raise KeyError(f"Topic no encontrado: {topic_id!r}") from exc

    def _require_decision(self, decision_id: str) -> DecisionRecord:
        self._validate_id(decision_id, "decision_id")
        try:
            return self._decisions[decision_id]
        except KeyError as exc:
            raise KeyError(f"Decision no encontrada: {decision_id!r}") from exc

    def _require_pending(self, question_id: str) -> PendingQuestion:
        self._validate_id(question_id, "question_id")
        try:
            return self._pending[question_id]
        except KeyError as exc:
            raise KeyError(f"Pregunta pendiente no encontrada: {question_id!r}") from exc

    def _require_hypothesis(self, hypothesis_id: str) -> HypothesisRecord:
        self._validate_id(hypothesis_id, "hypothesis_id")
        try:
            return self._hypotheses[hypothesis_id]
        except KeyError as exc:
            raise KeyError(f"Hipótesis no encontrada: {hypothesis_id!r}") from exc

    @classmethod
    def _topic_terms(
        cls,
        text: str,
    ) -> List[str]:
        normalized = unicodedata.normalize(
            "NFKD",
            str(text or "").lower(),
        )
        normalized = "".join(
            ch
            for ch in normalized
            if not unicodedata.combining(
                ch
            )
        )

        tokens = re.findall(
            r"[a-z0-9]+",
            normalized,
        )

        return [
            token
            for token in tokens
            if (
                token.isdigit()
                or (
                    len(token) >= 3
                    and token
                    not in cls.TOPIC_STOPWORDS
                )
            )
        ]

    @classmethod
    def _merge_topic_anchors(
        cls,
        topic: TopicRecord,
        text: str,
    ) -> None:
        anchors = list(
            topic.metadata.get(
                "anchors",
                [],
            )
            or []
        )

        for token in cls._topic_terms(
            text
        ):
            if token not in anchors:
                anchors.append(
                    token
                )

        topic.metadata["anchors"] = (
            anchors[-64:]
        )

    @staticmethod
    def _validate_id(value: str, name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} debe ser un string no vacío.")

    @staticmethod
    def _validate_text(value: str, name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} debe ser un string no vacío.")

    @staticmethod
    def _bounded(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number))

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.strip().lower().split())
