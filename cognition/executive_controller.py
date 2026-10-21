from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional

from .cognitive_state import CognitiveState


class ExecutiveController:
    """Primera capa de control ejecutivo de Quipu Cognitive Control 1.0."""

    VERSION = "1.0.0"

    LONG_RANGE_MARKERS = (
        "con toda la informacion",
        "todo el chat",
        "toda la conversacion",
        "a lo largo de la conversacion",
        "anteriormente",
        "desde el principio",
        "antes dijiste",
        "que dijiste antes",
        "sigue vigente",
        "siguen vigentes",
        "fue reemplazado",
        "fueron reemplazados",
        "historial",
        "aquel dato",
        "aquella informacion",
    )

    MEMORY_MARKERS = (
        "que recuerdas",
        "recuerdas",
        "memoria",
        "persistencia de memoria",
        "que sabes sobre",
        "que sabes de",
    )

    REVIEW_MARKERS = (
        "revisa",
        "comprueba",
        "verifica",
        "corrige",
        "corregir",
        "error",
        "fallo",
        "equivocaste",
        "contradiccion",
        "contradicciones",
        "inconsistencia",
        "inconsistencias",
        "chocan",
        "audita",
    )

    PLANNING_MARKERS = (
        "planifica",
        "planificar",
        "plan",
        "estrategia",
        "pasos",
        "como lo harias",
        "como harias",
        "disena",
        "organiza",
        "organizar",
        "hoja de ruta",
    )

    CALCULATION_MARKERS = (
        "calcula",
        "calcular",
        "cuanto",
        "cuanta",
        "cuantos",
        "cuantas",
        "porcentaje",
        "por ciento",
        "kg",
        "kilogramos",
        "toneladas",
        "dias",
        "meses",
        "anos",
    )

    TOOL_MARKERS = (
        "archivo",
        "carpeta",
        "web",
        "internet",
        "buscar",
        "busca",
        "python",
        "codigo",
    )

    def analyze(
        self,
        user_query: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        world_state: Optional[Dict[str, Any]] = None,
        conversation_state: Optional[Dict[str, Any]] = None,
    ) -> CognitiveState:
        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError("user_query debe ser un string no vacío.")

        state = CognitiveState(
            user_query=user_query.strip(),
            recent_context=list(recent_context or []),
            world_state=dict(world_state or {}),
            conversation_state=dict(conversation_state or {}),
        )

        normalized = self._normalize(user_query)

        long_range = self._contains_any(normalized, self.LONG_RANGE_MARKERS)
        memory = self._contains_any(normalized, self.MEMORY_MARKERS)
        review = self._contains_any(normalized, self.REVIEW_MARKERS)
        planning = self._contains_any(normalized, self.PLANNING_MARKERS)
        calculation = self._looks_like_calculation(normalized)
        tools = self._contains_any(normalized, self.TOOL_MARKERS)

        if review and self._references_prior_claim(normalized):
            long_range = True

        state.requires_long_range = long_range
        state.requires_memory = memory or long_range
        state.requires_self_review = review
        state.requires_planning = planning
        state.requires_calculation = calculation
        state.requires_tools = tools
        state.requires_verification = True

        if state.requires_long_range:
            state.intent = "long_range_reasoning"
            state.add_tool("long_range_retriever")
            state.add_tool("working_world_state")
            state.add_tool("memory_relevance_gate")
        elif state.requires_self_review:
            state.intent = "self_review"
            state.add_tool("self_correction_loop")
        elif state.requires_memory:
            state.intent = "memory_lookup"
            state.add_tool("memory_relevance_gate")
        elif state.requires_planning:
            state.intent = "planning"
            state.add_tool("planner")
        elif state.requires_calculation:
            state.intent = "calculation"
            state.add_tool("calculator")
        else:
            state.intent = "chat"

        if state.requires_memory:
            self._add_secondary(state, "memory")
        if state.requires_long_range:
            self._add_secondary(state, "long_range")
        if state.requires_calculation:
            self._add_secondary(state, "calculation")
            state.add_tool("calculator")
        if state.requires_planning:
            self._add_secondary(state, "planning")
            state.add_tool("planner")
        if state.requires_self_review:
            self._add_secondary(state, "self_review")
            state.add_tool("self_correction_loop")
        if state.requires_tools:
            self._add_secondary(state, "external_tools")

        state.add_tool("metacognitive_gate")
        state.add_tool("verifier")
        state.add_tool("output_guard")

        state.decision_confidence = self._decision_confidence(state)
        state.turn_metadata.update(
            {
                "controller_version": self.VERSION,
                "normalized_query": normalized,
                "detected": {
                    "memory": state.requires_memory,
                    "long_range": state.requires_long_range,
                    "calculation": state.requires_calculation,
                    "planning": state.requires_planning,
                    "self_review": state.requires_self_review,
                    "external_tools": state.requires_tools,
                },
            }
        )
        return state

    def _looks_like_calculation(self, normalized: str) -> bool:
        if self._contains_any(normalized, self.CALCULATION_MARKERS):
            return True
        if re.search(r"\d", normalized) and re.search(r"[%+\-*/=]", normalized):
            return True
        numbers = re.findall(r"\d+(?:[.,]\d+)?", normalized)
        return len(numbers) >= 2

    @staticmethod
    def _references_prior_claim(normalized: str) -> bool:
        prior_markers = (
            "dijiste",
            "antes",
            "anterior",
            "esa cifra",
            "ese dato",
            "esa respuesta",
            "tu conclusion",
            "tu respuesta",
        )
        return any(marker in normalized for marker in prior_markers)

    @staticmethod
    def _contains_any(text: str, markers: Iterable[str]) -> bool:
        return any(marker in text for marker in markers)

    @staticmethod
    def _add_secondary(state: CognitiveState, intent: str) -> None:
        if intent not in state.secondary_intents:
            state.secondary_intents.append(intent)

    @staticmethod
    def _decision_confidence(state: CognitiveState) -> float:
        signals = sum(
            [
                state.requires_memory,
                state.requires_long_range,
                state.requires_calculation,
                state.requires_planning,
                state.requires_self_review,
                state.requires_tools,
            ]
        )
        if state.intent == "chat" and signals == 0:
            return 0.70
        if signals == 1:
            return 0.82
        if signals == 2:
            return 0.90
        if signals >= 3:
            return 0.95
        return 0.75

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.strip().lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"\s+", " ", text)
        return text
