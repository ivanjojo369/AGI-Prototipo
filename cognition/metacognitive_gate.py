from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .cognitive_state import CognitiveState


@dataclass
class MetacognitiveDecision:
    """
    Resultado de la evaluación epistemológica previa a responder.
    """

    can_answer: bool
    status: str

    required_keys: List[str] = field(default_factory=list)
    known_keys: List[str] = field(default_factory=list)
    missing_keys: List[str] = field(default_factory=list)
    conflicting_keys: List[str] = field(default_factory=list)

    assumptions: List[str] = field(default_factory=list)

    evidence_completeness: float = 0.0
    evidence_confidence: float = 0.0
    epistemic_confidence: float = 0.0

    recommended_action: str = "answer"
    reason: str = ""

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


class MetacognitiveGate:
    """
    Quipu Cognitive Control 1.0
    MetacognitiveGate v1.0.2

    Objetivo:
      - Decidir si Quipu tiene información suficiente para responder.
      - Impedir que un hueco de información se convierta en una cifra inventada.
      - Bloquear respuestas cuando existe un conflicto activo relevante.
      - Separar:
          * "sé este dato"
          * "me falta este dato"
          * "hay datos en conflicto"
          * "estoy suponiendo algo"
      - Dejar una decisión auditable en CognitiveState.

    CAMBIO 1.0.2:
      - La confianza epistemológica usa completeness * evidence_confidence.
      - Se conserva el control de weakest_fact_confidence.
      - Se registra FORMULA_ID para auditoría.

    IMPORTANTE:
      Esta clase NO resuelve el problema ni genera la respuesta final.
      Solo decide si el razonamiento tiene permiso epistemológico para continuar.
    """

    VERSION = "1.0.2"
    FORMULA_ID = "multiplicative_confidence_v1"

    STATUS_READY = "ready"
    STATUS_INSUFFICIENT = "insufficient_information"
    STATUS_CONFLICT = "conflict"
    STATUS_REVIEW_REQUIRED = "review_required"

    ACTION_ANSWER = "answer"
    ACTION_ASK_FOR_DATA = "ask_for_data"
    ACTION_RESOLVE_CONFLICT = "resolve_conflict"
    ACTION_REVIEW = "review"

    def __init__(
        self,
        *,
        min_epistemic_confidence: float = 0.70,
        min_fact_confidence: float = 0.50,
        block_on_conflicts: bool = True,
    ) -> None:
        self.min_epistemic_confidence = self._bounded(
            min_epistemic_confidence
        )
        self.min_fact_confidence = self._bounded(
            min_fact_confidence
        )
        self.block_on_conflicts = bool(block_on_conflicts)

    def assess(
        self,
        state: CognitiveState,
        *,
        required_keys: Optional[Sequence[str]] = None,
        relevant_conflict_keys: Optional[Sequence[str]] = None,
        assumptions: Optional[Sequence[str]] = None,
        fact_confidence: Optional[Dict[str, float]] = None,
    ) -> MetacognitiveDecision:
        if not isinstance(state, CognitiveState):
            raise TypeError("state debe ser CognitiveState.")

        required = self._dedupe(required_keys or [])
        assumptions_list = self._dedupe(
            list(state.assumptions) + list(assumptions or [])
        )

        known: List[str] = []
        missing: List[str] = []

        for key in required:
            if self._is_known(state.world_state, key):
                known.append(key)
            else:
                missing.append(key)

        # Incorporar huecos ya detectados por módulos anteriores.
        self._merge_missing_from_state(state, required, missing)

        # Determinar conflictos relevantes.
        conflicts = self._detect_conflicts(
            state,
            required_keys=required,
            explicit_keys=relevant_conflict_keys,
        )

        completeness = self._completeness(required, known, missing)

        evidence_confidence = self._evidence_confidence(
            known,
            fact_confidence=fact_confidence,
            state=state,
        )

        # Las suposiciones reducen la confianza, pero no necesariamente bloquean.
        assumption_penalty = min(0.35, 0.08 * len(assumptions_list))

        # Ambas dimensiones son necesarias:
        # la completitud NO puede compensar evidencia de baja confianza.
        epistemic_confidence = max(
            0.0,
            min(
                1.0,
                (completeness * evidence_confidence)
                - assumption_penalty,
            ),
        )

        # Prioridad 1: falta información requerida.
        if missing:
            decision = MetacognitiveDecision(
                can_answer=False,
                status=self.STATUS_INSUFFICIENT,
                required_keys=required,
                known_keys=known,
                missing_keys=missing,
                conflicting_keys=conflicts,
                assumptions=assumptions_list,
                evidence_completeness=round(completeness, 4),
                evidence_confidence=round(evidence_confidence, 4),
                epistemic_confidence=round(epistemic_confidence, 4),
                recommended_action=self.ACTION_ASK_FOR_DATA,
                reason=(
                    "Falta información necesaria para sostener una "
                    "conclusión sin inventar datos."
                ),
            )
            return self._record(state, decision)

        # Prioridad 2: conflicto relevante.
        if self.block_on_conflicts and conflicts:
            decision = MetacognitiveDecision(
                can_answer=False,
                status=self.STATUS_CONFLICT,
                required_keys=required,
                known_keys=known,
                missing_keys=[],
                conflicting_keys=conflicts,
                assumptions=assumptions_list,
                evidence_completeness=round(completeness, 4),
                evidence_confidence=round(evidence_confidence, 4),
                epistemic_confidence=round(epistemic_confidence, 4),
                recommended_action=self.ACTION_RESOLVE_CONFLICT,
                reason=(
                    "Hay información relevante en conflicto y debe "
                    "resolverse antes de responder como si hubiera un "
                    "único estado válido."
                ),
            )
            return self._record(state, decision)

        # Prioridad 3: evidencia completa pero confianza demasiado baja.
        #
        # No basta con que "completeness=1.0" eleve el promedio total:
        # un hecho crítico con confianza muy baja debe forzar revisión.
        weakest_fact_confidence = self._weakest_fact_confidence(
            known,
            fact_confidence=fact_confidence,
            state=state,
        )

        if (
            epistemic_confidence < self.min_epistemic_confidence
            or weakest_fact_confidence < self.min_fact_confidence
        ):
            decision = MetacognitiveDecision(
                can_answer=False,
                status=self.STATUS_REVIEW_REQUIRED,
                required_keys=required,
                known_keys=known,
                missing_keys=[],
                conflicting_keys=conflicts,
                assumptions=assumptions_list,
                evidence_completeness=round(completeness, 4),
                evidence_confidence=round(evidence_confidence, 4),
                epistemic_confidence=round(epistemic_confidence, 4),
                recommended_action=self.ACTION_REVIEW,
                reason=(
                    "La información requerida está presente, pero la "
                    "confianza epistemológica es insuficiente; se requiere "
                    "revisión o verificación."
                ),
            )
            return self._record(state, decision)

        decision = MetacognitiveDecision(
            can_answer=True,
            status=self.STATUS_READY,
            required_keys=required,
            known_keys=known,
            missing_keys=[],
            conflicting_keys=conflicts,
            assumptions=assumptions_list,
            evidence_completeness=round(completeness, 4),
            evidence_confidence=round(evidence_confidence, 4),
            epistemic_confidence=round(epistemic_confidence, 4),
            recommended_action=self.ACTION_ANSWER,
            reason="Hay información suficiente y no hay conflictos bloqueantes.",
        )
        return self._record(state, decision)

    def require(
        self,
        state: CognitiveState,
        required_keys: Sequence[str],
        **kwargs: Any,
    ) -> MetacognitiveDecision:
        """
        Alias semántico para tareas que declaran explícitamente
        las variables necesarias para poder responder.
        """
        return self.assess(
            state,
            required_keys=required_keys,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Reglas internas
    # ------------------------------------------------------------------

    @staticmethod
    def _is_known(world_state: Dict[str, Any], key: str) -> bool:
        if key not in world_state:
            return False

        value = world_state.get(key)

        # None representa desconocido.
        if value is None:
            return False

        # Strings vacíos tampoco cuentan como conocimiento útil.
        if isinstance(value, str) and not value.strip():
            return False

        return True

    def _merge_missing_from_state(
        self,
        state: CognitiveState,
        required: Sequence[str],
        missing: List[str],
    ) -> None:
        """
        Reconoce mensajes generados por LongRangeRetriever del tipo:
          "Falta información requerida: <key>"
        """
        prefix = "Falta información requerida: "

        for item in state.missing_information:
            if not isinstance(item, str):
                continue

            if item.startswith(prefix):
                key = item[len(prefix):].strip()
                if key and (not required or key in required):
                    if key not in missing:
                        missing.append(key)

    def _detect_conflicts(
        self,
        state: CognitiveState,
        *,
        required_keys: Sequence[str],
        explicit_keys: Optional[Sequence[str]],
    ) -> List[str]:
        """
        Extrae keys conflictivas desde CognitiveState.conflicts.

        Formatos ya emitidos por módulos anteriores:
          "Conflicto activo en world_state: planet.climate.dominant"
          "Conflicto activo en planet.climate.dominant: ..."

        También puede recibir explicit_keys para no depender del formato textual.
        """
        found = set(self._dedupe(explicit_keys or []))

        for item in state.conflicts:
            if not isinstance(item, str):
                continue

            text = item.strip()

            marker_a = "Conflicto activo en world_state:"
            marker_b = "Conflicto activo en "

            if text.startswith(marker_a):
                key = text[len(marker_a):].strip()
                if key:
                    found.add(key)
                continue

            if text.startswith(marker_b):
                rest = text[len(marker_b):].strip()
                key = rest.split(":", 1)[0].strip()
                if key:
                    found.add(key)

        if required_keys:
            return sorted(k for k in found if k in required_keys)

        return sorted(found)

    def _weakest_fact_confidence(
        self,
        known_keys: Sequence[str],
        *,
        fact_confidence: Optional[Dict[str, float]],
        state: CognitiveState,
    ) -> float:
        """
        Confianza del hecho requerido más débil.

        Evita que una completitud perfecta o varios hechos fuertes oculten
        un dato crítico de muy baja confianza.
        """
        if not known_keys:
            return 0.0

        source = dict(
            state.turn_metadata.get("fact_confidence", {})
            if isinstance(state.turn_metadata, dict)
            else {}
        )

        if fact_confidence:
            source.update(fact_confidence)

        values = [
            self._bounded(source.get(key, 1.0))
            for key in known_keys
        ]
        return min(values) if values else 0.0

    def _evidence_confidence(
        self,
        known_keys: Sequence[str],
        *,
        fact_confidence: Optional[Dict[str, float]],
        state: CognitiveState,
    ) -> float:
        """
        Confianza media de la evidencia.

        Puede recibir fact_confidence explícito o leer:
          state.turn_metadata["fact_confidence"]

        Si no existe metadata de confianza, los hechos presentes se consideran
        utilizables con confianza base 1.0. Esto evita confundir la confianza
        del clasificador de intención con la confianza epistemológica.
        """
        if not known_keys:
            return 0.0

        source = dict(
            state.turn_metadata.get("fact_confidence", {})
            if isinstance(state.turn_metadata, dict)
            else {}
        )

        if fact_confidence:
            source.update(fact_confidence)

        values: List[float] = []
        for key in known_keys:
            if key in source:
                values.append(self._bounded(source[key]))
            else:
                values.append(1.0)

        if not values:
            return 0.0

        return sum(values) / len(values)

    @staticmethod
    def _completeness(
        required: Sequence[str],
        known: Sequence[str],
        missing: Sequence[str],
    ) -> float:
        if not required:
            # Si la tarea no declaró dependencias, no penalizar por ausencia
            # de requirements explícitos.
            return 1.0

        total = len(set(required))
        if total <= 0:
            return 1.0

        known_count = len(set(known) - set(missing))
        return max(0.0, min(1.0, known_count / total))

    def _record(
        self,
        state: CognitiveState,
        decision: MetacognitiveDecision,
    ) -> MetacognitiveDecision:
        state.turn_metadata.setdefault("metacognitive_gate", {})
        state.turn_metadata["metacognitive_gate"].update(
            {
                "version": self.VERSION,
                "formula_id": self.FORMULA_ID,
                **decision.snapshot(),
            }
        )

        for key in decision.missing_keys:
            message = f"Falta información requerida: {key}"
            if message not in state.missing_information:
                state.missing_information.append(message)

        return decision

    @staticmethod
    def _dedupe(values: Iterable[str]) -> List[str]:
        result: List[str] = []
        seen = set()

        for value in values:
            if not isinstance(value, str):
                continue

            item = value.strip()
            if not item or item in seen:
                continue

            seen.add(item)
            result.append(item)

        return result

    @staticmethod
    def _bounded(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, min(1.0, number))
