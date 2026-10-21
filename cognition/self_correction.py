from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .cognitive_state import CognitiveState


@dataclass
class CorrectionPlan:
    """
    Plan interno para rehacer una respuesta desde sus premisas.
    """

    can_continue: bool
    action: str

    attempt: int = 0
    max_attempts: int = 2

    issues: List[str] = field(default_factory=list)
    premises: Dict[str, Any] = field(default_factory=dict)

    missing_information: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)

    must_recompute: bool = True
    reuse_previous_answer: bool = False

    reason: str = ""

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CorrectionOutcome:
    accepted: bool
    action: str

    attempt: int
    answer: Optional[str] = None
    verification_passed: bool = False

    reason: str = ""

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


class SelfCorrectionLoop:
    """
    Quipu Cognitive Control 1.0
    SelfCorrectionLoop v1.0.0

    Objetivo:
      - Evitar que "corregir" signifique únicamente volver a redactar.
      - Forzar una recomputación desde las premisas vigentes.
      - No reutilizar silenciosamente la respuesta previa.
      - Bloquear la corrección si faltan datos o hay conflictos relevantes.
      - Exigir verificación antes de aceptar una respuesta corregida.
      - Limitar el número de intentos para evitar bucles infinitos.

    Este módulo NO genera texto con el LLM.
    Coordina el ciclo:

        error detectado
            ↓
        archivar respuesta anterior
            ↓
        limpiar draft
            ↓
        reconstruir desde premisas
            ↓
        generar nueva candidata externamente
            ↓
        verificar
            ↓
        aceptar / reintentar / detener
    """

    VERSION = "1.0.0"

    ACTION_RECOMPUTE = "recompute_from_premises"
    ACTION_ASK_FOR_DATA = "ask_for_data"
    ACTION_RESOLVE_CONFLICT = "resolve_conflict"
    ACTION_RETRY = "retry"
    ACTION_ACCEPT = "accept"
    ACTION_STOP = "stop"

    def __init__(self, *, max_attempts: int = 2) -> None:
        if int(max_attempts) < 1:
            raise ValueError("max_attempts debe ser >= 1.")
        self.max_attempts = int(max_attempts)

    def begin(
        self,
        state: CognitiveState,
        *,
        issues: Optional[Sequence[str]] = None,
        required_keys: Optional[Sequence[str]] = None,
    ) -> CorrectionPlan:
        """
        Inicia un intento de autocorrección.

        Regla central:
          la respuesta anterior se archiva para auditoría,
          pero NO se entrega como contexto de regeneración.
        """
        if not isinstance(state, CognitiveState):
            raise TypeError("state debe ser CognitiveState.")

        issues_list = self._dedupe(issues or [])

        if state.missing_information:
            plan = CorrectionPlan(
                can_continue=False,
                action=self.ACTION_ASK_FOR_DATA,
                attempt=state.correction_attempts,
                max_attempts=self.max_attempts,
                issues=issues_list,
                missing_information=list(state.missing_information),
                conflicts=list(state.conflicts),
                reason=(
                    "La autocorrección está bloqueada porque faltan datos. "
                    "No se debe regenerar inventando información."
                ),
            )
            return self._record_plan(state, plan)

        if state.conflicts:
            plan = CorrectionPlan(
                can_continue=False,
                action=self.ACTION_RESOLVE_CONFLICT,
                attempt=state.correction_attempts,
                max_attempts=self.max_attempts,
                issues=issues_list,
                missing_information=[],
                conflicts=list(state.conflicts),
                reason=(
                    "La autocorrección está bloqueada por conflictos activos "
                    "que deben resolverse antes de recomputar."
                ),
            )
            return self._record_plan(state, plan)

        if state.correction_attempts >= self.max_attempts:
            plan = CorrectionPlan(
                can_continue=False,
                action=self.ACTION_STOP,
                attempt=state.correction_attempts,
                max_attempts=self.max_attempts,
                issues=issues_list,
                reason=(
                    "Se alcanzó el máximo de intentos de autocorrección. "
                    "Debe escalarse a revisión externa o solicitar más contexto."
                ),
            )
            return self._record_plan(state, plan)

        premises = self._collect_premises(
            state,
            required_keys=required_keys,
        )

        previous_answer = state.draft_answer
        previous_verification = state.verification_result

        state.correction_attempts += 1

        # Guardar la respuesta previa SOLO para auditoría.
        self._archive_attempt(
            state,
            attempt=state.correction_attempts,
            previous_answer=previous_answer,
            previous_verification=previous_verification,
            issues=issues_list,
            premises=premises,
        )

        # Limpiar el borrador obliga al flujo superior a generar una nueva
        # solución desde las premisas, no a "parchar" la anterior.
        state.draft_answer = None
        state.verification_result = None

        plan = CorrectionPlan(
            can_continue=True,
            action=self.ACTION_RECOMPUTE,
            attempt=state.correction_attempts,
            max_attempts=self.max_attempts,
            issues=issues_list,
            premises=premises,
            missing_information=[],
            conflicts=[],
            must_recompute=True,
            reuse_previous_answer=False,
            reason=(
                "Recomputar desde las premisas vigentes y verificar la nueva "
                "respuesta antes de aceptarla."
            ),
        )

        return self._record_plan(state, plan)

    def build_regeneration_context(
        self,
        plan: CorrectionPlan,
    ) -> Dict[str, Any]:
        """
        Contexto seguro para una nueva generación.

        Intencionalmente NO incluye la respuesta anterior.
        """
        if not isinstance(plan, CorrectionPlan):
            raise TypeError("plan debe ser CorrectionPlan.")

        return {
            "mode": self.ACTION_RECOMPUTE,
            "attempt": plan.attempt,
            "issues_to_fix": list(plan.issues),
            "premises": dict(plan.premises),
            "rules": {
                "must_recompute": plan.must_recompute,
                "reuse_previous_answer": plan.reuse_previous_answer,
                "must_verify_after_generation": True,
            },
        }

    def submit_candidate(
        self,
        state: CognitiveState,
        candidate_answer: str,
        verification_result: Any,
    ) -> CorrectionOutcome:
        """
        Recibe la nueva candidata después de que un razonador externo la generó.

        La candidata solo se acepta si el resultado de verificación indica éxito.
        """
        if not isinstance(state, CognitiveState):
            raise TypeError("state debe ser CognitiveState.")

        if not isinstance(candidate_answer, str) or not candidate_answer.strip():
            raise ValueError("candidate_answer debe ser un string no vacío.")

        passed = self._verification_passed(verification_result)

        state.verification_result = self._normalize_verification(
            verification_result
        )

        if passed:
            state.draft_answer = candidate_answer.strip()

            outcome = CorrectionOutcome(
                accepted=True,
                action=self.ACTION_ACCEPT,
                attempt=state.correction_attempts,
                answer=state.draft_answer,
                verification_passed=True,
                reason=(
                    "La respuesta corregida superó la verificación y puede "
                    "sustituir a la conclusión anterior."
                ),
            )
            return self._record_outcome(state, outcome)

        # Guardamos la candidata fallida para auditoría, pero NO la convertimos
        # en respuesta válida.
        self._archive_failed_candidate(
            state,
            candidate_answer=candidate_answer.strip(),
            verification_result=state.verification_result,
        )
        state.draft_answer = None

        if state.correction_attempts >= self.max_attempts:
            outcome = CorrectionOutcome(
                accepted=False,
                action=self.ACTION_STOP,
                attempt=state.correction_attempts,
                answer=None,
                verification_passed=False,
                reason=(
                    "La candidata corregida volvió a fallar y se alcanzó el "
                    "máximo de intentos."
                ),
            )
            return self._record_outcome(state, outcome)

        outcome = CorrectionOutcome(
            accepted=False,
            action=self.ACTION_RETRY,
            attempt=state.correction_attempts,
            answer=None,
            verification_passed=False,
            reason=(
                "La candidata no superó la verificación. Debe iniciarse un "
                "nuevo intento desde las premisas."
            ),
        )
        return self._record_outcome(state, outcome)

    # ------------------------------------------------------------------
    # Premisas
    # ------------------------------------------------------------------

    def _collect_premises(
        self,
        state: CognitiveState,
        *,
        required_keys: Optional[Sequence[str]],
    ) -> Dict[str, Any]:
        if required_keys:
            result: Dict[str, Any] = {}
            for key in required_keys:
                if key in state.world_state:
                    result[key] = state.world_state[key]
            return result

        return dict(state.world_state)

    # ------------------------------------------------------------------
    # Auditoría
    # ------------------------------------------------------------------

    def _archive_attempt(
        self,
        state: CognitiveState,
        *,
        attempt: int,
        previous_answer: Optional[str],
        previous_verification: Any,
        issues: List[str],
        premises: Dict[str, Any],
    ) -> None:
        root = state.turn_metadata.setdefault("self_correction", {})
        root["version"] = self.VERSION
        root["max_attempts"] = self.max_attempts
        history = root.setdefault("history", [])

        history.append(
            {
                "attempt": attempt,
                "previous_answer": previous_answer,
                "previous_verification": previous_verification,
                "issues": list(issues),
                "premises": dict(premises),
            }
        )

    def _archive_failed_candidate(
        self,
        state: CognitiveState,
        *,
        candidate_answer: str,
        verification_result: Dict[str, Any],
    ) -> None:
        root = state.turn_metadata.setdefault("self_correction", {})
        failures = root.setdefault("failed_candidates", [])
        failures.append(
            {
                "attempt": state.correction_attempts,
                "candidate_answer": candidate_answer,
                "verification_result": dict(verification_result),
            }
        )

    def _record_plan(
        self,
        state: CognitiveState,
        plan: CorrectionPlan,
    ) -> CorrectionPlan:
        root = state.turn_metadata.setdefault("self_correction", {})
        root["version"] = self.VERSION
        root["last_plan"] = plan.snapshot()
        return plan

    def _record_outcome(
        self,
        state: CognitiveState,
        outcome: CorrectionOutcome,
    ) -> CorrectionOutcome:
        root = state.turn_metadata.setdefault("self_correction", {})
        root["version"] = self.VERSION
        root["last_outcome"] = outcome.snapshot()
        return outcome

    # ------------------------------------------------------------------
    # Verifier interoperability
    # ------------------------------------------------------------------

    @staticmethod
    def _verification_passed(result: Any) -> bool:
        if isinstance(result, bool):
            return result

        if isinstance(result, dict):
            for key in ("passed", "valid", "ok", "success", "verified"):
                if key in result:
                    return bool(result[key])

            status = str(result.get("status", "")).strip().lower()
            if status in {"pass", "passed", "valid", "ok", "success", "verified"}:
                return True
            if status in {"fail", "failed", "invalid", "error", "rejected"}:
                return False

        return False

    @staticmethod
    def _normalize_verification(result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return dict(result)
        if isinstance(result, bool):
            return {"passed": result}
        return {
            "passed": False,
            "raw": str(result),
        }

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
