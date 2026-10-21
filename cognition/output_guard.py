from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import re
import unicodedata

from .cognitive_state import CognitiveState


@dataclass
class OutputGuardDecision:
    """
    Resultado de la inspección final de una respuesta.
    """

    accepted: bool
    action: str

    output: Optional[str] = None
    issues: List[str] = field(default_factory=list)

    expected_language: str = "es"
    detected_language: str = "unknown"

    sanitized: bool = False
    removed_lines: List[str] = field(default_factory=list)

    reason: str = ""

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


class OutputGuard:
    """
    Quipu Cognitive Control 1.0
    OutputGuard v1.0.0

    Última capa antes de mostrar una respuesta al usuario.

    OBJETIVO
      - Evitar filtraciones de metadata/control interno.
      - Detectar deriva clara de idioma.
      - Detectar salidas crudas tipo key=value que parecen estado interno.
      - Eliminar líneas de control conocidas cuando existe una respuesta
        natural válida alrededor.
      - Pedir regeneración cuando la salida queda contaminada o vacía.

    NO hace razonamiento profundo.
    NO corrige matemáticas.
    NO sustituye al Verifier.
    """

    VERSION = "1.0.0"

    ACTION_ACCEPT = "accept"
    ACTION_ACCEPT_SANITIZED = "accept_sanitized"
    ACTION_REGENERATE = "regenerate"
    ACTION_STOP = "stop"

    # Tokens/líneas internas que nunca deberían aparecer como respuesta final.
    INTERNAL_EXACT_TOKENS = {
        "ejercicios_memoria",
        "memory_lookup",
        "long_range_reasoning",
        "self_review",
        "python_exec",
        "project_plan",
        "filesystem_read",
        "search_web",
        "memory_relevance_gate",
        "long_range_retriever",
        "working_world_state",
        "metacognitive_gate",
        "self_correction_loop",
        "conversation_state_manager",
        "output_guard",
    }

    INTERNAL_LINE_PREFIXES = (
        "intent:",
        "secondary:",
        "tools:",
        "route:",
        "router:",
        "controller:",
        "controller_version:",
        "turn_metadata:",
        "memory_candidates:",
        "accepted_memories:",
        "rejected_memories:",
        "verification_result:",
        "required_keys:",
        "missing_keys:",
        "formula_id:",
    )

    RAW_METADATA_KEYS = {
        "proyecto",
        "lenguaje",
        "estado",
        "especialidad",
        "objetivo",
        "condicion",
        "condición",
        "siguiente_paso",
        "siguiente paso",
        "memory_type",
        "retrieval_score",
    }

    # Marcadores conservadores de portugués. No intentamos hacer detección
    # lingüística general; solo capturar la deriva que ya observamos.
    PORTUGUESE_MARKERS = {
        "não",
        "nao",
        "população",
        "populacao",
        "pessoas",
        "dados",
        "fornecidos",
        "necessário",
        "necessario",
        "precisaria",
        "seria",
        "possível",
        "possivel",
        "então",
        "entao",
        "também",
        "tambem",
        "para calcular a",
        "com base nos",
    }

    SPANISH_MARKERS = {
        "que",
        "para",
        "con",
        "los",
        "las",
        "una",
        "unos",
        "unas",
        "datos",
        "información",
        "informacion",
        "puede",
        "puedo",
        "porque",
        "entonces",
        "también",
        "tambien",
        "sería",
        "seria",
        "necesario",
        "posible",
    }

    RAW_REQUEST_MARKERS = (
        "literal",
        "exactamente",
        "sin resumir",
        "raw",
        "metadata",
        "metadatos",
        "clave valor",
        "key value",
    )

    def __init__(
        self,
        *,
        expected_language: str = "es",
        allow_sanitization: bool = True,
        max_internal_lines: int = 3,
    ) -> None:
        self.expected_language = expected_language.strip().lower() or "es"
        self.allow_sanitization = bool(allow_sanitization)
        self.max_internal_lines = max(0, int(max_internal_lines))

    def inspect(
        self,
        state: CognitiveState,
        candidate_answer: str,
        *,
        expected_language: Optional[str] = None,
        allow_raw_metadata: bool = False,
    ) -> OutputGuardDecision:
        if not isinstance(state, CognitiveState):
            raise TypeError("state debe ser CognitiveState.")

        if not isinstance(candidate_answer, str):
            raise TypeError("candidate_answer debe ser string.")

        expected = (
            expected_language.strip().lower()
            if isinstance(expected_language, str) and expected_language.strip()
            else self.expected_language
        )

        original = candidate_answer.strip()

        if not original:
            return self._record(
                state,
                OutputGuardDecision(
                    accepted=False,
                    action=self.ACTION_REGENERATE,
                    output=None,
                    issues=["empty_output"],
                    expected_language=expected,
                    detected_language="unknown",
                    sanitized=False,
                    reason="La respuesta está vacía.",
                ),
            )

        detected_language = self._detect_language(original)

        # 1) Deriva de idioma.
        if expected == "es" and detected_language == "pt":
            return self._record(
                state,
                OutputGuardDecision(
                    accepted=False,
                    action=self.ACTION_REGENERATE,
                    output=None,
                    issues=["language_drift:pt"],
                    expected_language=expected,
                    detected_language=detected_language,
                    sanitized=False,
                    reason=(
                        "La salida deriva claramente al portugués; "
                        "debe regenerarse en español."
                    ),
                ),
            )

        # 2) Token interno aislado: no hay suficiente contenido natural que salvar.
        normalized_whole = self._normalize(original)
        if normalized_whole in {
            self._normalize(token)
            for token in self.INTERNAL_EXACT_TOKENS
        }:
            return self._record(
                state,
                OutputGuardDecision(
                    accepted=False,
                    action=self.ACTION_REGENERATE,
                    output=None,
                    issues=["internal_token_leak"],
                    expected_language=expected,
                    detected_language=detected_language,
                    sanitized=False,
                    reason=(
                        "La respuesta contiene únicamente un identificador "
                        "interno y no una respuesta para el usuario."
                    ),
                ),
            )

        # 3) Salida cruda key=value.
        if (
            not allow_raw_metadata
            and not self._user_requested_raw(state.user_query)
            and self._looks_like_raw_metadata_dump(original)
        ):
            return self._record(
                state,
                OutputGuardDecision(
                    accepted=False,
                    action=self.ACTION_REGENERATE,
                    output=None,
                    issues=["raw_internal_metadata"],
                    expected_language=expected,
                    detected_language=detected_language,
                    sanitized=False,
                    reason=(
                        "La salida parece un volcado crudo de metadata/estado "
                        "interno y debe transformarse en lenguaje natural."
                    ),
                ),
            )

        # 4) Intentar limpiar líneas de control internas si existe texto natural.
        cleaned, removed_lines = self._sanitize_internal_lines(original)

        if removed_lines:
            if not self.allow_sanitization:
                return self._record(
                    state,
                    OutputGuardDecision(
                        accepted=False,
                        action=self.ACTION_REGENERATE,
                        output=None,
                        issues=["internal_control_lines"],
                        expected_language=expected,
                        detected_language=detected_language,
                        sanitized=False,
                        removed_lines=removed_lines,
                        reason=(
                            "Se detectaron líneas internas y la sanitización "
                            "está deshabilitada."
                        ),
                    ),
                )

            if len(removed_lines) > self.max_internal_lines:
                return self._record(
                    state,
                    OutputGuardDecision(
                        accepted=False,
                        action=self.ACTION_REGENERATE,
                        output=None,
                        issues=["excessive_internal_leakage"],
                        expected_language=expected,
                        detected_language=detected_language,
                        sanitized=False,
                        removed_lines=removed_lines,
                        reason=(
                            "Hay demasiadas líneas internas para considerar "
                            "segura una limpieza automática."
                        ),
                    ),
                )

            if not cleaned.strip():
                return self._record(
                    state,
                    OutputGuardDecision(
                        accepted=False,
                        action=self.ACTION_REGENERATE,
                        output=None,
                        issues=["output_empty_after_sanitization"],
                        expected_language=expected,
                        detected_language=detected_language,
                        sanitized=True,
                        removed_lines=removed_lines,
                        reason=(
                            "Tras eliminar líneas internas no queda una "
                            "respuesta útil."
                        ),
                    ),
                )

            decision = OutputGuardDecision(
                accepted=True,
                action=self.ACTION_ACCEPT_SANITIZED,
                output=cleaned.strip(),
                issues=["internal_control_lines_removed"],
                expected_language=expected,
                detected_language=detected_language,
                sanitized=True,
                removed_lines=removed_lines,
                reason=(
                    "Se eliminaron líneas de control internas y quedó una "
                    "respuesta natural válida."
                ),
            )
            state.draft_answer = decision.output
            return self._record(state, decision)

        # 5) Aceptación normal.
        decision = OutputGuardDecision(
            accepted=True,
            action=self.ACTION_ACCEPT,
            output=original,
            issues=[],
            expected_language=expected,
            detected_language=detected_language,
            sanitized=False,
            reason="La salida no presenta problemas visibles de formato o filtración.",
        )
        state.draft_answer = decision.output
        return self._record(state, decision)

    # ------------------------------------------------------------------
    # Detección / sanitización
    # ------------------------------------------------------------------

    def _sanitize_internal_lines(
        self,
        text: str,
    ) -> Tuple[str, List[str]]:
        kept: List[str] = []
        removed: List[str] = []

        for line in text.splitlines():
            stripped = line.strip()
            normalized = self._normalize(stripped)

            if not stripped:
                kept.append(line)
                continue

            if any(
                normalized.startswith(self._normalize(prefix))
                for prefix in self.INTERNAL_LINE_PREFIXES
            ):
                removed.append(stripped)
                continue

            if normalized in {
                self._normalize(token)
                for token in self.INTERNAL_EXACT_TOKENS
            }:
                removed.append(stripped)
                continue

            kept.append(line)

        cleaned = "\n".join(kept)

        # Evitar acumulación de demasiadas líneas vacías tras limpiar.
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        return cleaned, removed

    def _looks_like_raw_metadata_dump(self, text: str) -> bool:
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if len(lines) < 2:
            return False

        metadata_lines = 0

        for line in lines:
            match = re.match(
                r"^([A-Za-zÁÉÍÓÚáéíóúÑñ_ ]+)\s*=\s*(.+)$",
                line,
            )
            if not match:
                continue

            key = self._normalize(match.group(1)).replace("_", " ")
            if key in {
                self._normalize(k).replace("_", " ")
                for k in self.RAW_METADATA_KEYS
            }:
                metadata_lines += 1

        return metadata_lines >= 2 and metadata_lines >= max(2, len(lines) // 2)

    def _detect_language(self, text: str) -> str:
        normalized = self._normalize(text)
        tokens = set(re.findall(r"[a-záéíóúñãõç]+", normalized))

        pt_score = 0
        es_score = 0

        for marker in self.PORTUGUESE_MARKERS:
            marker_norm = self._normalize(marker)
            if " " in marker_norm:
                if marker_norm in normalized:
                    pt_score += 2
            elif marker_norm in tokens:
                pt_score += 1

        for marker in self.SPANISH_MARKERS:
            marker_norm = self._normalize(marker)
            if " " in marker_norm:
                if marker_norm in normalized:
                    es_score += 2
            elif marker_norm in tokens:
                es_score += 1

        # Conservador: solo marcamos portugués cuando hay evidencia clara.
        if pt_score >= 3 and pt_score >= es_score + 2:
            return "pt"

        if es_score >= 2:
            return "es"

        return "unknown"

    def _user_requested_raw(self, query: str) -> bool:
        normalized = self._normalize(query)
        return any(
            self._normalize(marker) in normalized
            for marker in self.RAW_REQUEST_MARKERS
        )

    # ------------------------------------------------------------------
    # Auditoría
    # ------------------------------------------------------------------

    def _record(
        self,
        state: CognitiveState,
        decision: OutputGuardDecision,
    ) -> OutputGuardDecision:
        state.turn_metadata.setdefault("output_guard", {})
        state.turn_metadata["output_guard"].update(
            {
                "version": self.VERSION,
                **decision.snapshot(),
            }
        )

        if not decision.accepted:
            state.draft_answer = None

        return decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        value = str(text).strip().lower()
        value = unicodedata.normalize("NFKD", value)
        value = "".join(
            ch for ch in value
            if not unicodedata.combining(ch)
        )
        value = re.sub(r"\s+", " ", value)
        return value
