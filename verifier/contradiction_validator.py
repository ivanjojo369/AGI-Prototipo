from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional, Sequence
import re
import unicodedata


def _normalize(text: Any) -> str:
    value = str(text or "").strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("**", "").replace("`", "")
    value = re.sub(r"\s+", " ", value)
    return value


def _tokens(text: Any) -> set[str]:
    stop = {
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        "de", "del", "al", "a", "en", "por", "para", "con", "sin",
        "y", "o", "que", "se", "su", "sus", "es", "son", "esta",
        "estan", "ser", "como", "desde", "hasta", "muy", "mas",
        "the", "an", "of", "to", "in", "is", "are", "and",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", _normalize(text))
        if len(token) > 2 and token not in stop
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


@dataclass
class ContradictionValidationResult:
    valid: bool
    confidence: float
    reason: str
    contradiction_type: str = "unproven"
    evidence_id: Optional[str] = None
    evidence_quote: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))
        if self.metadata is None:
            self.metadata = {}

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


class ContradictionValidator:
    """
    Validador determinista y conservador para veredictos CONTRADICTED.

    Un evidence_id válido sólo prueba procedencia, no contradicción.
    Si no puede demostrarse una incompatibilidad explícita, el juicio debe
    degradarse a UNCERTAIN en FactualVerifier.
    """

    VERSION = "1.0.0"

    NEGATION_WORDS = {
        "no", "nunca", "jamas", "jamás", "sin", "not", "never",
    }

    SAME_PATTERNS = (
        r"\bcoincide(?:n)?\s+con\b",
        r"\bequivale(?:n)?\s+a\b",
        r"\bes\s+(?:exactamente\s+)?(?:la|el|lo)\s+mism[oa]\b",
        r"\bson\s+(?:exactamente\s+)?(?:la|el|lo)\s+mism[oa]s?\b",
        r"\bmisma\s+ubicacion\b",
        r"\bmismo\s+lugar\b",
        r"\bes\s+la\s+singularidad\b",
        r"\bes\s+el\s+horizonte\b",
    )

    DISTINCT_PATTERNS = (
        r"\bno\s+coincide(?:n)?\s+con\b",
        r"\bno\s+es\s+la\s+singularidad\b",
        r"\bno\s+es\s+el\s+horizonte\b",
        r"\bno\s+son\s+(?:la|el|lo)\s+mism[oa]s?\b",
        r"\bno\s+son\s+la\s+misma\s+ubicacion\b",
        r"\bregiones?\s+distint[oa]s?\b",
        r"\bubicaciones?\s+distint[oa]s?\b",
        r"\bson\s+distint[oa]s?\b",
        r"\bson\s+diferentes?\b",
        r"\bes\s+distint[oa]\b",
        r"\bes\s+diferente\b",
        r"\bregion\s+separad[oa]\b",
        r"\bregiones\s+separad[oa]s\b",
    )

    def validate(
        self,
        claim: str,
        evidence: Sequence[Any],
        judge_payload: Optional[Mapping[str, Any]] = None,
    ) -> ContradictionValidationResult:
        payload = dict(judge_payload or {})
        claim_text = str(claim or "").strip()

        evidence_map: Dict[str, Dict[str, Any]] = {}
        for index, item in enumerate(evidence or [], 1):
            row = self._coerce_evidence(item, index)
            if row:
                evidence_map[row["evidence_id"]] = row

        requested_ids = [
            str(value).strip()
            for value in (payload.get("evidence_ids") or [])
            if str(value).strip()
        ]
        valid_ids = [
            evidence_id
            for evidence_id in requested_ids
            if evidence_id in evidence_map
        ]

        if not valid_ids:
            return ContradictionValidationResult(
                False,
                0.0,
                "No hay evidence_ids válidos que demuestren la contradicción.",
                metadata={"validator_version": self.VERSION},
            )

        quote = str(payload.get("evidence_quote") or "").strip()
        quote_norm = _normalize(quote)

        if quote:
            matched_quote_id = None
            for evidence_id in valid_ids:
                text_norm = _normalize(evidence_map[evidence_id]["text"])
                if quote_norm and quote_norm in text_norm:
                    matched_quote_id = evidence_id
                    break

            if matched_quote_id is None:
                return ContradictionValidationResult(
                    False,
                    0.0,
                    "evidence_quote no aparece en los fragmentos citados.",
                    contradiction_type="quote_not_found",
                    evidence_quote=quote,
                    metadata={"validator_version": self.VERSION},
                )

            candidate_rows = [{
                **evidence_map[matched_quote_id],
                "validation_text": quote,
            }]
        else:
            candidate_rows = [
                {
                    **evidence_map[evidence_id],
                    "validation_text": evidence_map[evidence_id]["text"],
                }
                for evidence_id in valid_ids
            ]

        claim_fact = str(payload.get("claim_fact") or claim_text).strip()
        evidence_fact = str(payload.get("evidence_fact") or "").strip()
        declared_type = str(
            payload.get("contradiction_type") or ""
        ).strip().lower()

        for row in candidate_rows:
            ev_text = evidence_fact or row["validation_text"]
            identity = self._identity_conflict(claim_fact, ev_text)

            if identity is not None:
                valid, reason = identity

                if valid:
                    return ContradictionValidationResult(
                        True,
                        0.99,
                        reason,
                        contradiction_type="identity_conflict",
                        evidence_id=row["evidence_id"],
                        evidence_quote=quote or None,
                        metadata={
                            "validator_version": self.VERSION,
                            "declared_type": declared_type or None,
                        },
                    )

                return ContradictionValidationResult(
                    False,
                    0.95,
                    reason,
                    contradiction_type="compatible_identity_relation",
                    evidence_id=row["evidence_id"],
                    evidence_quote=quote or None,
                    metadata={
                        "validator_version": self.VERSION,
                        "declared_type": declared_type or None,
                    },
                )

        for row in candidate_rows:
            ev_text = evidence_fact or row["validation_text"]
            negation_reason = self._explicit_negation_conflict(
                claim_fact,
                ev_text,
            )

            if negation_reason:
                return ContradictionValidationResult(
                    True,
                    0.94,
                    negation_reason,
                    contradiction_type="explicit_negation",
                    evidence_id=row["evidence_id"],
                    evidence_quote=quote or None,
                    metadata={
                        "validator_version": self.VERSION,
                        "declared_type": declared_type or None,
                    },
                )

        if quote:
            symbolic_reason = self._symbolic_conflict(
                claim_fact,
                quote,
            )
            if symbolic_reason:
                return ContradictionValidationResult(
                    True,
                    0.96,
                    symbolic_reason,
                    contradiction_type="symbolic_conflict",
                    evidence_id=candidate_rows[0]["evidence_id"],
                    evidence_quote=quote,
                    metadata={
                        "validator_version": self.VERSION,
                        "declared_type": declared_type or None,
                    },
                )

        return ContradictionValidationResult(
            False,
            0.35,
            (
                "La evidencia citada puede ser relevante, pero no se pudo "
                "demostrar determinísticamente una incompatibilidad explícita."
            ),
            contradiction_type="unproven",
            evidence_id=candidate_rows[0]["evidence_id"] if candidate_rows else None,
            evidence_quote=quote or None,
            metadata={
                "validator_version": self.VERSION,
                "declared_type": declared_type or None,
            },
        )

    @staticmethod
    def _coerce_evidence(
        item: Any,
        index: int,
    ) -> Optional[Dict[str, Any]]:
        if isinstance(item, Mapping):
            row = dict(item)
        elif hasattr(item, "snapshot"):
            try:
                row = dict(item.snapshot())
            except Exception:
                return None
        else:
            row = {
                "evidence_id": getattr(item, "evidence_id", None),
                "text": getattr(item, "text", None),
                "source": getattr(item, "source", None),
            }

        text = str(row.get("text") or "").strip()
        if not text:
            return None

        evidence_id = str(
            row.get("evidence_id")
            or row.get("id")
            or f"evidence_{index}"
        ).strip()

        return {
            "evidence_id": evidence_id,
            "text": text,
            "source": str(row.get("source") or "unknown"),
        }

    @classmethod
    def _identity_polarity(
        cls,
        text: str,
    ) -> Optional[str]:
        value = _normalize(text)

        distinct = any(
            re.search(pattern, value)
            for pattern in cls.DISTINCT_PATTERNS
        )
        same = any(
            re.search(pattern, value)
            for pattern in cls.SAME_PATTERNS
        )

        if distinct:
            return "distinct"
        if same:
            return "same"
        return None

    @classmethod
    def _identity_conflict(
        cls,
        claim: str,
        evidence: str,
    ) -> Optional[tuple[bool, str]]:
        claim_pol = cls._identity_polarity(claim)
        evidence_pol = cls._identity_polarity(evidence)

        if claim_pol is None or evidence_pol is None:
            return None

        if claim_pol != evidence_pol:
            return (
                True,
                (
                    "La claim y la evidencia expresan relaciones de identidad "
                    "opuestas: misma entidad/ubicación frente a entidades o "
                    "regiones distintas."
                ),
            )

        return (
            False,
            (
                "La claim y la evidencia expresan la misma polaridad de "
                "identidad/separación; esa evidencia no demuestra contradicción."
            ),
        )

    @classmethod
    def _explicit_negation_conflict(
        cls,
        claim: str,
        evidence: str,
    ) -> Optional[str]:
        cn = _normalize(claim)
        en = _normalize(evidence)

        c_tokens = _tokens(cn)
        e_tokens = _tokens(en)

        if _jaccard(c_tokens, e_tokens) < 0.58:
            return None

        c_neg = any(
            re.search(rf"\b{re.escape(word)}\b", cn)
            for word in cls.NEGATION_WORDS
        )
        e_neg = any(
            re.search(rf"\b{re.escape(word)}\b", en)
            for word in cls.NEGATION_WORDS
        )

        if c_neg == e_neg:
            return None

        return (
            "Claim y evidencia tienen contenido léxico altamente alineado "
            "pero polaridad de negación opuesta."
        )

    @staticmethod
    def _symbolic_conflict(
        claim: str,
        evidence_quote: str,
    ) -> Optional[str]:
        def equations(text: str) -> Dict[str, str]:
            out: Dict[str, str] = {}
            for var, value in re.findall(
                r"\b([a-zA-Z][a-zA-Z0-9_]*)\s*=\s*"
                r"([a-zA-Z0-9_./^*+\-]+)",
                str(text or ""),
            ):
                out[var.lower()] = value.lower()
            return out

        claim_eq = equations(claim)
        evidence_eq = equations(evidence_quote)

        for variable in set(claim_eq) & set(evidence_eq):
            if claim_eq[variable] != evidence_eq[variable]:
                return (
                    f"La misma variable {variable} tiene valores incompatibles "
                    "en la claim y en la cita de evidencia."
                )

        return None
