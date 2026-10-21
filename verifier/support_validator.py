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
    return re.sub(r"\s+", " ", value)


STOPWORDS = {
    "el","la","los","las","un","una","unos","unas","de","del","al","a",
    "en","por","para","con","sin","y","e","o","u","que","se","su","sus",
    "es","son","esta","estan","ser","como","desde","hasta","muy","mas",
    "menos","lo","le","les","donde","cuando","the","an","of","to","in",
    "is","are","and","or"
}


def _content_tokens(text: Any) -> list[str]:
    return [
        t for t in re.findall(r"[a-z0-9_]+", _normalize(text))
        if len(t) > 2 and t not in STOPWORDS
    ]


def _stems(text: Any) -> set[str]:
    out = set()
    for token in _content_tokens(text):
        out.add(token if len(token) <= 5 else token[:5])
    return out


def _coverage(source: Any, target: Any) -> float:
    src, tgt = _stems(source), _stems(target)
    if not src:
        return 1.0
    return len(src & tgt) / len(src)


@dataclass
class SupportValidationResult:
    valid: bool
    confidence: float
    reason: str
    evidence_id: Optional[str] = None
    evidence_quote: Optional[str] = None
    fidelity_score: float = 0.0
    support_score: float = 0.0
    missing_anchors: list[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))
        self.fidelity_score = max(0.0, min(1.0, float(self.fidelity_score or 0.0)))
        self.support_score = max(0.0, min(1.0, float(self.support_score or 0.0)))
        if self.missing_anchors is None:
            self.missing_anchors = []
        if self.metadata is None:
            self.metadata = {}

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


class SupportValidator:
    VERSION = "1.1.1"

    RELATION_ANCHOR_PATTERNS = {
        "negation": r"\b(?:no|nunca|jamas|not|never)\b",
        "near": r"\b(?:cerca|cercano|cercana|near|close)\b",
        "far": r"\b(?:lejos|lejano|lejana|far)\b",
        "center": r"\b(?:centro|central|center|centre)\b",
        "inside": r"\b(?:dentro|interior|inside|within)\b",
        "outside": r"\b(?:fuera|exterior|outside)\b",
        "same": r"\b(?:coincid\w*|igual\w*|mism[oa]s?|same|equival\w*)\b",
        "different": r"\b(?:distint\w*|diferent\w*|separad\w*|different|separate)\b",
        "reach": r"\b(?:llega\w*|alcanza\w*|reach\w*)\b",
        "cause": r"\b(?:causa\w*|produce\w*|provoca\w*|cause\w*)\b",
        "increase": r"\b(?:aument\w*|increment\w*|increase\w*)\b",
        "decrease": r"\b(?:disminu\w*|reduce\w*|decrease\w*)\b",
        "greater": r"\b(?:mayor|superior|greater|larger)\b",
        "less": r"\b(?:menor|inferior|less|smaller)\b",
        "before": r"\b(?:antes|before|previo\w*)\b",
        "after": r"\b(?:despues|after|posterior\w*)\b",
        "all": r"\b(?:todos?|todas?|all|every)\b",
        "none": r"\b(?:ningun\w*|none)\b",
    }

    def validate(
        self,
        claim: str,
        evidence: Sequence[Any],
        judge_payload: Optional[Mapping[str, Any]] = None,
    ) -> SupportValidationResult:
        payload = dict(judge_payload or {})
        original = str(claim or "").strip()
        claim_fact = str(payload.get("claim_fact") or original).strip()
        evidence_fact = str(payload.get("evidence_fact") or "").strip()
        quote = str(payload.get("evidence_quote") or "").strip()

        evidence_map: Dict[str, Dict[str, Any]] = {}
        for index, item in enumerate(evidence or [], 1):
            row = self._coerce_evidence(item, index)
            if row:
                evidence_map[row["evidence_id"]] = row

        requested_ids = [
            str(v).strip()
            for v in (payload.get("evidence_ids") or [])
            if str(v).strip()
        ]
        valid_ids = [eid for eid in requested_ids if eid in evidence_map]
        if not valid_ids:
            return SupportValidationResult(
                False, 0.0, "SUPPORTED sin evidence_ids válidos.",
                metadata={"validator_version": self.VERSION}
            )

        fidelity = _coverage(original, claim_fact)
        missing_anchors = sorted(self._anchors(original) - self._anchors(claim_fact))
        min_fidelity = 0.42 if len(_content_tokens(original)) >= 8 else 0.50

        if fidelity < min_fidelity or missing_anchors:
            return SupportValidationResult(
                False, min(0.55, fidelity),
                "claim_fact no conserva toda la afirmación original.",
                evidence_id=valid_ids[0],
                evidence_quote=quote or None,
                fidelity_score=fidelity,
                missing_anchors=missing_anchors,
                metadata={"validator_version": self.VERSION, "min_fidelity": min_fidelity},
            )

        rows = [evidence_map[eid] for eid in valid_ids]
        if quote:
            qn = _normalize(quote)
            matched = next((row for row in rows if qn and qn in _normalize(row["text"])), None)
            if matched is None:
                return SupportValidationResult(
                    False, 0.0,
                    "evidence_quote no aparece literalmente en la evidencia citada.",
                    evidence_quote=quote,
                    fidelity_score=fidelity,
                    metadata={"validator_version": self.VERSION},
                )
            support_text = quote
            evidence_id = matched["evidence_id"]
        else:
            support_text = " ".join(row["text"] for row in rows)
            evidence_id = rows[0]["evidence_id"]

        if evidence_fact:
            quote_alignment = _coverage(evidence_fact, support_text)
            if quote_alignment < 0.22:
                return SupportValidationResult(
                    False, 0.35,
                    "evidence_fact no representa fielmente la evidencia citada.",
                    evidence_id=evidence_id,
                    evidence_quote=quote or None,
                    fidelity_score=fidelity,
                    support_score=quote_alignment,
                    metadata={"validator_version": self.VERSION},
                )
        # Defense-in-depth: lexical overlap is NOT factual entailment.
        # Explicit numeric mismatch and opposite polarity must not be marked
        # SUPPORTED merely because their surrounding words are similar.
        quoted_numbers = self._numbers(support_text)
        asserted_numbers = self._numbers(original)
        if asserted_numbers and not asserted_numbers.issubset(quoted_numbers):
            return SupportValidationResult(
                False, 0.0,
                "La evidencia citada no contiene los valores numéricos afirmados.",
                evidence_id=evidence_id, evidence_quote=quote or None,
                fidelity_score=fidelity,
                missing_anchors=sorted(f"num:{n}" for n in asserted_numbers - quoted_numbers),
                metadata={"validator_version": self.VERSION},
            )

        # Guard against negation reversal of the SAME proposition.
        # An unrelated negative clause in a full source chunk must not turn
        # a positive, correctly supported sentence into a contradiction.
        # This check is deliberately conservative, not semantic entailment.
        def polarity(value: str) -> bool:
            return bool(re.search(r"\b(?:no|nunca|jamas|not|never)\b", _normalize(value)))

        # Judge-supplied literal quote takes precedence. Without a quote,
        # compare the claim with the most relevant sentence/clause instead
        # of the entire document (which can include unrelated negations).
        clauses = [
            part.strip() for part in re.split(r"(?<=[.!?;])\s+", support_text)
            if part.strip()
        ]
        if not clauses:
            clauses = [support_text]
        claim_numbers = self._numbers(original)
        ranked_clauses = sorted(
            clauses,
            key=lambda part: (
                len(claim_numbers & self._numbers(part)),
                _coverage(original, part),
            ),
            reverse=True,
        )
        selected = ranked_clauses[0]
        if (polarity(original) != polarity(selected)
                and _coverage(original, selected) >= 0.52):
            return SupportValidationResult(
                False, 0.0, "La evidencia tiene polaridad de negación diferente.",
                evidence_id=evidence_id, evidence_quote=quote or None,
                fidelity_score=fidelity,
                metadata={"validator_version": self.VERSION},
            )

        # Evidence facts produced by the judge may help explain citations,
        # but they must NOT boost the support score above the cited text.
        support_score = _coverage(claim_fact, support_text)

        if support_score < 0.28:
            return SupportValidationResult(
                False, 0.40,
                "La evidencia citada no cubre suficientemente el hecho juzgado.",
                evidence_id=evidence_id,
                evidence_quote=quote or None,
                fidelity_score=fidelity,
                support_score=support_score,
                metadata={"validator_version": self.VERSION},
            )

        return SupportValidationResult(
            True,
            min(0.99, 0.55 + 0.25*fidelity + 0.20*support_score),
            "La claim juzgada conserva la original y tiene soporte textual suficiente.",
            evidence_id=evidence_id,
            evidence_quote=quote or None,
            fidelity_score=fidelity,
            support_score=support_score,
            metadata={"validator_version": self.VERSION},
        )

    @staticmethod
    def _numbers(text: Any) -> set[str]:
        """Conservative literal numeric anchors; no inferred unit conversions."""
        normalized = _normalize(text)
        numbers = re.findall(r"(?<![\w.])\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?%?", normalized)
        return {value.replace(",", ".") for value in numbers}

    @classmethod
    def _anchors(cls, text: str) -> set[str]:
        value = _normalize(text)
        anchors = {
            name for name, pattern in cls.RELATION_ANCHOR_PATTERNS.items()
            if re.search(pattern, value)
        }
        for number in re.findall(r"\b\d+(?:[.,]\d+)?%?\b", value):
            anchors.add(f"num:{number.replace(',', '.')}")
        for term in (
            "singularidad","schwarzschild","horizonte","centro","radio",
            "temperatura","presion","masa","energia","velocidad","densidad",
        ):
            nt = _normalize(term)
            if nt in value:
                anchors.add(f"term:{nt}")
        return anchors

    @staticmethod
    def _coerce_evidence(item: Any, index: int) -> Optional[Dict[str, Any]]:
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
            }
        text = str(row.get("text") or "").strip()
        if not text:
            return None
        return {
            "evidence_id": str(
                row.get("evidence_id") or row.get("id") or f"evidence_{index}"
            ).strip(),
            "text": text,
        }
