from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
import json
import re
import unicodedata

from .contradiction_validator import ContradictionValidator
from .support_validator import SupportValidator

EvidenceProvider = Callable[[str, int], Sequence[Any]]
JudgeAdapter = Callable[[str, Sequence["EvidenceItem"]], Mapping[str, Any] | str]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bounded(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value, 0.0)))


def _normalize(text: str) -> str:
    value = str(text or "").strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", value)


@dataclass
class Claim:
    claim_id: str
    text: str
    source_sentence: Optional[str] = None
    domain: Optional[str] = None
    critical: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceItem:
    evidence_id: str
    text: str
    source: str = "unknown"
    score: float = 0.0
    authority: float = 0.5
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = _bounded(self.score)
        self.authority = _bounded(self.authority)

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimVerdict:
    claim_id: str
    claim: str
    status: str
    confidence: float
    reason: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    contradiction_validation: Dict[str, Any] = field(default_factory=dict)
    support_validation: Dict[str, Any] = field(default_factory=dict)
    requires_regeneration: bool = False

    def __post_init__(self) -> None:
        self.status = str(self.status or "").strip().lower()
        if self.status not in FactualVerifier.VALID_STATUSES:
            self.status = FactualVerifier.STATUS_UNVERIFIABLE
        self.confidence = _bounded(self.confidence)

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FactualVerificationReport:
    answer: str
    claims: List[Claim] = field(default_factory=list)
    verdicts: List[ClaimVerdict] = field(default_factory=list)
    safe_to_answer: bool = True
    correction_required: bool = False
    requires_caveat: bool = False
    supported_count: int = 0
    contradicted_count: int = 0
    uncertain_count: int = 0
    unverifiable_count: int = 0
    judge_error_count: int = 0
    support_failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "claims": [x.snapshot() for x in self.claims],
            "verdicts": [x.snapshot() for x in self.verdicts],
            "safe_to_answer": self.safe_to_answer,
            "correction_required": self.correction_required,
            "requires_caveat": self.requires_caveat,
            "supported_count": self.supported_count,
            "contradicted_count": self.contradicted_count,
            "uncertain_count": self.uncertain_count,
            "unverifiable_count": self.unverifiable_count,
            "judge_error_count": self.judge_error_count,
            "support_failure_count": self.support_failure_count,
            "metadata": dict(self.metadata),
        }


class FactualVerifier:
    """Quipu Factual Verification 1.1.0 — support & claim fidelity validation."""

    VERSION = "1.2.0"
    STATUS_SUPPORTED = "supported"
    STATUS_CONTRADICTED = "contradicted"
    STATUS_UNCERTAIN = "uncertain"
    STATUS_UNVERIFIABLE = "unverifiable"
    STATUS_JUDGE_ERROR = "judge_error"
    VALID_STATUSES = {
        STATUS_SUPPORTED, STATUS_CONTRADICTED,
        STATUS_UNCERTAIN, STATUS_UNVERIFIABLE, STATUS_JUDGE_ERROR,
    }

    SCIENCE_MARKERS = (
        "fisica", "física", "astronomia", "astronomía", "agujero negro",
        "horizonte de sucesos", "relatividad", "cuantica", "cuántica",
        "quimica", "química", "molecula", "molécula", "atomo", "átomo",
        "biologia", "biología", "genetica", "genética", "evolucion",
        "evolución", "geologia", "geología", "planeta", "estrella",
        "gravedad", "radiacion", "radiación", "celula", "célula",
        "especie", "proteina", "proteína", "energia", "energía",
        "fosil", "fósil", "fosiles", "fósiles", "paleontologia",
        "paleontología", "taxonomia", "taxonomía", "filogenia",
        "reino biológico", "eucariota", "procariota", "mamifero",
        "mamífero", "dinosaurio", "organismo", "extincion", "extinción",
    )

    def __init__(self, evidence_provider: Optional[EvidenceProvider] = None,
                 judge_adapter: Optional[JudgeAdapter] = None, *,
                 evidence_top_k: int = 5, min_evidence_score: float = 0.10,
                 min_judge_confidence: float = 0.60, max_claims: int = 12,
                 contradiction_validator: Optional[ContradictionValidator] = None,
                 support_validator: Optional[SupportValidator] = None) -> None:
        self.evidence_provider = evidence_provider
        self.judge_adapter = judge_adapter
        self.evidence_top_k = max(1, int(evidence_top_k))
        self.min_evidence_score = _bounded(min_evidence_score)
        self.min_judge_confidence = _bounded(min_judge_confidence)
        self.max_claims = max(1, int(max_claims))
        self.contradiction_validator = (
            contradiction_validator
            if contradiction_validator is not None
            else ContradictionValidator()
        )
        self.support_validator = (
            support_validator
            if support_validator is not None
            else SupportValidator()
        )

    @classmethod
    def should_verify(cls, user_text: str, answer: str) -> bool:
        text = _normalize(f"{user_text} {answer}")
        return any(_normalize(m) in text for m in cls.SCIENCE_MARKERS)

    @staticmethod
    def _looks_like_verifiable_atom(text: str) -> bool:
        norm = _normalize(text)
        if len(norm) < 8:
            return False
        return bool(re.search(
            r"\b(?:es|son|esta|estan|se|tiene|tienen|puede|pueden|ocurre|"
            r"existe|existen|equivale|orbita|contiene|consiste|representa|"
            r"pertenece|pertenecen|pertenecia|pertenecian|vivio|vivieron|"
            r"evoluciono|evolucionaron|era|eran|fue|fueron|"
            r"define|produce|causa|coincide|coinciden|encuentra|ubicad[oa]s?|"
            r"llega|alcanza|incluye|is|are|has|have|can|contains|orbits|"
            r"represents|reaches)\b",
            norm,
        ))

    @classmethod
    def _split_atomic_sentence(cls, sentence: str) -> List[str]:
        value = str(sentence or "").strip()
        if not value:
            return []
        parts = re.split(
            r"\s*;\s*|,\s*(?=(?:a menudo|ademas|además|pero|mientras|aunque|sin embargo)\b)",
            value,
            flags=re.IGNORECASE,
        )
        parts = [p.strip(" ,;") for p in parts if p.strip(" ,;")]
        out: List[str] = []
        for part in parts:
            candidates = re.split(r"\s+(?:y|e|pero)\s+", part, flags=re.IGNORECASE)
            if len(candidates) == 2 and all(
                cls._looks_like_verifiable_atom(item) for item in candidates
            ):
                out.extend(item.strip() for item in candidates)
            else:
                out.append(part)
        return out or [value]

    @staticmethod
    def _is_context_only_claim(value: str) -> bool:
        """Ignore bare agreement/disagreement about a *previous* proposition.

        It is not an independent factual claim. The propositions following it
        are still checked normally. Do not suppress negated scientific facts.
        """
        normalized = _normalize(value).strip(" \t\r\n.!¡¿?;:\"'`*_-")
        normalized = re.sub(r"\s+", " ", normalized)
        return bool(re.fullmatch(
            r"(?:(?:no|si|yes|correcto|incorrecto)\s*,?\s*)?"
            r"(?:(?:esa|esta|la)\s+)?afirmacion\s+"
            r"(?:es\s+)?(?:falsa|verdadera|correcta|incorrecta)"
            r"|(?:no|si)\s*,?\s*(?:eso|esa afirmacion)\s+"
            r"(?:no\s+)?(?:es\s+)?(?:cierto|verdad|correcto|falso)"
            r"|(?:no|si)\s*,?\s*(?:es\s+)?"
            r"(?:correcto|incorrecto|cierto|falso)",
            normalized,
        ))

    def extract_claims(self, answer: str, *, domain: Optional[str] = None) -> List[Claim]:
        text = str(answer or "").strip()
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
        claims: List[Claim] = []
        for sentence in sentences:
            source_sentence = sentence.strip(" \t\r\n-•")
            if not source_sentence:
                continue
            atoms = self._split_atomic_sentence(source_sentence)
            for atom_index, atom in enumerate(atoms, 1):
                value = atom.strip(" \t\r\n-•")
                if len(value) < 8 or value.startswith("¿") or value.endswith("?"):
                    continue
                if not re.search(r"[a-z]", _normalize(value)):
                    continue
                if self._is_context_only_claim(value):
                    continue
                if not self._looks_like_verifiable_atom(value):
                    continue
                claims.append(Claim(
                    f"claim_{len(claims)+1}",
                    value,
                    source_sentence,
                    domain,
                    True,
                    {
                        "atomic_split": len(atoms) > 1,
                        "atomic_index": atom_index,
                        "atomic_count": len(atoms),
                    },
                ))
                if len(claims) >= self.max_claims:
                    return claims
        return claims

    def retrieve_evidence(self, claim: Claim) -> List[EvidenceItem]:
        if self.evidence_provider is None:
            return []
        try:
            raw_items = self.evidence_provider(claim.text, self.evidence_top_k)
        except Exception:
            return []
        items: List[EvidenceItem] = []
        for i, item in enumerate(raw_items or [], 1):
            evidence = self._coerce_evidence(item, i)
            if evidence and evidence.score >= self.min_evidence_score:
                items.append(evidence)
        # Si un chunk del mismo documento contiene front matter con autoridad
        # curada, propágala a sus chunks hermanos. No se usa el retrieval score
        # como autoridad factual.
        source_authority: Dict[str, float] = {}
        for evidence in items:
            if evidence.source and evidence.authority > 0.5:
                source_authority[evidence.source] = max(
                    source_authority.get(evidence.source, 0.0),
                    evidence.authority,
                )

        for evidence in items:
            inherited = source_authority.get(evidence.source)
            if inherited is not None and evidence.authority <= 0.5:
                evidence.authority = inherited

        items.sort(key=lambda x: (x.authority, x.score), reverse=True)
        return items[:self.evidence_top_k]

    @staticmethod
    def _coerce_evidence(item: Any, index: int) -> Optional[EvidenceItem]:
        if isinstance(item, EvidenceItem):
            return item
        if not isinstance(item, Mapping):
            return None
        text = str(item.get("text") or item.get("content") or item.get("snippet") or "").strip()
        if not text:
            return None
        metadata = dict(
            item.get("metadata")
            or item.get("meta")
            or {}
        )

        source = str(
            item.get("source")
            or item.get("src")
            or item.get("title")
            or metadata.get("path")
            or metadata.get("source")
            or "unknown"
        ).strip()

        authority_raw = item.get("authority")
        if authority_raw is None:
            authority_raw = metadata.get("authority")

        # Science Knowledge Corpus: when a chunk contains the document
        # front matter, preserve its curated authority score.
        if authority_raw is None:
            match = re.search(
                r"(?im)^\s*curation_authority_score\s*:\s*"
                r"([01](?:\.\d+)?)\s*$",
                text[:1800],
            )
            if match:
                authority_raw = match.group(1)

        if authority_raw is None:
            authority_raw = 0.5

        return EvidenceItem(
            evidence_id=str(
                item.get("evidence_id")
                or item.get("id")
                or item.get("citation_id")
                or f"evidence_{index}"
            ),
            text=text,
            source=source or "unknown",
            score=_safe_float(
                item.get("score", item.get("similarity", 0.5)),
                0.5,
            ),
            # IMPORTANTE: relevancia de recuperación != autoridad factual.
            # Nunca se reutiliza score/similarity como authority.
            authority=_safe_float(authority_raw, 0.5),
            url=str(item.get("url")).strip() if item.get("url") else None,
            metadata=metadata,
        )

    def judge_claim(self, claim: Claim, evidence: Sequence[EvidenceItem]) -> ClaimVerdict:
        if not evidence:
            return ClaimVerdict(
                claim.claim_id, claim.text, self.STATUS_UNVERIFIABLE, 0.0,
                "No se recuperó evidencia suficiente."
            )

        if self.judge_adapter is None:
            return self._conservative_fallback(claim, evidence)

        try:
            raw = self.judge_adapter(claim.text, evidence)
        except Exception as exc:
            return self._judge_error_verdict(
                claim, evidence,
                f"El juez factual lanzó una excepción: {type(exc).__name__}."
            )

        parsed = self._parse_judge_output(raw)
        if not parsed:
            return self._judge_error_verdict(
                claim, evidence,
                "El juez factual devolvió JSON inválido o truncado."
            )

        status = str(parsed.get("status") or "").strip().lower()
        confidence = _bounded(parsed.get("confidence", 0.0))
        reason = str(parsed.get("reason") or "").strip()

        if status not in self.VALID_STATUSES:
            return self._judge_error_verdict(
                claim, evidence,
                f"El juez factual devolvió un status inválido: {status or '<vacío>'}."
            )

        if status == self.STATUS_JUDGE_ERROR:
            return self._judge_error_verdict(
                claim, evidence,
                reason or "El juez factual no produjo una salida válida."
            )

        if confidence < self.min_judge_confidence:
            status = self.STATUS_UNCERTAIN

        raw_ids = [
            str(x).strip()
            for x in (parsed.get("evidence_ids") or [])
            if str(x).strip()
        ]
        emap = {x.evidence_id: x for x in evidence}
        valid_ids = [x for x in raw_ids if x in emap]
        contradiction_validation: Dict[str, Any] = {}
        support_validation: Dict[str, Any] = {}
        requires_regeneration = False

        if status == self.STATUS_SUPPORTED:
            cited = [emap[eid] for eid in valid_ids if eid in emap]
            if not cited:
                status = self.STATUS_UNCERTAIN
                confidence = min(confidence, 0.55)
                requires_regeneration = True
                reason = (
                    f"{reason} SUPPORTED degradado a UNCERTAIN: "
                    "faltó evidencia explícita válida."
                ).strip()
            else:
                support = self.support_validator.validate(
                    claim.text, cited, parsed
                )
                support_validation = support.snapshot()
                if not support.valid:
                    status = self.STATUS_UNCERTAIN
                    confidence = min(confidence, 0.55)
                    requires_regeneration = True
                    reason = (
                        f"{reason} SUPPORTED degradado a UNCERTAIN por "
                        f"SupportValidator: {support.reason}"
                    ).strip()
                else:
                    # Probe the literal quote for an explicit conflict with
                    # the ORIGINAL claim, not the model's shortened claim_fact.
                    probe = {
                        "evidence_ids": valid_ids,
                        "claim_fact": claim.text,
                        "evidence_fact": "",
                        "evidence_quote": parsed.get("evidence_quote"),
                        "contradiction_type": "other",
                    }
                    conflict = self.contradiction_validator.validate(
                        claim.text, cited, probe
                    )
                    if conflict.valid:
                        status = self.STATUS_CONTRADICTED
                        contradiction_validation = conflict.snapshot()
                        requires_regeneration = True
                        reason = (
                            f"{reason} SUPPORTED rechazado por contradicción "
                            f"explícita: {conflict.reason}"
                        ).strip()
                    else:
                        confidence = min(
                            confidence,
                            max(0.60, support.confidence),
                        )

        if status == self.STATUS_CONTRADICTED:
            if not reason or not valid_ids:
                status = self.STATUS_UNCERTAIN
                confidence = min(confidence, 0.55)
                reason = (
                    f"{reason} CONTRADICTED degradado a UNCERTAIN: "
                    "faltó evidencia explícita válida."
                ).strip()
            else:
                cited = [emap[x] for x in valid_ids if x in emap]
                validation = self.contradiction_validator.validate(
                    claim.text, cited, parsed
                )
                contradiction_validation = validation.snapshot()

                if not validation.valid:
                    status = self.STATUS_UNCERTAIN
                    confidence = min(confidence, 0.55)
                    reason = (
                        f"{reason} CONTRADICTED degradado a UNCERTAIN por "
                        f"ContradictionValidator: {validation.reason}"
                    ).strip()
                else:
                    requires_regeneration = True
                    confidence = min(
                        confidence,
                        max(0.60, validation.confidence),
                    )
                    reason = (
                        f"{reason} Contradicción validada determinísticamente: "
                        f"{validation.reason}"
                    ).strip()

        ids = (
            valid_ids
            if status == self.STATUS_CONTRADICTED
            else valid_ids or [x.evidence_id for x in evidence]
        )
        used = [emap[x].snapshot() for x in ids if x in emap]
        if not used and status != self.STATUS_CONTRADICTED:
            used = [x.snapshot() for x in evidence]

        return ClaimVerdict(
            claim.claim_id, claim.text, status, confidence, reason,
            [x["evidence_id"] for x in used], used,
            contradiction_validation=contradiction_validation,
            support_validation=support_validation,
            requires_regeneration=requires_regeneration,
        )

    def _judge_error_verdict(
        self,
        claim: Claim,
        evidence: Sequence[EvidenceItem],
        reason: str,
    ) -> ClaimVerdict:
        return ClaimVerdict(
            claim.claim_id,
            claim.text,
            self.STATUS_JUDGE_ERROR,
            0.0,
            str(reason or "Fallo técnico del juez factual."),
            [x.evidence_id for x in evidence],
            [x.snapshot() for x in evidence],
        )

    @staticmethod
    def _parse_judge_output(raw: Mapping[str, Any] | str) -> Dict[str, Any]:
        if isinstance(raw, Mapping):
            return dict(raw)
        value = re.sub(r"```(?:json)?\s*|```", "", str(raw or "").strip(), flags=re.I).strip()
        try:
            obj = json.loads(value)
            return dict(obj) if isinstance(obj, dict) else {}
        except Exception:
            start, end = value.find("{"), value.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(value[start:end+1])
                    return dict(obj) if isinstance(obj, dict) else {}
                except Exception:
                    pass
        return {}

    def _conservative_fallback(self, claim: Claim, evidence: Sequence[EvidenceItem]) -> ClaimVerdict:
        cn = _normalize(claim.text)
        for item in evidence:
            en = _normalize(item.text)
            if cn and cn in en:
                conf = min(0.95, 0.60 + 0.20*item.score + 0.15*item.authority)
                return ClaimVerdict(claim.claim_id, claim.text, self.STATUS_SUPPORTED, conf,
                                    "La afirmación aparece explícitamente respaldada por la evidencia.",
                                    [item.evidence_id], [item.snapshot()])
        return ClaimVerdict(claim.claim_id, claim.text, self.STATUS_UNVERIFIABLE, 0.25,
                            "Hay evidencia relacionada, pero no basta para decidir de forma determinista.",
                            [x.evidence_id for x in evidence], [x.snapshot() for x in evidence])

    def verify(self, answer: str, *, user_text: str = "", domain: Optional[str] = None,
               force: bool = False) -> FactualVerificationReport:
        answer_text = str(answer or "").strip()
        report = FactualVerificationReport(answer=answer_text,
                                            metadata={"version": self.VERSION, "domain": domain, "forced": bool(force)})
        if not force and not self.should_verify(user_text, answer_text):
            report.metadata.update({"skipped": True, "skip_reason": "no_science_signal"})
            return report
        report.claims = self.extract_claims(answer_text, domain=domain)
        if not report.claims:
            report.metadata.update({"skipped": True, "skip_reason": "no_verifiable_claims"})
            return report
        for claim in report.claims:
            verdict = self.judge_claim(claim, self.retrieve_evidence(claim))
            report.verdicts.append(verdict)
            if verdict.status == self.STATUS_SUPPORTED:
                report.supported_count += 1
            elif verdict.status == self.STATUS_CONTRADICTED:
                report.contradicted_count += 1
            elif verdict.status == self.STATUS_UNCERTAIN:
                report.uncertain_count += 1
            elif verdict.status == self.STATUS_JUDGE_ERROR:
                report.judge_error_count += 1
            else:
                report.unverifiable_count += 1

            if (
                bool(getattr(verdict, "requires_regeneration", False))
                and verdict.status != self.STATUS_CONTRADICTED
            ):
                report.support_failure_count += 1

        report.correction_required = (
            report.contradicted_count > 0
            or report.support_failure_count > 0
        )
        report.safe_to_answer = (
            not report.correction_required
            and report.judge_error_count == 0
        )
        report.requires_caveat = (
            report.uncertain_count > 0
            or report.unverifiable_count > 0
        )
        report.metadata["checked_claims"] = len(report.claims)
        report.metadata["status_counts"] = {
            "supported": report.supported_count,
            "contradicted": report.contradicted_count,
            "uncertain": report.uncertain_count,
            "unverifiable": report.unverifiable_count,
            "judge_error": report.judge_error_count,
            "support_failure": report.support_failure_count,
        }
        report.metadata["technical_failure"] = bool(report.judge_error_count)
        report.metadata["claim_fidelity_failure"] = bool(
            report.support_failure_count
        )
        return report
