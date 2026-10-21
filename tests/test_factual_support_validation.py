from __future__ import annotations

import json
import tempfile
from pathlib import Path

from quipu_loop import QuipuLoop
from verifier.factual_verifier import FactualVerifier
from verifier.support_validator import SupportValidator


EVIDENCE = [{
    "id": "bh-1",
    "text": (
        "El horizonte de sucesos no es la singularidad. "
        "Para Schwarzschild, el horizonte está en el radio de Schwarzschild, "
        "r_s = 2GM/c^2. La singularidad está en el centro, en r = 0. "
        "Por lo tanto, horizonte y singularidad son regiones distintas."
    ),
    "score": 0.61,
    "authority": 0.98,
    "meta": {"path": "science/black_holes_event_horizon.md"},
}]


def provider(_claim, top_k):
    return EVIDENCE[:top_k]


class FakeCfg:
    max_tokens = 512
    temperature = 0.7
    use_verifier = True
    use_rag = True
    use_planner = True
    use_cognitive_control = False


class CaptureAdapter:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        return json.dumps(self.payload, ensure_ascii=False)


def make_loop(adapter):
    loop = object.__new__(QuipuLoop)
    loop.cfg = FakeCfg()
    loop.adapter = adapter
    loop.session_path = Path(tempfile.mkdtemp()) / "session.jsonl"
    loop._factual_debug_enabled = False
    loop._factual_debug_phase = None
    loop._factual_debug_path = loop.session_path.with_name("session_factual_debug.jsonl")
    loop._last_factual_report = {}
    return loop


def test_1_support_validator_rejects_truncated_claim_fact():
    result = SupportValidator().validate(
        (
            "Se encuentra muy cerca del centro del agujero negro, "
            "donde el radio de Schwarzschild llega a la singularidad."
        ),
        [{"evidence_id": "bh-1", "text": EVIDENCE[0]["text"]}],
        {
            "evidence_ids": ["bh-1"],
            "claim_fact": "El horizonte está en el radio de Schwarzschild.",
            "evidence_fact": "El horizonte está en el radio de Schwarzschild.",
            "evidence_quote": (
                "el horizonte está en el radio de Schwarzschild, r_s = 2GM/c^2"
            ),
        },
    )
    assert result.valid is False
    assert result.missing_anchors
    assert result.fidelity_score < 0.50
    print("TEST 1 — claim_fact recortada => soporte rechazado")


def test_2_support_validator_accepts_faithful_supported_claim():
    result = SupportValidator().validate(
        "El horizonte de sucesos está en el radio de Schwarzschild.",
        [{"evidence_id": "bh-1", "text": EVIDENCE[0]["text"]}],
        {
            "evidence_ids": ["bh-1"],
            "claim_fact": "El horizonte de sucesos está en el radio de Schwarzschild.",
            "evidence_fact": "El horizonte está en el radio de Schwarzschild.",
            "evidence_quote": (
                "el horizonte está en el radio de Schwarzschild, r_s = 2GM/c^2"
            ),
        },
    )
    assert result.valid is True
    print("TEST 2 — claim fiel + cita real => SUPPORTED validable")


def test_3_atomic_split_separates_compound_scientific_claim():
    verifier = FactualVerifier(provider, None)
    claims = verifier.extract_claims(
        (
            "El horizonte se encuentra cerca del centro, a menudo ubicado "
            "en la región donde el radio de Schwarzschild llega a la singularidad."
        ),
        domain="astronomy",
    )
    assert len(claims) == 2
    assert "cerca del centro" in claims[0].text.lower()
    assert "singularidad" in claims[1].text.lower()
    assert claims[0].metadata["atomic_split"] is True
    print("TEST 3 — claim compuesta => 2 claims atómicas")


def test_4_false_supported_becomes_uncertain_and_requests_regeneration():
    def bad_supported(_claim, _evidence):
        return {
            "status": "supported",
            "confidence": 0.95,
            "reason": "Solo evalúo la parte correcta.",
            "evidence_ids": ["bh-1"],
            "claim_fact": "El horizonte está en el radio de Schwarzschild.",
            "evidence_fact": "El horizonte está en el radio de Schwarzschild.",
            "evidence_quote": (
                "el horizonte está en el radio de Schwarzschild, r_s = 2GM/c^2"
            ),
            "contradiction_type": None,
        }

    verifier = FactualVerifier(provider, bad_supported)
    report = verifier.verify(
        (
            "Se encuentra muy cerca del centro del agujero negro, "
            "donde el radio de Schwarzschild llega a la singularidad."
        ),
        user_text="Explícame Schwarzschild.",
        force=True,
    )
    assert report.supported_count == 0
    assert report.uncertain_count >= 1
    assert report.support_failure_count >= 1
    assert report.correction_required is True
    assert report.safe_to_answer is False
    print("TEST 4 — falso SUPPORTED => UNCERTAIN + regeneración")


def test_5_judge_payload_hides_retrieval_score_and_authority():
    adapter = CaptureAdapter({
        "status": "uncertain",
        "confidence": 0.50,
        "reason": "prueba",
        "evidence_ids": ["bh-1"],
        "claim_fact": None,
        "evidence_fact": None,
        "evidence_quote": None,
        "contradiction_type": None,
    })
    loop = make_loop(adapter)
    loop._factual_judge_adapter(
        "El horizonte está en el radio de Schwarzschild.",
        [{
            "evidence_id": "bh-1",
            "source": "NASA",
            "score": 0.613272,
            "authority": 0.98,
            "text": EVIDENCE[0]["text"],
        }],
    )
    payload = json.loads(adapter.calls[-1][-1]["content"])
    row = payload["evidence"][0]
    assert "score" not in row
    assert "authority" not in row
    assert row["evidence_id"] == "bh-1"
    print("TEST 5 — juez no ve retrieval score ni authority")


def test_6_real_schwarzschild_failure_is_caught():
    def real_bad_judge(claim, _evidence):
        return {
            "status": "supported",
            "confidence": 0.95,
            "reason": "Reduzco la claim.",
            "evidence_ids": ["bh-1"],
            "claim_fact": "El horizonte está en el radio de Schwarzschild.",
            "evidence_fact": "El horizonte está en el radio de Schwarzschild.",
            "evidence_quote": (
                "el horizonte está en el radio de Schwarzschild, r_s = 2GM/c^2"
            ),
        }

    verifier = FactualVerifier(provider, real_bad_judge)
    report = verifier.verify(
        (
            "El horizonte se encuentra muy cerca del centro del agujero negro, "
            "a menudo ubicado en la región donde el radio de Schwarzschild "
            "llega al punto de ruptura o singularidad del agujero negro."
        ),
        user_text="¿Dónde está el horizonte respecto a la singularidad?",
        force=True,
    )
    assert report.safe_to_answer is False
    assert report.correction_required is True
    assert report.support_failure_count >= 1
    print("TEST 6 — caso real Schwarzschild ya no queda safe_to_answer")


def main():
    test_1_support_validator_rejects_truncated_claim_fact()
    test_2_support_validator_accepts_faithful_supported_claim()
    test_3_atomic_split_separates_compound_scientific_claim()
    test_4_false_supported_becomes_uncertain_and_requests_regeneration()
    test_5_judge_payload_hides_retrieval_score_and_authority()
    test_6_real_schwarzschild_failure_is_caught()
    print("=" * 100)
    print("FACTUAL SUPPORT & CLAIM FIDELITY 1.0.6: 6/6 pruebas superadas.")


if __name__ == "__main__":
    main()
