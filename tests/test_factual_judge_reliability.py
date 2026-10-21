from __future__ import annotations

import json
import tempfile
from pathlib import Path

from quipu_loop import QuipuLoop
from verifier.factual_verifier import FactualVerifier

EVIDENCE = [{
    "id": "src_schwarzschild_1",
    "text": (
        "El horizonte de sucesos no es la singularidad. "
        "El horizonte está en r_s = 2GM/c^2. "
        "La singularidad clásica está en r = 0."
    ),
    "score": 0.92,
    "authority": 0.98,
    "meta": {"path": "knowledge/science/astronomy/black_holes_event_horizon.md"},
}]

class FakeCfg:
    max_tokens = 512
    temperature = 0.7
    use_verifier = True
    use_rag = True
    use_planner = True
    use_cognitive_control = False

class RetrySucceedsAdapter:
    def __init__(self):
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if len(self.calls) == 1:
            return (
                '{"status":"contradicted","confidence":0.98,'
                '"reason":"La evidencia separa horizonte y singularidad",'
                '"evidence_ids":["src_schwarzschild_1"],'
                '"claim_fact":"radio de Schwarzschild llega a singularidad",'
                '"evidence'
            )
        return json.dumps({
            "status": "contradicted",
            "confidence": 0.98,
            "reason": "Horizonte y singularidad son distintos.",
            "evidence_ids": ["src_schwarzschild_1"],
            "claim_fact": "El horizonte coincide con la singularidad.",
            "evidence_fact": "El horizonte no es la singularidad.",
            "evidence_quote": "El horizonte de sucesos no es la singularidad.",
            "contradiction_type": "identity_conflict",
        })

class AlwaysMalformedAdapter:
    def __init__(self):
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return '{"status":"contradicted","confidence":0.98,"reason":"truncado"'

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

def provider(_claim, top_k):
    return EVIDENCE[:top_k]

def test_1_truncated_json_retries():
    adapter = RetrySucceedsAdapter()
    loop = make_loop(adapter)
    parsed = json.loads(loop._factual_judge_adapter(
        "El horizonte coincide con la singularidad.", EVIDENCE
    ))
    assert parsed["status"] == "contradicted"
    assert len(adapter.calls) == 2
    assert adapter.calls[0]["kwargs"]["max_tokens"] >= 512
    assert adapter.calls[1]["kwargs"]["max_tokens"] == 384
    print("TEST 1 — JSON truncado => retry; primer intento >=512 tokens")

def test_2_two_invalid_outputs_become_judge_error():
    adapter = AlwaysMalformedAdapter()
    loop = make_loop(adapter)
    parsed = json.loads(loop._factual_judge_adapter(
        "El horizonte coincide con la singularidad.", EVIDENCE
    ))
    assert parsed["status"] == "judge_error"
    assert len(adapter.calls) == 2
    print("TEST 2 — Dos JSON inválidos => JUDGE_ERROR")

def test_3_malformed_generic_judge_is_not_unverifiable():
    def malformed(_claim, _evidence):
        return '{"status":"contradicted","confidence":0.98'
    verifier = FactualVerifier(provider, malformed)
    report = verifier.verify(
        "El horizonte coincide con la singularidad.",
        user_text="Explícame un agujero negro.",
        force=True,
    )
    assert report.judge_error_count == 1
    assert report.unverifiable_count == 0
    assert report.safe_to_answer is False
    assert report.verdicts[0].status == "judge_error"
    print("TEST 3 — JSON malformado => JUDGE_ERROR, no UNVERIFIABLE")

def test_4_retry_success_reaches_contradiction_validator():
    adapter = RetrySucceedsAdapter()
    loop = make_loop(adapter)
    verifier = FactualVerifier(provider, loop._factual_judge_adapter)
    report = verifier.verify(
        "El horizonte coincide con la singularidad.",
        user_text="Explícame un agujero negro.",
        force=True,
    )
    assert report.judge_error_count == 0
    assert report.contradicted_count == 1
    assert report.verdicts[0].contradiction_validation["valid"] is True
    print("TEST 4 — Retry válido llega a ContradictionValidator")

def test_5_report_marks_technical_failure():
    def judge_error(_claim, _evidence):
        return {
            "status": "judge_error",
            "confidence": 0.0,
            "reason": "JSON inválido en dos intentos.",
            "evidence_ids": ["src_schwarzschild_1"],
        }
    verifier = FactualVerifier(provider, judge_error)
    report = verifier.verify(
        "El horizonte coincide con la singularidad.",
        user_text="Explícame un agujero negro.",
        force=True,
    )
    assert report.safe_to_answer is False
    assert report.judge_error_count == 1
    assert report.metadata["technical_failure"] is True
    print("TEST 5 — JUDGE_ERROR marca technical_failure y fail-closed")

def main():
    test_1_truncated_json_retries()
    test_2_two_invalid_outputs_become_judge_error()
    test_3_malformed_generic_judge_is_not_unverifiable()
    test_4_retry_success_reaches_contradiction_validator()
    test_5_report_marks_technical_failure()
    print("=" * 100)
    print("FACTUAL JUDGE OUTPUT RELIABILITY 1.0.5: 5/5 pruebas superadas.")

if __name__ == "__main__":
    main()
