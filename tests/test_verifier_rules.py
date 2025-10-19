# tests/test_verifier_rules.py
from verifier.verifier import verify_and_autocorrect

def test_verifier_marks_too_short_and_autocorrects():
    res = {"output": "corto"}
    out = verify_and_autocorrect(res)
    assert out["autocorrected"] is True
    assert out["ok"] is False
    assert "too_short" in out.get("reasons", [])
