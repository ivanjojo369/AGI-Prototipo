from root.verifier import verify, attempt_self_correction

def test_verifier_autocorrect_short_text():
    bad = "ok"
    ctx = {"rag_hits": [], "min_score": 0.45}
    v = verify(bad, ctx)
    assert not v["passed"]
    ac = attempt_self_correction(bad, ctx)
    assert isinstance(ac["output"], str) and len(ac["output"]) >= len(bad)
