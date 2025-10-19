from root.quipu_loop import QuipuLoop

def test_loop_runs_and_verifies():
    loop = QuipuLoop(project_id="proj", min_score=0.10, top_k=3)
    out = loop.run("define episodios de memoria")
    assert out["ok"]
    assert "output" in out and isinstance(out["output"], str)
    assert "stats" in out and out["stats"]["rag_hits"] >= 0
    assert "verified" in out and isinstance(out["verified"]["ok"], bool)
