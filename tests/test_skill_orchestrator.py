from reasoner.orchestrator import SkillOrchestrator

def fake_provider(q, k):
    return [{"text":"foo", "score":0.8, "citation_id":"mem://x"}]

def test_choose_and_execute():
    orch = SkillOrchestrator(fake_provider)
    assert orch.choose_skill("calcula 2+2").skill == "python_exec"
    assert orch.choose_skill("leer 'README.md'").skill == "filesystem_read"
    out = orch.execute("Kant 1781", k=2)
    assert out["ok"] and out["skill"] in ("memory_search","python_exec","filesystem_read","search_web")
