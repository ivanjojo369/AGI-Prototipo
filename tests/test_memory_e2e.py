from memory.memory import write, search, prune, reindex

def test_memory_write_search_prune_reindex(tmp_path, monkeypatch):
    # no tocamos rutas globales: solo validamos contrato
    r1 = write("nota e2e memoria", user="tester", project_id="proj", tags=["e2e"])
    assert r1["ok"] and r1["id"]
    hits = search("nota e2e", topk=3, project_id="proj")
    assert hits and hits[0]["score"] >= 0.1
    pr = prune()
    assert pr["ok"]
    r2 = reindex(project_id="proj")
    assert r2["ok"] and r2["reindexed"] >= 1
