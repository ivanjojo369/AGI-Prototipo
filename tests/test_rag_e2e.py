import subprocess, sys, json
from rag.retriever import search

def test_rag_index_and_search(tmp_path):
    # indexa un mini archivo
    (tmp_path / "a.md").write_text("La memoria episódica guarda eventos personales.", encoding="utf-8")
    cmd = [sys.executable, "-m", "scripts.index_folder", "--path", str(tmp_path), "--ext", ".md", "--mode", "fresh"]
    assert subprocess.run(cmd).returncode == 0

    hits = search("memoria episodica", top_k=3, min_score=0.10)
    assert len(hits) >= 1
    assert all("score" in h for h in hits)
