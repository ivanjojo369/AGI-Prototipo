# tests/test_tools_rag_ok.py
from tools.registry import ToolRegistry
from rag.retriever import add_document

def test_qa_rag_returns_text_and_citations(tmp_path, monkeypatch):
    add_document("Prueba de documento de RAG para Kant.", {"src":"test-doc"})
    reg = ToolRegistry()
    tool = reg.get("qa_rag")
    out = tool.run(query="Kant", project_id="default")
    assert isinstance(out.get("text"), str) and out["text"]
    assert isinstance(out.get("citations"), list)
