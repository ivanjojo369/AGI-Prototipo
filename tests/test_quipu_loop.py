# tests/test_quipu_loop.py
from quipu_loop import run_loop
from rag.retriever import add_document

def test_cycle_smoke():
    add_document("La Crítica de la Razón Pura es una obra de Kant que examina las condiciones del conocimiento a priori.", {"src": "nota-local"})
    out = run_loop("Explica en 3 líneas la idea central de la Crítica de la Razón Pura.", project_id="prueba")
    assert out["ok"] is True
    assert isinstance(out["output"], str) and len(out["output"]) > 20
