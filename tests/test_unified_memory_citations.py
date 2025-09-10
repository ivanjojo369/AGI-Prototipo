# tests/test_unified_memory_citations.py
from pathlib import Path
from memory.unified_memory import UnifiedMemory

def test_citations_and_ids(tmp_path: Path):
    um = UnifiedMemory(memory_dir=str(tmp_path/"mem"), vector_dim=64, use_gpu=False)
    um.add_to_vector_memory("Kant publicó en 1781", result_quality="good", confidence=0.9, trace_id="t1")
    um.add_to_vector_memory("Otro dato irrelevante", result_quality="ok", confidence=0.5, trace_id="t2")

    raw = um.search_memory("Kant 1781", top_k=2)
    assert len(raw) >= 1
    assert "mem_id" in raw[0] and "citation_id" in raw[0]
    assert raw[0]["mem_id"].startswith("mem://")

    rich = um.retrieve_relevant_memories("Kant 1781", top_k=2)
    assert len(rich) >= 1
    r0 = rich[0]
    assert "mem_id" in r0 and "citation_id" in r0
    assert isinstance(r0["confidence"], float)  # etiqueta sigue presente
