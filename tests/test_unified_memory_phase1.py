# tests/test_unified_memory_phase1.py
import os
from pathlib import Path

import pytest

from memory.unified_memory import UnifiedMemory


def make_um(tmp_path: Path) -> UnifiedMemory:
    # vector_dim pequeño para velocidad; CPU siempre
    return UnifiedMemory(
        memory_dir=str(tmp_path / "memstore"),
        vector_dim=64,
        use_gpu=False,
    )


def test_labels_propagate_and_counters(tmp_path: Path):
    um = make_um(tmp_path)

    # Add dos recuerdos etiquetados
    tid = "trace-123"
    um.add_to_vector_memory(
        "Kant publicó la Crítica en 1781.",
        metadata={"source": "test"},
        result_quality="good",
        confidence=0.9,
        trace_id=tid,
    )
    um.add_to_vector_memory(
        "La Ilustración enfatiza la autonomía.",
        metadata={"source": "test"},
        result_quality="ok",
        confidence=0.7,
        trace_id=tid,
    )

    # Counters después de adds
    st = um.get_status()
    assert st["counters"]["adds_total"] == 2
    assert st["last_trace_id"] == tid

    # Buscar y verificar que viajan las etiquetas
    res = um.retrieve_relevant_memories("Crítica razón pura", top_k=2)
    assert isinstance(res, list) and len(res) > 0
    r0 = res[0]
    assert "result_quality" in r0 and "confidence" in r0 and "trace_id" in r0
    assert isinstance(r0["confidence"], float)

    # Counters y latencias de búsqueda
    st = um.get_status()
    assert st["counters"]["search_total"] >= 1
    lat = st["latency_ms"]
    assert lat["add_avg"] is not None and lat["search_avg"] is not None
    assert "add_p95" in lat and "search_p95" in lat


def test_interactions_reflections_events_and_aliases(tmp_path: Path):
    um = make_um(tmp_path)

    # Interacciones
    um.add_interaction(
        "hola", "qué tal",
        result_quality="good", confidence=0.8, trace_id="t-int"
    )
    ctx = um.get_context(limit=2)
    assert isinstance(ctx, list) and len(ctx) >= 1
    # Estructura esperada: lista de tuplas (role, content)
    assert isinstance(ctx[0], tuple) and len(ctx[0]) == 2

    # Reflexiones
    um.store_reflection("Insight A", content="detalle X",
                        result_quality="ok", confidence=0.6, trace_id="t-refl")
    refl = um.retrieve_recent_reflections(limit=1)
    assert len(refl) == 1
    # Debe traer alias esperados por tests ('contenido' / 'titulo')
    assert "contenido" in refl[0] and "titulo" in refl[0]
    assert "result_quality" in refl[0] and "trace_id" in refl[0]

    # Eventos
    um.store_event("system", "evento Y",
                   result_quality="bad", confidence=0.4, trace_id="t-ev")
    evs = um.get_recent_events(limit=1)
    assert len(evs) == 1
    # Alias 'contenido' / 'tipo'
    assert "contenido" in evs[0] and "tipo" in evs[0]
    assert evs[0]["tipo"] in ("system", "event")


def test_search_raw_and_tags(tmp_path: Path):
    um = make_um(tmp_path)
    um.add_memory("dato A", result_quality="good", confidence=0.95, trace_id="t1")
    res = um.search_memory("dato", top_k=1)
    assert len(res) == 1
    r = res[0]
    # Campo 'distance' + etiquetas en resultados crudos
    assert "distance" in r
    assert "result_quality" in r and "confidence" in r and "trace_id" in r


def test_persistence_roundtrip(tmp_path: Path):
    um = make_um(tmp_path)
    um.add_to_vector_memory("persistencia 1",
                            result_quality="ok", confidence=0.5, trace_id="round-1")
    um.add_to_vector_memory("persistencia 2",
                            result_quality="ok", confidence=0.6, trace_id="round-2")
    um.save_to_disk()

    # Nuevo objeto sobre el mismo directorio
    um2 = UnifiedMemory(memory_dir=str(tmp_path / "memstore"), vector_dim=64, use_gpu=False)
    mems = um2.export_memories()
    assert isinstance(mems, list) and len(mems) >= 2
    # Asegura que se conservaron etiquetas
    assert "result_quality" in mems[-1] and "confidence" in mems[-1] and "trace_id" in mems[-1]
