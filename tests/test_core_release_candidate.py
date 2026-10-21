from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from cognition.conversation_state import ConversationStateManager
from planner import htn_simple
from planner.planner import Planner
from quipu_loop import QUIPU_CORE_VERSION, QuipuLoop
from rag import retriever
from rag.pipeline import RagPipeline
from tests.test_scientific_engine_integration import make_loop


def test_1_core_version_is_final_4():
    assert QUIPU_CORE_VERSION == "4.0.0"


def test_2_rag_pipeline_contract_is_real_not_stub():
    old_store = retriever.RAG_STORE_JSON
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "rag_store.json"
        text = "La tectónica de placas explica gran parte de la actividad sísmica."
        store.write_text(
            json.dumps(
                [
                    {
                        "id": "geo::0",
                        "text": text,
                        "meta": {"path": "geology/plates.md", "chunk": 0},
                        "vec": retriever.embed_text(text),
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        try:
            retriever.RAG_STORE_JSON = str(store)
            pipe = RagPipeline(top_k=3, min_score=0.0)
            hits = pipe.retrieve("tectónica placas sismos")
            assert hits
            rendered = pipe.render_context(hits)
            assert "geology/plates.md" in rendered
            assert "tectónica" in rendered.lower()
        finally:
            retriever.RAG_STORE_JSON = old_store


def test_3_planner_never_calls_legacy_http_without_opt_in():
    old_env = os.environ.pop("QUIPU_ENABLE_LEGACY_PLANNER_HTTP", None)
    original = htn_simple._call_llm
    htn_simple._call_llm = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("No debe tocar :8010")
    )
    try:
        result = Planner().plan("Preparar un informe técnico y revisarlo")
        assert result["subgoals"]
    finally:
        htn_simple._call_llm = original
        if old_env is not None:
            os.environ["QUIPU_ENABLE_LEGACY_PLANNER_HTTP"] = old_env


def test_4_planner_uses_injected_current_model():
    seen = {"calls": 0}

    def fake(messages, max_tokens, temperature):
        seen["calls"] += 1
        return (
            '{"subgoals":['
            '{"desc":"Analizar requisitos","done_when":"requisitos claros"},'
            '{"desc":"Redactar resultado","done_when":"borrador listo"}'
            ']}'
        )

    result = Planner().plan("Preparar informe", llm_call=fake)
    assert seen["calls"] == 1
    assert len(result["subgoals"]) == 2


def _session_loop() -> QuipuLoop:
    loop = object.__new__(QuipuLoop)
    loop.conversation_state_manager = ConversationStateManager()
    loop._default_conversation_state_manager = loop.conversation_state_manager
    loop._conversation_state_managers = {}
    loop._active_session_id = None
    loop._active_recent_context_override = None
    loop._topic_seq = 0
    loop._topic_world_states = {}
    return loop


def test_5_conversation_managers_are_isolated_by_session():
    loop = _session_loop()
    loop._activate_conversation_session("chat-A")
    manager_a = loop.conversation_state_manager
    manager_a.start_topic("topic_same", "Tema de A")

    loop._activate_conversation_session("chat-B")
    manager_b = loop.conversation_state_manager

    assert manager_a is not manager_b
    assert manager_b.get_topic("topic_same") is None

    loop._activate_conversation_session("chat-A")
    assert loop.conversation_state_manager is manager_a
    assert loop.conversation_state_manager.get_topic("topic_same") is not None


def test_6_world_state_is_session_scoped_even_with_same_topic_id():
    loop = _session_loop()
    loop._activate_conversation_session("chat-A")
    state_a = loop._topic_world_state("topic_1")

    loop._activate_conversation_session("chat-B")
    state_b = loop._topic_world_state("topic_1")

    assert state_a is not state_b
    assert "chat-A::topic_1" in loop._topic_world_states
    assert "chat-B::topic_1" in loop._topic_world_states


def test_7_recent_context_override_is_bounded_and_clean():
    rows = [
        {"role": "system", "content": "ignorar"},
        *[
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(12)
        ],
    ]
    clean = QuipuLoop._normalize_recent_context_override(rows, limit=8)
    assert clean is not None
    assert len(clean) == 8
    assert clean[0]["content"] == "m4"
    assert all(row["role"] in ("user", "assistant") for row in clean)


def test_8_scientific_preflight_does_not_build_probabilistic_context():
    loop = make_loop(use_verifier=True)
    loop._build_messages = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("Ciencia determinista no debe construir RAG/planner/memory prompt")
    )
    answer = loop.run_once("Convierte 72 km/h a m/s")
    assert "20" in answer
    assert loop.adapter.calls == 0


def main():
    tests = [
        test_1_core_version_is_final_4,
        test_2_rag_pipeline_contract_is_real_not_stub,
        test_3_planner_never_calls_legacy_http_without_opt_in,
        test_4_planner_uses_injected_current_model,
        test_5_conversation_managers_are_isolated_by_session,
        test_6_world_state_is_session_scoped_even_with_same_topic_id,
        test_7_recent_context_override_is_bounded_and_clean,
        test_8_scientific_preflight_does_not_build_probabilistic_context,
    ]
    for idx, fn in enumerate(tests, 1):
        fn()
        print(f"TEST {idx:02d} — OK — {fn.__name__}")
    print("=" * 100)
    print("QUIPU CORE 4.0.0 RELEASE CANDIDATE: 8/8 pruebas superadas.")


if __name__ == "__main__":
    main()
