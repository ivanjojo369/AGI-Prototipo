"""Pruebas funcionales que no requieren Qwen ni los datos privados del usuario."""
from types import SimpleNamespace
from scientific_engine.engine import ScientificEngine
from memory.unified_memory import UnifiedMemory
from quipu_loop import QuipuLoop

QUESTION = (
    "Resuelve la ecuación x² - 4 = 0, calcula la derivada de x³ + 2x "
    "y convierte 72 km/h a m/s. Explica brevemente cada resultado."
)


def test_multi_science_three_tools_no_llm():
    result = ScientificEngine().handle(QUESTION)
    assert result.handled is True
    assert result.deterministic is True
    assert result.intent == "multi_operation"
    assert result.metadata["count"] == 3
    assert not result.metadata["tool_error"]
    assert result.facts["tasks"][0]["facts"]["solutions"] == ["-2", "2"]
    assert "3*x**2 + 2" in result.answer
    assert result.facts["tasks"][2]["facts"]["output_value"] == 20.0


def test_single_science_unmodified():
    result = ScientificEngine().handle("Deriva x³ + 2x")
    assert result.intent == "derivative" and result.deterministic


def test_legacy_followup_phrase_recognized():
    assert UnifiedMemory.is_contextual_followup("¿Cómo se llama el proyecto ficticio que te mencioné, cuáles son sus tres módulos?")
    assert not UnifiedMemory.is_contextual_followup("Hola")
    assert not UnifiedMemory.is_contextual_followup("Ahora hablemos de otra cosa")


def minimal_loop():
    loop = object.__new__(QuipuLoop)
    loop._conversation_turns_by_session = {}
    loop._active_recent_context_override = None
    loop._active_session_id = None
    loop.memory = SimpleNamespace(is_contextual_followup=lambda _t: False)
    loop.cfg = SimpleNamespace(use_planner=False, use_rag=False)
    loop._render_orchestration_context = lambda _orch: ""
    loop._retrieve_recalls_for_cognition = lambda *_a, **_k: []
    loop._render_memory = lambda _a: ""
    return loop


def test_recent_user_facts_have_priority_in_messages():
    loop = minimal_loop()
    loop._remember_session_turn(
        "Recuerda: Helios tiene Atlas, Nexo y Prisma; estudia exoplanetas.",
        "Comprendido.", None,
    )
    query = "¿Cómo se llama el proyecto ficticio que te mencioné y cuáles son sus tres módulos?"
    assert loop._is_contextual_followup(query)
    messages = loop._build_messages(query)
    assert messages[-1]["content"] == query
    assert "[CONTEXTO_RECIENTE_DEL_USUARIO]" in messages[-2]["content"]
    assert all(k in messages[-2]["content"] for k in ("Helios", "Atlas", "Nexo", "Prisma"))


def test_two_conversations_do_not_bleed():
    loop = minimal_loop()
    loop._remember_session_turn("Proyecto Helios: Atlas y Nexo", "ok", "chat-a")
    loop._remember_session_turn("Proyecto Lumen: Prisma", "ok", "chat-b")
    loop._active_session_id = "chat-b"
    assert not any("Helios" in r["content"] for r in loop._session_recent_turns())
    assert any("Lumen" in r["content"] for r in loop._session_recent_turns())
    loop._active_recent_context_override = [
        {"role": "user", "content": "Solo datos externos de chat-B"}
    ]
    assert len(loop._session_recent_turns()) == 1
    assert "chat-B" in loop._session_recent_turns()[0]["content"]


def test_planning_not_blocked_by_unrelated_semantic_missing_key():
    loop = minimal_loop()
    state = SimpleNamespace(turn_metadata={"metacognitive_gate": {
        "can_answer": False, "status": "insufficient_information",
        "missing_keys": ["semantic.aster.proyecto"],
    }})
    prompt = "Quiero desarrollar un sistema de investigación de fósiles con IA. Divide el proyecto en fases."
    assert loop._metacognitive_block_answer(state, prompt) is None
    msg = loop._metacognitive_block_answer(state, "¿Cuál es la cifra exacta de ayer?")
    assert msg and "semantic.aster.proyecto" not in msg


def test_not_bypass_context_dependent_plan():
    assert not QuipuLoop._self_contained_planning_request("Explica ese plan anterior")
