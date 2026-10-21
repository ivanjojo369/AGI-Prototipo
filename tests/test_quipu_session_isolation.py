"""RC2: continuidad y aislamiento por sesión, sin LLM ni memoria del usuario."""
from types import SimpleNamespace

from quipu_loop import QuipuLoop


class SpyCognitiveController:
    def analyze(self, user_text, *, recent_context, world_state, conversation_state):
        return SimpleNamespace(
            user_query=user_text,
            recent_context=list(recent_context),
            turn_metadata={},
        )


def make_loop():
    loop = object.__new__(QuipuLoop)
    loop._conversation_turns_by_session = {}
    loop._active_session_id = None
    loop._active_recent_context_override = None
    loop.conversation_state_manager = None
    loop.cognitive_controller = SpyCognitiveController()
    loop._cognitive_enabled = lambda: True
    loop._prepare_conversation_tracking = lambda _text: {}
    loop._current_topic_id = lambda: None
    loop._ingest_explicit_turn_facts = lambda _text, *, topic_id: {}
    loop._topic_world_state_snapshot = lambda _topic_id: {}
    return loop


def test_two_conversations_have_separate_user_facts():
    loop = make_loop()
    loop._remember_session_turn("Helios: Atlas, Nexo, Prisma", "Bien.", "chat-a")
    loop._remember_session_turn("Lumen: Voyager", "Entendido.", "chat-b")
    loop._active_session_id = "chat-a"
    a = loop._session_recent_turns()
    loop._active_session_id = "chat-b"
    b = loop._session_recent_turns()
    assert any("Helios" in row["content"] for row in a)
    assert not any("Lumen" in row["content"] for row in a)
    assert any("Lumen" in row["content"] for row in b)
    assert not any("Helios" in row["content"] for row in b)


def test_blocked_answer_is_not_persisted_as_context():
    loop = make_loop()
    loop._remember_session_turn(
        "¿Cuántos habrá en un siglo?",
        "No tengo suficiente información para responder con seguridad.",
        "chat-a",
    )
    loop._active_session_id = "chat-a"
    recent = loop._session_recent_turns()
    assert len(recent) == 1 and recent[0]["role"] == "user"
    assert "siglo" in recent[0]["content"]


def test_http_override_excludes_non_reusable_answer_before_cognitive_analysis():
    loop = make_loop()
    loop._active_session_id = "chat-a"
    loop._active_recent_context_override = loop._normalize_recent_context_override([
        {"role": "user", "content": "Una colonia tiene 100 habitantes."},
        {"role": "assistant", "content": "RESPUESTA_BLOQUEADA", "metadata": {"context_reuse": False}},
        {"role": "assistant", "content": "Respuesta segura.", "metadata": {"context_reuse": True}},
    ])
    loop._is_contextual_followup = lambda _text: True
    state = loop._prepare_cognitive_state("¿Y si crece al 2%?")
    texts = [row["content"] for row in state.recent_context]
    assert "Una colonia tiene 100 habitantes." in texts
    assert "Respuesta segura." in texts
    assert "RESPUESTA_BLOQUEADA" not in texts


def test_explicit_empty_http_context_never_falls_back_to_another_session():
    loop = make_loop()
    loop._remember_session_turn("Helios secreto", "ok", "chat-a")
    loop._active_session_id = "chat-b"
    loop._active_recent_context_override = []
    loop._is_contextual_followup = lambda _text: True
    state = loop._prepare_cognitive_state("¿Y lo anterior?")
    assert state.recent_context == []


def test_non_followup_does_not_inherit_session_turns():
    loop = make_loop()
    loop._remember_session_turn("Helios secreto", "ok", "chat-a")
    loop._active_session_id = "chat-a"
    loop._is_contextual_followup = lambda _text: False
    state = loop._prepare_cognitive_state("Hola")
    assert state.recent_context == []


def test_cli_history_is_separate_from_http_chat():
    loop = make_loop()
    loop._remember_session_turn("CLI_PRIVATE", "ok", None)
    loop._remember_session_turn("HTTP_ONLY", "ok", "chat-01")
    loop._active_session_id = None
    assert all("HTTP_ONLY" not in r["content"] for r in loop._session_recent_turns())
    loop._active_session_id = "chat-01"
    assert all("CLI_PRIVATE" not in r["content"] for r in loop._session_recent_turns())


def test_http_metadata_false_is_preserved_by_normalizer():
    raw = [
        {"role": "assistant", "content": "No se debe reutilizar", "metadata": {"context_reuse": False}},
        {"role": "system", "content": "no permitido"},
    ]
    clean = QuipuLoop._normalize_recent_context_override(raw)
    assert len(clean) == 1
    assert clean[0]["metadata"]["context_reuse"] is False
