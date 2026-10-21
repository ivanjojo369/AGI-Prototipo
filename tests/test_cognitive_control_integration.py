from __future__ import annotations

import tempfile
from pathlib import Path

from quipu_loop import LoopConfig, QuipuLoop
from cognition.executive_controller import ExecutiveController
from cognition.relevance_gate import MemoryRelevanceGate
from cognition.working_world_state import WorkingWorldState
from cognition.long_range_retriever import LongRangeRetriever
from cognition.metacognitive_gate import MetacognitiveGate
from cognition.self_correction import SelfCorrectionLoop
from cognition.conversation_state import ConversationStateManager
from cognition.output_guard import OutputGuard
from memory.unified_memory import UnifiedMemory
from tools.verification import Verifier


class FakeAdapter:
    def generate(
        self,
        messages,
        max_tokens=512,
        temperature=0.7,
        stream=False,
        stop=None,
    ):
        joined = "\n".join(
            str(m.get("content") or "")
            for m in messages
        )

        if "auditor de dependencias" in joined:
            return (
                '{"sufficient":false,'
                '"missing":["dato cuantitativo faltante"],'
                '"assumptions":[]}'
            )

        return "Respuesta limpia en español."


class FakeMemory:
    def __init__(self, recalls=None, context=None):
        self.recalls = list(recalls or [])
        self.context = list(context or [])
        self.saved = []
        self.saved_metadata = []

    @staticmethod
    def is_contextual_followup(query):
        return UnifiedMemory.is_contextual_followup(
            query
        )

    def _saved_as_recalls(self):
        rows = []
        for (role, content), metadata in zip(
            self.saved,
            self.saved_metadata,
        ):
            rows.append(
                {
                    "role": role,
                    "content": content,
                    "source": "interaction",
                    "score": 0.55,
                    "metadata": dict(metadata),
                    "context_reuse": metadata.get(
                        "context_reuse",
                        True,
                    ),
                }
            )
        return rows

    def recall(self, query, top_k=10):
        rows = (
            list(self.recalls)
            + self._saved_as_recalls()
        )
        return rows[-top_k:]

    def get_context_records(
        self,
        limit=8,
        *,
        reusable_only=False,
    ):
        rows = [
            {
                "role": role,
                "content": content,
                "metadata": {},
            }
            for role, content in self.context
        ]

        for (role, content), metadata in zip(
            self.saved,
            self.saved_metadata,
        ):
            if (
                reusable_only
                and metadata.get(
                    "context_reuse"
                )
                is False
            ):
                continue

            rows.append(
                {
                    "role": role,
                    "content": content,
                    "metadata": dict(metadata),
                }
            )

        return rows[-limit:]

    def get_context(self, limit=8):
        return [
            (
                item["role"],
                item["content"],
            )
            for item in self.get_context_records(
                limit=limit,
                reusable_only=False,
            )
        ]

    def add(self, role, content, metadata=None):
        self.saved.append(
            (role, content)
        )
        self.saved_metadata.append(
            dict(metadata or {})
        )
        return len(self.saved) - 1


class FakeRag:
    def retrieve(self, query, top_k=4):
        return []

    def render_context(self, docs):
        return ""


class FakePlanner:
    def plan(self, user_text):
        return None


class FakeVerifier:
    def verify(self, answer, messages, evidence=None, structured_facts=None):
        return str(answer)

    @staticmethod
    def _check_arithmetic(answer):
        return []

    @staticmethod
    def _check_relational_reasoning(user_text, answer):
        return []


def make_loop(recalls=None, context=None):
    loop = object.__new__(QuipuLoop)

    loop.cfg = LoopConfig(
        model_backend="stub",
        use_rag=False,
        use_planner=False,
        use_verifier=True,
        use_cognitive_control=True,
    )

    loop.adapter = FakeAdapter()
    loop.memory = FakeMemory(
        recalls=recalls,
        context=context,
    )
    loop.rag = FakeRag()
    loop.planner = FakePlanner()
    loop.verifier = FakeVerifier()

    loop.session_path = (
        Path(tempfile.mkdtemp())
        / "session.jsonl"
    )

    loop._last_recalls = []
    loop._last_orchestration = {}
    loop._last_cognitive_state = None

    loop.orchestrator = None

    loop.cognitive_controller = ExecutiveController()
    loop.relevance_gate = MemoryRelevanceGate()
    loop.working_world_state = WorkingWorldState()
    loop.long_range_retriever = LongRangeRetriever(
        relevance_gate=loop.relevance_gate
    )
    loop.metacognitive_gate = MetacognitiveGate()
    loop.self_correction = SelfCorrectionLoop(
        max_attempts=2
    )
    loop.conversation_state_manager = ConversationStateManager()
    loop.output_guard = OutputGuard(
        expected_language="es"
    )
    loop._topic_seq = 0

    return loop


def test_cognitive_control_is_active():
    loop = make_loop()

    assert loop._cognitive_enabled() is True

    print("=" * 100)
    print("TEST 1 — Cognitive Control está activo")


def test_aster_is_filtered_from_asteroid_question():
    recalls = [
        {
            "role": "memory",
            "content": (
                "proyecto = Aster\n"
                "lenguaje = Python\n"
                "estado = segunda fase"
            ),
            "source": "semantic",
            "semantic_subject": "Aster",
            "score": 0.99,
            "fields": {
                "proyecto": "Aster",
                "lenguaje": "Python",
                "estado": "segunda fase",
            },
        },
        {
            "role": "memory",
            "content": (
                "La colonia necesita un sistema de defensa "
                "y alerta contra asteroides."
            ),
            "source": "episode",
            "episode_key": "colonia",
            "score": 0.80,
            "fields": {},
        },
    ]

    loop = make_loop(recalls=recalls)
    state = loop.cognitive_controller.analyze(
        "¿Cómo defenderíamos la colonia contra asteroides o cometas?"
    )

    filtered = loop._apply_cognitive_memory_control(
        state,
        recalls,
    )

    joined = "\n".join(
        item["content"]
        for item in filtered
    )

    assert "Aster" not in joined
    assert "asteroides" in joined
    assert len(state.accepted_memories) == 1
    assert len(state.rejected_memories) == 1

    print("=" * 100)
    print("TEST 2 — Aster no secuestra una pregunta sobre asteroides")
    print("FILTERED:", joined)


def test_population_projection_is_blocked_without_growth_model():
    loop = make_loop()

    answer = loop.run_once(
        "En el año 100, ¿cuántos habitantes tendrá la colonia?"
    )

    state = loop._last_cognitive_state

    assert state is not None
    assert (
        "tasa o modelo demográfico de crecimiento"
        in answer
    )
    assert state.missing_information
    assert (
        state.turn_metadata[
            "metacognitive_gate"
        ]["status"]
        == "insufficient_information"
    )

    print("=" * 100)
    print("TEST 3 — Proyección demográfica sin modelo queda bloqueada")
    print("ANSWER:", answer)


def test_normal_chat_still_works():
    loop = make_loop()

    answer = loop.run_once(
        "Hola, ¿cómo estás?"
    )

    assert answer == "Respuesta limpia en español."
    assert loop.memory.saved[-1] == (
        "assistant",
        "Respuesta limpia en español.",
    )

    print("=" * 100)
    print("TEST 4 — Chat normal sigue funcionando")
    print("ANSWER:", answer)


def test_self_review_does_not_feed_previous_assistant_answer_back():
    context = [
        (
            "user",
            "¿Cuánto quedaba de reservas?",
        ),
        (
            "assistant",
            "Quedan 40000 kg.",
        ),
    ]

    loop = make_loop(context=context)
    # Core 4.0.1: los turnos activos viven en la sesión, no en memoria global.
    loop._conversation_turns_by_session = {
        "__cli__": [
            {"role": "user", "content": context[0][1]},
            {"role": "assistant", "content": context[1][1]},
        ]
    }

    state = loop._prepare_cognitive_state(
        "Revisa lo que dijiste antes y corrige el cálculo."
    )

    messages = loop._build_messages(
        "Revisa lo que dijiste antes y corrige el cálculo.",
        orchestration={
            "ok": True,
            "skill": "chat",
            "confidence": 0.55,
            "reason": "test",
            "output": None,
        },
        cognitive_state=state,
    )

    rendered = "\n".join(
        str(item.get("content") or "")
        for item in messages
    )

    history = (
        state.turn_metadata
        .get("self_correction", {})
        .get("history", [])
    )

    assert history
    assert (
        history[0]["previous_answer"]
        == "Quedan 40000 kg."
    )
    assert "Quedan 40000 kg." not in rendered
    assert "Autocorrección obligatoria" in rendered

    print("=" * 100)
    print("TEST 5 — SelfCorrection archiva pero no recicla la respuesta previa")



def test_independent_greeting_does_not_inherit_demographic_context():
    context = [
        (
            "user",
            "Una colonia comienza con 100 habitantes. "
            "¿Cuántos tendrá dentro de 100 años?",
        ),
        (
            "assistant",
            "No tengo suficiente información. "
            "Me falta una tasa de crecimiento.",
        ),
    ]

    recalls = [
        {
            "role": "user",
            "content": context[0][1],
            "source": "interaction",
            "score": 0.55,
            "metadata": {},
        },
        {
            "role": "assistant",
            "content": context[1][1],
            "source": "interaction",
            "score": 0.55,
            "metadata": {
                "context_reuse": False,
                "cognitive_status": "blocked",
            },
            "context_reuse": False,
        },
    ]

    loop = make_loop(
        recalls=recalls,
        context=context,
    )

    state = loop._prepare_cognitive_state(
        "Hola, ¿cómo estás?"
    )

    assert state.recent_context == []
    assert (
        state.turn_metadata["context_policy"]
        ["contextual_followup"]
        is False
    )

    messages = loop._build_messages(
        "Hola, ¿cómo estás?",
        orchestration={
            "ok": True,
            "skill": "chat",
            "confidence": 0.5,
            "reason": "test",
            "output": None,
        },
        cognitive_state=state,
    )

    rendered = "\n".join(
        str(item.get("content") or "")
        for item in messages
    )

    assert "tasa de crecimiento" not in rendered
    assert "100 habitantes" not in rendered

    print("=" * 100)
    print("TEST 6 — Un saludo independiente no hereda el problema anterior")


def test_real_followup_keeps_relevant_recent_context_but_not_blocked_answer():
    context = [
        (
            "user",
            "Una colonia comienza con 100 habitantes. "
            "¿Cuántos tendrá dentro de 100 años?",
        ),
        (
            "assistant",
            "No tengo suficiente información. "
            "Me falta una tasa de crecimiento.",
        ),
    ]

    loop = make_loop(
        context=context,
        recalls=[
            {
                "role": "user",
                "content": context[0][1],
                "source": "interaction",
                "score": 0.55,
                "metadata": {},
            },
            {
                "role": "assistant",
                "content": context[1][1],
                "source": "interaction",
                "score": 0.55,
                "metadata": {
                    "context_reuse": False,
                    "cognitive_status": "blocked",
                },
                "context_reuse": False,
            },
        ],
    )

    # Core 4.0.1: contexto permitido solo desde el chat actual.
    # Se conserva la pregunta anterior, nunca la respuesta bloqueada.
    loop._conversation_turns_by_session = {
        "__cli__": [
            {"role": "user", "content": context[0][1]},
            {
                "role": "assistant",
                "content": context[1][1],
                "metadata": {"context_reuse": False, "cognitive_status": "blocked"},
            },
        ]
    }
    state = loop._prepare_cognitive_state(
        "¿Y si te doy una tasa de crecimiento anual del 2%?"
    )

    assert state.recent_context
    assert (
        state.turn_metadata["context_policy"]
        ["contextual_followup"]
        is True
    )

    messages = loop._build_messages(
        "¿Y si te doy una tasa de crecimiento anual del 2%?",
        orchestration={
            "ok": True,
            "skill": "chat",
            "confidence": 0.5,
            "reason": "test",
            "output": None,
        },
        cognitive_state=state,
    )

    rendered = "\n".join(
        str(item.get("content") or "")
        for item in messages
    )

    assert "100 habitantes" in rendered
    assert "Me falta una tasa de crecimiento." not in rendered

    print("=" * 100)
    print("TEST 7 — Una continuación real conserva contexto útil sin reciclar el bloqueo")


def test_y_eso_de_does_not_create_a_new_topic():
    loop = make_loop()

    first = loop._prepare_conversation_tracking(
        "Hablemos del sistema energético de la colonia."
    )
    current_before = loop.conversation_state_manager.current_topic_id

    assert current_before is not None

    loop._prepare_conversation_tracking(
        "Y eso de la energía solar, ¿cómo lo implementaríamos?"
    )

    current_after = loop.conversation_state_manager.current_topic_id

    assert current_after == current_before

    print("=" * 100)
    print("TEST 8 — 'y eso de...' profundiza el tema actual; no lo fragmenta")


def test_metacognitive_block_is_saved_as_non_reusable_context():
    loop = make_loop()

    answer = loop.run_once(
        "En el año 100, ¿cuántos habitantes tendrá la colonia?"
    )

    assert "tasa o modelo demográfico" in answer

    assistant_metadata = loop.memory.saved_metadata[-1]

    assert (
        assistant_metadata.get("context_reuse")
        is False
    )
    assert (
        assistant_metadata.get("cognitive_status")
        == "blocked"
    )
    assert (
        assistant_metadata.get("metacognitive_status")
        == "insufficient_information"
    )

    print("=" * 100)
    print("TEST 9 — Un bloqueo metacognitivo se guarda, pero no se reutiliza ciegamente")


def test_unified_memory_followup_policy_distinguishes_continuation_from_new_topic():
    assert (
        UnifiedMemory.is_contextual_followup(
            "Hola, ¿cómo estás?"
        )
        is False
    )
    assert (
        UnifiedMemory.is_contextual_followup(
            "¿Qué es la gravedad?"
        )
        is False
    )
    assert (
        UnifiedMemory.is_contextual_followup(
            "¿Y por qué?"
        )
        is True
    )
    assert (
        UnifiedMemory.is_contextual_followup(
            "Con toda la información anterior, ¿qué conclusión sacas?"
        )
        is True
    )
    assert (
        UnifiedMemory.is_contextual_followup(
            "¿Y si te doy una tasa de crecimiento anual del 2%? "
            "Calcula entonces cuántos habitantes tendría dentro de 100 años."
        )
        is True
    )
    assert (
        UnifiedMemory.is_contextual_followup(
            "Cambiando de tema, hablemos de volcanes."
        )
        is False
    )

    print("=" * 100)
    print("TEST 10 — La política central distingue seguimiento vs. nuevo tema")



def test_blocked_topic_is_reactivated_after_intermediate_greeting():
    loop = make_loop()

    first_answer = loop.run_once(
        "En el año 100, ¿cuántos habitantes tendrá la colonia "
        "si comienza con 100 habitantes?"
    )
    assert "tasa o modelo demográfico" in first_answer

    demographic_topic = (
        loop.conversation_state_manager
        .export_state()["topics"]["topic_1"]
    )
    assert demographic_topic["status"] == "blocked"

    greeting_answer = loop.run_once(
        "Hola, ¿cómo estás?"
    )
    assert greeting_answer == "Respuesta limpia en español."
    assert (
        loop.conversation_state_manager.current_topic_id
        == "topic_2"
    )

    state = loop._prepare_cognitive_state(
        "¿Y si te doy una tasa de crecimiento anual del 2%? "
        "Calcula entonces cuántos habitantes tendría dentro de 100 años."
    )

    assert (
        loop.conversation_state_manager.current_topic_id
        == "topic_1"
    )
    assert (
        loop.conversation_state_manager
        .get_topic("topic_1")
        .status
        == "active"
    )
    assert (
        state.turn_metadata["context_policy"]["topic_id"]
        == "topic_1"
    )

    print("=" * 100)
    print("TEST 11 — Un tema bloqueado se reactiva después de un saludo intermedio")


def test_explicit_growth_rate_enters_topic_world_state():
    loop = make_loop()

    loop.run_once(
        "En el año 100, ¿cuántos habitantes tendrá la colonia "
        "si comienza con 100 habitantes?"
    )
    loop.run_once(
        "Hola, ¿cómo estás?"
    )

    state = loop._prepare_cognitive_state(
        "¿Y si te doy una tasa de crecimiento anual del 2%? "
        "Calcula entonces cuántos habitantes tendría dentro de 100 años."
    )

    assert state.world_state["colony.population.initial"] == 100
    assert (
        state.world_state["colony.population.growth_rate_annual"]
        == 0.02
    )
    assert state.world_state["projection.horizon_years"] == 100

    print("=" * 100)
    print("TEST 12 — 2% anual se convierte en un hecho estructurado del tema")


def test_llm_auditor_cannot_erase_explicit_growth_rate():
    loop = make_loop()

    explicit = {
        "colony.population.growth_rate_annual": 0.02,
    }
    audit = {
        "sufficient": False,
        "missing": [
            "tasa de crecimiento anual",
        ],
        "assumptions": [],
    }

    clean = loop._sanitize_dependency_audit(
        "La tasa de crecimiento anual es del 2%.",
        audit,
        explicit,
    )

    assert clean["missing"] == []
    assert clean["sufficient"] is True
    assert clean["discarded_missing"] == [
        "tasa de crecimiento anual"
    ]

    print("=" * 100)
    print("TEST 13 — El auditor LLM no puede declarar ausente un dato explícito")


def test_population_projection_gate_becomes_ready_after_growth_rate():
    loop = make_loop()

    loop.run_once(
        "En el año 100, ¿cuántos habitantes tendrá la colonia "
        "si comienza con 100 habitantes?"
    )
    loop.run_once(
        "Hola, ¿cómo estás?"
    )

    state = loop._prepare_cognitive_state(
        "¿Y si te doy una tasa de crecimiento anual del 2%? "
        "Calcula entonces cuántos habitantes tendría dentro de 100 años."
    )

    loop._apply_cognitive_memory_control(
        state,
        [],
    )

    meta = state.turn_metadata[
        "metacognitive_gate"
    ]

    assert meta["status"] == "ready"
    assert meta["can_answer"] is True
    assert meta["missing_keys"] == []

    print("=" * 100)
    print("TEST 14 — 100 habitantes + 2% + 100 años deja al Gate en READY")



def test_population_projection_solver_returns_expected_value():
    loop = make_loop()

    solution = loop._population_projection_solution(
        {
            "colony.population.initial": 100,
            "colony.population.growth_rate_annual": 0.02,
            "projection.horizon_years": 100,
        }
    )

    assert solution is not None
    assert abs(solution["result"] - 724.4646118252348) < 1e-9

    answer = loop._format_population_projection_answer(
        solution
    )

    assert "724.4646" in answer
    assert "724 habitantes" in answer

    print("=" * 100)
    print("TEST 15 — El solver demográfico calcula 100*(1.02)^100 correctamente")


def test_verifier_corrects_wrong_population_projection_number():
    verifier = Verifier(adapter=None)

    messages = [
        {
            "role": "user",
            "content": (
                "Calcula cuántos habitantes tendría la colonia dentro de "
                "100 años."
            ),
        }
    ]

    corrected = verifier.verify(
        "271869.4053",
        messages,
        evidence=None,
        structured_facts={
            "colony.population.initial": 100,
            "colony.population.growth_rate_annual": 0.02,
            "projection.horizon_years": 100,
        },
    )

    assert "271869.4053" not in corrected
    assert "724.4646" in corrected
    assert "724 habitantes" in corrected

    print("=" * 100)
    print("TEST 16 — Verifier corrige una cifra demográfica incorrecta")


def test_run_once_uses_deterministic_population_solver():
    loop = make_loop()

    first = loop.run_once(
        "En el año 100, ¿cuántos habitantes tendrá la colonia "
        "si comienza con 100 habitantes?"
    )
    assert "tasa o modelo demográfico" in first

    greeting = loop.run_once(
        "Hola, ¿cómo estás?"
    )
    assert greeting == "Respuesta limpia en español."

    answer = loop.run_once(
        "¿Y si te doy una tasa de crecimiento anual del 2%? "
        "Calcula entonces cuántos habitantes tendría dentro de 100 años."
    )

    assert "724.4646" in answer
    assert "724 habitantes" in answer
    assert "271869.4053" not in answer

    meta = (
        loop._last_cognitive_state
        .turn_metadata
        .get("deterministic_quantitative_solver", {})
    )

    assert meta.get("kind") == "population_projection"
    assert abs(float(meta.get("result")) - 724.4646118252348) < 1e-9

    print("=" * 100)
    print("TEST 17 — run_once usa el solver determinista y no la aritmética de Qwen")


def main():
    test_cognitive_control_is_active()
    test_aster_is_filtered_from_asteroid_question()
    test_population_projection_is_blocked_without_growth_model()
    test_normal_chat_still_works()
    test_self_review_does_not_feed_previous_assistant_answer_back()
    test_independent_greeting_does_not_inherit_demographic_context()
    test_real_followup_keeps_relevant_recent_context_but_not_blocked_answer()
    test_y_eso_de_does_not_create_a_new_topic()
    test_metacognitive_block_is_saved_as_non_reusable_context()
    test_unified_memory_followup_policy_distinguishes_continuation_from_new_topic()
    test_blocked_topic_is_reactivated_after_intermediate_greeting()
    test_explicit_growth_rate_enters_topic_world_state()
    test_llm_auditor_cannot_erase_explicit_growth_rate()
    test_population_projection_gate_becomes_ready_after_growth_rate()
    test_population_projection_solver_returns_expected_value()
    test_verifier_corrects_wrong_population_projection_number()
    test_run_once_uses_deterministic_population_solver()

    print("=" * 100)
    print(
        "COGNITIVE CONTROL INTEGRATION 1.0.3: "
        "17/17 pruebas superadas."
    )


if __name__ == "__main__":
    main()
