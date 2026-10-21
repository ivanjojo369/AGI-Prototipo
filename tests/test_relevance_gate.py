from cognition.cognitive_state import CognitiveState, MemoryCandidate
from cognition.relevance_gate import MemoryRelevanceGate


def _show(title, state):
    print("=" * 88)
    print(title)
    print("QUERY:", state.user_query)
    print("ACCEPTED:")
    for c in state.accepted_memories:
        info = c.metadata.get("relevance_gate", {})
        print("  +", c.content)
        print("    score=", info.get("score"), "reason=", info.get("reason"))

    print("REJECTED:")
    for c in state.rejected_memories:
        info = c.metadata.get("relevance_gate", {})
        print("  -", c.content)
        print("    score=", info.get("score"), "reason=", info.get("reason"))


def test_aster_does_not_hijack_asteroid_question():
    state = CognitiveState(
        user_query="¿Cómo defenderíamos la colonia contra asteroides o cometas?"
    )

    state.memory_candidates = [
        MemoryCandidate(
            content="proyecto = Aster; lenguaje = Python; estado = segunda fase",
            memory_type="semantic",
            confidence=1.0,
            metadata={
                "entities": ["Aster"],
                "tags": ["proyecto", "vision artificial", "python"],
                "retrieval_score": 0.98,
            },
        ),
        MemoryCandidate(
            content="La colonia necesita un sistema de defensa y alerta contra asteroides.",
            memory_type="episodic",
            confidence=0.9,
            metadata={
                "tags": ["colonia", "defensa", "asteroides", "alerta"],
                "retrieval_score": 0.80,
            },
        ),
    ]

    gate = MemoryRelevanceGate()
    gate.filter_state(state)

    assert len(state.accepted_memories) == 1
    assert "asteroides" in str(state.accepted_memories[0].content).lower()

    assert len(state.rejected_memories) == 1
    assert "proyecto = Aster" in str(state.rejected_memories[0].content)

    _show("TEST 1 — Aster no secuestra una pregunta sobre asteroides", state)


def test_aster_is_accepted_when_query_is_about_aster():
    state = CognitiveState(user_query="¿Qué recuerdas sobre Aster?")

    state.memory_candidates = [
        MemoryCandidate(
            content="proyecto = Aster; lenguaje = Python; estado = segunda fase",
            memory_type="semantic",
            confidence=1.0,
            metadata={
                "entities": ["Aster"],
                "tags": ["Aster", "Python", "proyecto"],
                "retrieval_score": 0.95,
            },
        )
    ]

    gate = MemoryRelevanceGate()
    gate.filter_state(state)

    assert len(state.accepted_memories) == 1
    assert len(state.rejected_memories) == 0

    _show("TEST 2 — Aster sí entra cuando la consulta realmente es sobre Aster", state)


def test_workstation_goal_does_not_hijack_colony_reserves():
    state = CognitiveState(
        user_query=(
            "Con toda la información anterior, ¿las reservas actuales "
            "alcanzan para alimentar a la colonia durante el primer año?"
        )
    )

    state.memory_candidates = [
        MemoryCandidate(
            content={
                "objetivo": "migrar Quipu a la workstation nueva",
                "estado": "pendiente",
                "condicion": "terminar Memory 3.0",
            },
            memory_type="prospective",
            confidence=1.0,
            metadata={
                "entities": ["Quipu"],
                "tags": ["workstation", "migracion", "Memory 3.0"],
                "retrieval_score": 0.99,
            },
        ),
        MemoryCandidate(
            content="La colonia comenzó con 100 habitantes y 40000 kg de reservas.",
            memory_type="episodic",
            confidence=0.95,
            metadata={
                "tags": ["colonia", "100 habitantes", "reservas", "40000 kg"],
                "status": "historical",
                "retrieval_score": 0.82,
            },
        ),
        MemoryCandidate(
            content="Antes de la primera cosecha se perdió el 25% de las reservas.",
            memory_type="episodic",
            confidence=0.95,
            metadata={
                "tags": ["colonia", "reservas", "25%", "averia"],
                "status": "current",
                "retrieval_score": 0.79,
            },
        ),
    ]

    gate = MemoryRelevanceGate()
    gate.filter_state(state)

    accepted_text = " ".join(str(c.content).lower() for c in state.accepted_memories)
    rejected_text = " ".join(str(c.content).lower() for c in state.rejected_memories)

    assert "workstation" not in accepted_text
    assert "workstation" in rejected_text
    assert "reservas" in accepted_text

    _show("TEST 3 — El objetivo de workstation no secuestra la colonia", state)


def test_memory_meta_question_rejects_unrelated_aster_memory():
    state = CognitiveState(
        user_query="¿Cómo está tu persistencia de memoria y qué alcance puedes mantener?"
    )

    state.memory_candidates = [
        MemoryCandidate(
            content="proyecto = Aster; lenguaje = Python; estado = segunda fase",
            memory_type="semantic",
            confidence=1.0,
            metadata={
                "entities": ["Aster"],
                "tags": ["proyecto", "python", "vision artificial"],
                "retrieval_score": 0.97,
            },
        )
    ]

    gate = MemoryRelevanceGate()
    gate.filter_state(state)

    assert len(state.accepted_memories) == 0
    assert len(state.rejected_memories) == 1

    _show("TEST 4 — Una pregunta metacognitiva sobre memoria no devuelve Aster", state)


def main():
    test_aster_does_not_hijack_asteroid_question()
    test_aster_is_accepted_when_query_is_about_aster()
    test_workstation_goal_does_not_hijack_colony_reserves()
    test_memory_meta_question_rejects_unrelated_aster_memory()

    print("=" * 88)
    print("RELEVANCE GATE 1.0.1: 4/4 pruebas superadas.")


if __name__ == "__main__":
    main()
