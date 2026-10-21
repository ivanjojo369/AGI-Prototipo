from cognition.cognitive_state import CognitiveState
from cognition.metacognitive_gate import MetacognitiveGate


def _print_decision(title, decision):
    print("=" * 100)
    print(title)
    print("CAN ANSWER:", decision.can_answer)
    print("STATUS:", decision.status)
    print("KNOWN:", decision.known_keys)
    print("MISSING:", decision.missing_keys)
    print("CONFLICTS:", decision.conflicting_keys)
    print("ASSUMPTIONS:", decision.assumptions)
    print("COMPLETENESS:", decision.evidence_completeness)
    print("EPISTEMIC CONFIDENCE:", decision.epistemic_confidence)
    print("ACTION:", decision.recommended_action)
    print("REASON:", decision.reason)


def test_population_projection_is_blocked_without_growth_model():
    state = CognitiveState(
        user_query=(
            "En el año 100, ¿cuántos habitantes tendrá la colonia?"
        )
    )

    state.world_state = {
        "colony.population.initial": 100,
        "colony.year.initial": 1,
        "colony.year.current": 100,
    }

    gate = MetacognitiveGate()

    decision = gate.require(
        state,
        [
            "colony.population.initial",
            "colony.year.initial",
            "colony.year.current",
            "colony.population.growth_rate",
        ],
    )

    assert decision.can_answer is False
    assert decision.status == "insufficient_information"
    assert decision.recommended_action == "ask_for_data"
    assert decision.missing_keys == [
        "colony.population.growth_rate"
    ]

    _print_decision(
        "TEST 1 — No inventar 2.5 millones sin tasa/modelo demográfico",
        decision,
    )


def test_population_projection_is_allowed_when_required_data_exists():
    state = CognitiveState(
        user_query=(
            "En el año 100, ¿cuántos habitantes tendrá la colonia?"
        )
    )

    state.world_state = {
        "colony.population.initial": 100,
        "colony.year.initial": 1,
        "colony.year.current": 100,
        "colony.population.growth_rate": 0.02,
    }

    gate = MetacognitiveGate()

    decision = gate.require(
        state,
        [
            "colony.population.initial",
            "colony.year.initial",
            "colony.year.current",
            "colony.population.growth_rate",
        ],
    )

    assert decision.can_answer is True
    assert decision.status == "ready"
    assert decision.missing_keys == []
    assert decision.recommended_action == "answer"

    _print_decision(
        "TEST 2 — Con modelo demográfico suficiente, el razonamiento puede continuar",
        decision,
    )


def test_food_reserves_are_ready_when_all_dependencies_exist():
    state = CognitiveState(
        user_query=(
            "¿Alcanzan las reservas actuales para alimentar a la colonia "
            "durante el primer año?"
        )
    )

    state.world_state = {
        "colony.population.initial": 100,
        "colony.food_reserves_kg": 30000,
        "colony.food_consumption_kg_per_person_day": 0.8,
    }

    gate = MetacognitiveGate()

    decision = gate.require(
        state,
        [
            "colony.population.initial",
            "colony.food_reserves_kg",
            "colony.food_consumption_kg_per_person_day",
        ],
    )

    assert decision.can_answer is True
    assert decision.status == "ready"
    assert decision.evidence_completeness == 1.0

    _print_decision(
        "TEST 3 — Población + reservas + consumo: listo para calcular",
        decision,
    )


def test_relevant_conflict_blocks_answer():
    state = CognitiveState(
        user_query="¿Cuál es el clima dominante del planeta?"
    )

    state.world_state = {
        "planet.climate.dominant": "calido humedo",
    }

    state.conflicts = [
        (
            "Conflicto activo en planet.climate.dominant: "
            "'desertico' source=model"
        )
    ]

    gate = MetacognitiveGate()

    decision = gate.require(
        state,
        ["planet.climate.dominant"],
    )

    assert decision.can_answer is False
    assert decision.status == "conflict"
    assert decision.recommended_action == "resolve_conflict"
    assert decision.conflicting_keys == [
        "planet.climate.dominant"
    ]

    _print_decision(
        "TEST 4 — Un conflicto relevante bloquea una respuesta definitiva",
        decision,
    )


def test_missing_from_long_range_is_preserved_and_not_invented():
    state = CognitiveState(
        user_query=(
            "Con toda la información anterior, ¿alcanzan las reservas?"
        )
    )

    state.world_state = {
        "colony.population.initial": 100,
        "colony.food_reserves_kg": 30000,
    }

    state.missing_information = [
        (
            "Falta información requerida: "
            "colony.food_consumption_kg_per_person_day"
        )
    ]

    gate = MetacognitiveGate()

    decision = gate.require(
        state,
        [
            "colony.population.initial",
            "colony.food_reserves_kg",
            "colony.food_consumption_kg_per_person_day",
        ],
    )

    assert decision.can_answer is False
    assert decision.status == "insufficient_information"
    assert "colony.food_consumption_kg_per_person_day" in decision.missing_keys

    _print_decision(
        "TEST 5 — Respeta el missing detectado por LongRangeRetriever",
        decision,
    )


def test_low_confidence_evidence_requires_review():
    state = CognitiveState(
        user_query="¿Cuál es el valor vigente?"
    )

    state.world_state = {
        "test.value": 42,
    }

    gate = MetacognitiveGate(
        min_epistemic_confidence=0.70,
    )

    decision = gate.require(
        state,
        ["test.value"],
        fact_confidence={
            "test.value": 0.20,
        },
    )

    assert decision.can_answer is False
    assert decision.status == "review_required"
    assert decision.recommended_action == "review"

    _print_decision(
        "TEST 6 — Dato presente pero de baja confianza: requiere revisión",
        decision,
    )


def main():
    test_population_projection_is_blocked_without_growth_model()
    test_population_projection_is_allowed_when_required_data_exists()
    test_food_reserves_are_ready_when_all_dependencies_exist()
    test_relevant_conflict_blocks_answer()
    test_missing_from_long_range_is_preserved_and_not_invented()
    test_low_confidence_evidence_requires_review()

    print("=" * 100)
    print("METACOGNITIVE GATE 1.0.1: 6/6 pruebas superadas.")


if __name__ == "__main__":
    main()
