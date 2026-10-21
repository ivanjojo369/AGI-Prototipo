from cognition.cognitive_state import CognitiveState
from cognition.conversation_state import ConversationStateManager


def _print_state(title, manager):
    print("=" * 100)
    print(title)
    state = manager.export_state()

    print("CURRENT TOPIC:", state["current_topic_id"])
    print("TOPICS:")
    for key, value in state["topics"].items():
        print(
            f"  {key}: status={value['status']} "
            f"title={value['title']!r}"
        )

    print("DECISIONS:")
    for key, value in state["decisions"].items():
        print(
            f"  {key}: accepted={value['accepted']} "
            f"text={value['text']!r}"
        )

    print(
        "PENDING:",
        list(state["pending_questions"].keys()),
    )
    print(
        "HYPOTHESES:",
        {
            key: value["status"]
            for key, value in state["hypotheses"].items()
        },
    )
    print("DIRECTIVES:", state["directives"])


def test_resolved_topic_is_not_repeated_after_user_approval():
    manager = ConversationStateManager()

    manager.start_topic(
        "evacuation_plan",
        "Plan de evacuación de la colonia",
    )

    action = manager.apply_user_feedback(
        "evacuation_plan",
        "Sí, ese plan queda bien.",
        accepted_decision_id="evacuation_v1",
        accepted_decision_text="Usar el plan de evacuación propuesto.",
    )

    assert action == "resolved"
    assert manager.get_topic("evacuation_plan").status == "resolved"
    assert manager.should_repeat_topic("evacuation_plan") is False

    decisions = manager.accepted_decisions(
        topic_id="evacuation_plan"
    )
    assert len(decisions) == 1
    assert decisions[0].decision_id == "evacuation_v1"

    _print_state(
        "TEST 1 — Un plan aprobado pasa a RESOLVED y no debe repetirse",
        manager,
    )


def test_explicit_reopen_allows_returning_to_resolved_topic():
    manager = ConversationStateManager()

    manager.start_topic(
        "food_distribution",
        "Distribución de alimentos",
    )
    manager.resolve_topic(
        "food_distribution",
        summary="Plan inicial aprobado.",
    )

    assert manager.should_repeat_topic("food_distribution") is False

    action = manager.apply_user_feedback(
        "food_distribution",
        "Regresando a la distribución de alimentos, quiero cambiar algo.",
    )

    assert action == "reopened"
    assert manager.get_topic("food_distribution").status == "active"
    assert manager.should_repeat_topic("food_distribution") is True

    _print_state(
        "TEST 2 — Un tema resuelto puede reabrirse explícitamente",
        manager,
    )


def test_pending_question_lifecycle():
    manager = ConversationStateManager()

    manager.start_topic(
        "population_projection",
        "Proyección demográfica",
    )

    question = manager.add_pending_question(
        "growth_rate_needed",
        "¿Cuál es la tasa de crecimiento anual?",
        topic_id="population_projection",
    )

    assert question.status == "pending"
    assert len(manager.pending_questions()) == 1

    manager.resolve_pending_question(
        "growth_rate_needed",
        "La tasa será 2% anual.",
    )

    assert manager.pending_questions() == []
    assert question.status == "resolved"
    assert question.answer == "La tasa será 2% anual."

    _print_state(
        "TEST 3 — Las preguntas pendientes desaparecen cuando se resuelven",
        manager,
    )


def test_hypothesis_state_is_tracked():
    manager = ConversationStateManager()

    manager.start_topic(
        "plant_growth",
        "Crecimiento anómalo de plantas",
    )

    hypothesis = manager.add_hypothesis(
        "magnetic_field",
        "El campo magnético acelera el crecimiento.",
        topic_id="plant_growth",
        confidence=0.4,
    )

    assert hypothesis.status == "open"
    assert manager.open_hypotheses()[0].hypothesis_id == "magnetic_field"

    manager.update_hypothesis(
        "magnetic_field",
        status="supported",
        confidence=0.8,
    )

    assert manager.open_hypotheses() == []
    assert hypothesis.status == "supported"
    assert hypothesis.confidence == 0.8

    _print_state(
        "TEST 4 — Las hipótesis conservan estado y confianza",
        manager,
    )


def test_blocked_topic_is_not_presented_as_resolved():
    manager = ConversationStateManager()

    manager.start_topic(
        "year_100_population",
        "Población en el año 100",
    )

    manager.block_topic(
        "year_100_population",
        "Falta una tasa o modelo demográfico.",
    )

    topic = manager.get_topic("year_100_population")

    assert topic.status == "blocked"
    assert manager.should_repeat_topic("year_100_population") is False

    directives = manager.response_directives()
    assert "year_100_population" in directives["blocked_topics"]
    assert directives["do_not_present_blocked_topics_as_resolved"] is True

    _print_state(
        "TEST 5 — Un tema bloqueado no se trata como si estuviera resuelto",
        manager,
    )


def test_syncs_into_cognitive_state():
    manager = ConversationStateManager()

    manager.start_topic(
        "colony_food",
        "Reservas y alimentación",
    )

    manager.start_topic(
        "evacuation_plan",
        "Evacuación",
    )
    manager.resolve_topic(
        "evacuation_plan",
        summary="Plan aprobado.",
    )

    manager.add_pending_question(
        "daily_consumption",
        "¿Cuál es el consumo diario por persona?",
        topic_id="colony_food",
    )

    state = CognitiveState(
        user_query="Sigamos con la colonia."
    )

    manager.sync_to_cognitive_state(state)

    assert state.conversation_state["current_topic_id"] == "colony_food"
    assert (
        state.conversation_state["topics"]["evacuation_plan"]["status"]
        == "resolved"
    )
    assert (
        state.conversation_state["topics"]["colony_food"]["status"]
        == "active"
    )
    assert (
        "daily_consumption"
        in state.conversation_state["pending_questions"]
    )

    meta = state.turn_metadata["conversation_state_manager"]
    assert meta["active_topic_count"] == 1
    assert meta["resolved_topic_count"] == 1
    assert meta["pending_question_count"] == 1

    print("=" * 100)
    print("TEST 6 — ConversationStateManager sincroniza con CognitiveState")
    print("CONVERSATION STATE:", state.conversation_state)
    print("META:", meta)



def test_closing_subtopic_restores_previous_active_topic():
    manager = ConversationStateManager()

    manager.start_topic(
        "colony_food",
        "Reservas y alimentación",
    )
    manager.start_topic(
        "evacuation_plan",
        "Plan de evacuación",
    )

    manager.resolve_topic(
        "evacuation_plan",
        summary="Plan aprobado.",
    )

    assert manager.current_topic_id == "colony_food"

    _print_state(
        "TEST 7 — Al cerrar un subtema vuelve el tema ACTIVE anterior",
        manager,
    )


def test_finds_blocked_topic_from_missing_dependency_language():
    manager = ConversationStateManager()

    manager.start_topic(
        "demography",
        "En el año 100, cuántos habitantes tendrá la colonia si comienza con 100 habitantes",
    )
    manager.block_topic(
        "demography",
        "Falta tasa o modelo demográfico de crecimiento.",
    )

    manager.start_topic(
        "greeting",
        "Hola, ¿cómo estás?",
    )

    match = manager.find_relevant_topic(
        "¿Y si te doy una tasa de crecimiento anual del 2%? "
        "Calcula cuántos habitantes tendría dentro de 100 años.",
        exclude_topic_id="greeting",
    )

    assert match is not None
    assert match.topic_id == "demography"
    assert match.status == "blocked"

    _print_state(
        "TEST 8 — El contenido identifica el tema demográfico bloqueado",
        manager,
    )


def test_reopening_matched_blocked_topic_restores_it_as_current():
    manager = ConversationStateManager()

    manager.start_topic(
        "demography",
        "Población de la colonia en el año 100",
    )
    manager.block_topic(
        "demography",
        "Falta tasa de crecimiento anual.",
    )
    manager.start_topic(
        "greeting",
        "Hola",
    )

    match = manager.find_relevant_topic(
        "La tasa de crecimiento anual será del 2% para la población de la colonia.",
        exclude_topic_id="greeting",
    )

    assert match is not None
    manager.reopen_topic(
        match.topic_id
    )

    assert manager.current_topic_id == "demography"
    assert manager.get_topic("demography").status == "active"
    assert (
        manager.get_topic("demography").metadata["previous_block_reason"]
        == "Falta tasa de crecimiento anual."
    )

    _print_state(
        "TEST 9 — Un tema bloqueado coincidente puede reabrirse como actual",
        manager,
    )


def main():
    test_resolved_topic_is_not_repeated_after_user_approval()
    test_explicit_reopen_allows_returning_to_resolved_topic()
    test_pending_question_lifecycle()
    test_hypothesis_state_is_tracked()
    test_blocked_topic_is_not_presented_as_resolved()
    test_syncs_into_cognitive_state()
    test_closing_subtopic_restores_previous_active_topic()
    test_finds_blocked_topic_from_missing_dependency_language()
    test_reopening_matched_blocked_topic_restores_it_as_current()

    print("=" * 100)
    print("CONVERSATION STATE MANAGER 1.0.2: 9/9 pruebas superadas.")


if __name__ == "__main__":
    main()
