from cognition.cognitive_state import CognitiveState
from cognition.self_correction import SelfCorrectionLoop


def _print_plan(title, plan):
    print("=" * 100)
    print(title)
    print("CAN CONTINUE:", plan.can_continue)
    print("ACTION:", plan.action)
    print("ATTEMPT:", plan.attempt)
    print("ISSUES:", plan.issues)
    print("PREMISES:", plan.premises)
    print("MISSING:", plan.missing_information)
    print("CONFLICTS:", plan.conflicts)
    print("MUST RECOMPUTE:", plan.must_recompute)
    print("REUSE PREVIOUS:", plan.reuse_previous_answer)
    print("REASON:", plan.reason)


def _print_outcome(title, outcome):
    print("=" * 100)
    print(title)
    print("ACCEPTED:", outcome.accepted)
    print("ACTION:", outcome.action)
    print("ATTEMPT:", outcome.attempt)
    print("VERIFIED:", outcome.verification_passed)
    print("ANSWER:", outcome.answer)
    print("REASON:", outcome.reason)


def test_begin_discards_old_answer_and_rebuilds_from_premises():
    state = CognitiveState(
        user_query="Revisa la población del año 100."
    )
    state.world_state = {
        "colony.population.initial": 100,
        "colony.year.initial": 1,
        "colony.year.current": 100,
        "colony.population.growth_rate": 0.02,
    }
    state.draft_answer = "En el año 100 habrá 2.5 millones."

    loop = SelfCorrectionLoop(max_attempts=2)

    plan = loop.begin(
        state,
        issues=["La conclusión anterior no estaba justificada."],
        required_keys=[
            "colony.population.initial",
            "colony.year.initial",
            "colony.year.current",
            "colony.population.growth_rate",
        ],
    )

    assert plan.can_continue is True
    assert plan.action == "recompute_from_premises"
    assert plan.must_recompute is True
    assert plan.reuse_previous_answer is False
    assert state.correction_attempts == 1
    assert state.draft_answer is None

    context = loop.build_regeneration_context(plan)
    assert "previous_answer" not in context
    assert "2.5 millones" not in str(context)
    assert context["premises"]["colony.population.initial"] == 100

    history = state.turn_metadata["self_correction"]["history"]
    assert history[0]["previous_answer"] == "En el año 100 habrá 2.5 millones."

    _print_plan(
        "TEST 1 — Corregir significa recomputar desde premisas, no parchar la respuesta vieja",
        plan,
    )


def test_missing_information_blocks_regeneration():
    state = CognitiveState(
        user_query="Corrige la proyección demográfica."
    )
    state.world_state = {
        "colony.population.initial": 100,
    }
    state.missing_information = [
        "Falta información requerida: colony.population.growth_rate"
    ]
    state.draft_answer = "2.5 millones"

    loop = SelfCorrectionLoop()

    plan = loop.begin(
        state,
        issues=["Falta modelo demográfico."],
    )

    assert plan.can_continue is False
    assert plan.action == "ask_for_data"
    assert state.correction_attempts == 0
    assert state.draft_answer == "2.5 millones"

    _print_plan(
        "TEST 2 — Si faltan datos, no se regenera inventando información",
        plan,
    )


def test_active_conflict_blocks_regeneration():
    state = CognitiveState(
        user_query="Corrige la descripción del clima."
    )
    state.world_state = {
        "planet.climate.dominant": "calido humedo",
    }
    state.conflicts = [
        "Conflicto activo en planet.climate.dominant: 'desertico' source=model"
    ]
    state.draft_answer = "El clima es desértico."

    loop = SelfCorrectionLoop()

    plan = loop.begin(
        state,
        issues=["Existe una contradicción sobre el clima."],
    )

    assert plan.can_continue is False
    assert plan.action == "resolve_conflict"
    assert state.correction_attempts == 0

    _print_plan(
        "TEST 3 — Un conflicto activo debe resolverse antes de autocorregir",
        plan,
    )


def test_verified_candidate_replaces_old_conclusion():
    state = CognitiveState(
        user_query="Revisa el cálculo."
    )
    state.world_state = {
        "colony.food_reserves_kg": 30000,
    }
    state.draft_answer = "Quedan 40000 kg."

    loop = SelfCorrectionLoop(max_attempts=2)

    plan = loop.begin(
        state,
        issues=["No se aplicó la pérdida del 25%."],
        required_keys=["colony.food_reserves_kg"],
    )
    assert plan.can_continue is True

    outcome = loop.submit_candidate(
        state,
        "Después de la pérdida quedan 30000 kg.",
        {"passed": True, "checks": ["math", "world_state"]},
    )

    assert outcome.accepted is True
    assert outcome.action == "accept"
    assert state.draft_answer == "Después de la pérdida quedan 30000 kg."
    assert state.verification_result["passed"] is True

    _print_outcome(
        "TEST 4 — Una candidata corregida solo sustituye a la anterior después de verificar",
        outcome,
    )


def test_failed_candidate_requests_retry_and_is_not_exposed_as_valid():
    state = CognitiveState(
        user_query="Corrige el cálculo."
    )
    state.world_state = {
        "colony.food_reserves_kg": 30000,
    }
    state.draft_answer = "Quedan 40000 kg."

    loop = SelfCorrectionLoop(max_attempts=2)

    loop.begin(
        state,
        issues=["Cálculo incorrecto."],
        required_keys=["colony.food_reserves_kg"],
    )

    outcome = loop.submit_candidate(
        state,
        "Después de la pérdida quedan 25000 kg.",
        {"passed": False, "reason": "Resultado matemático incorrecto."},
    )

    assert outcome.accepted is False
    assert outcome.action == "retry"
    assert state.draft_answer is None

    failed = state.turn_metadata["self_correction"]["failed_candidates"]
    assert failed[0]["candidate_answer"] == "Después de la pérdida quedan 25000 kg."

    _print_outcome(
        "TEST 5 — Una corrección fallida no se presenta como si estuviera arreglada",
        outcome,
    )


def test_max_attempts_stops_infinite_correction_loop():
    state = CognitiveState(
        user_query="Corrige otra vez."
    )
    state.world_state = {
        "test.value": 42,
    }
    state.draft_answer = "41"

    loop = SelfCorrectionLoop(max_attempts=1)

    first = loop.begin(
        state,
        issues=["Valor incorrecto."],
        required_keys=["test.value"],
    )
    assert first.can_continue is True
    assert state.correction_attempts == 1

    outcome = loop.submit_candidate(
        state,
        "43",
        {"passed": False},
    )
    assert outcome.action == "stop"
    assert outcome.accepted is False

    second = loop.begin(
        state,
        issues=["Sigue incorrecto."],
        required_keys=["test.value"],
    )
    assert second.can_continue is False
    assert second.action == "stop"
    assert state.correction_attempts == 1

    _print_outcome(
        "TEST 6 — El loop se detiene al alcanzar el máximo de intentos",
        outcome,
    )


def main():
    test_begin_discards_old_answer_and_rebuilds_from_premises()
    test_missing_information_blocks_regeneration()
    test_active_conflict_blocks_regeneration()
    test_verified_candidate_replaces_old_conclusion()
    test_failed_candidate_requests_retry_and_is_not_exposed_as_valid()
    test_max_attempts_stops_infinite_correction_loop()

    print("=" * 100)
    print("SELF CORRECTION LOOP 1.0: 6/6 pruebas superadas.")


if __name__ == "__main__":
    main()
