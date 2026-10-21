from cognition.cognitive_state import CognitiveState
from cognition.output_guard import OutputGuard


def _print_decision(title, decision):
    print("=" * 100)
    print(title)
    print("ACCEPTED:", decision.accepted)
    print("ACTION:", decision.action)
    print("LANGUAGE:", decision.detected_language)
    print("SANITIZED:", decision.sanitized)
    print("ISSUES:", decision.issues)
    print("REMOVED:", decision.removed_lines)
    print("OUTPUT:", decision.output)
    print("REASON:", decision.reason)


def test_clean_spanish_answer_passes():
    state = CognitiveState(
        user_query="¿Cuántas reservas quedan?"
    )

    guard = OutputGuard()

    decision = guard.inspect(
        state,
        "Después de la pérdida quedan 30000 kg de reservas.",
    )

    assert decision.accepted is True
    assert decision.action == "accept"
    assert state.draft_answer == (
        "Después de la pérdida quedan 30000 kg de reservas."
    )

    _print_decision(
        "TEST 1 — Una respuesta natural en español pasa",
        decision,
    )


def test_portuguese_drift_requires_regeneration():
    state = CognitiveState(
        user_query="¿Cuántos habitantes habrá?"
    )

    guard = OutputGuard()

    decision = guard.inspect(
        state,
        (
            "Com base nos dados fornecidos, não é possível calcular "
            "a população sem mais dados."
        ),
    )

    assert decision.accepted is False
    assert decision.action == "regenerate"
    assert decision.detected_language == "pt"
    assert state.draft_answer is None

    _print_decision(
        "TEST 2 — Deriva clara al portugués obliga a regenerar",
        decision,
    )


def test_isolated_internal_token_requires_regeneration():
    state = CognitiveState(
        user_query="¿Qué ejercicios recomiendas para la memoria?"
    )

    guard = OutputGuard()

    decision = guard.inspect(
        state,
        "ejercicios_memoria",
    )

    assert decision.accepted is False
    assert decision.action == "regenerate"
    assert "internal_token_leak" in decision.issues

    _print_decision(
        "TEST 3 — 'ejercicios_memoria' no puede salir al usuario",
        decision,
    )


def test_raw_aster_metadata_dump_is_blocked():
    state = CognitiveState(
        user_query="¿Cómo defenderíamos la colonia contra asteroides?"
    )

    guard = OutputGuard()

    decision = guard.inspect(
        state,
        (
            "proyecto = Aster\n"
            "lenguaje = Python\n"
            "estado = segunda fase\n"
            "especialidad = visión artificial"
        ),
    )

    assert decision.accepted is False
    assert decision.action == "regenerate"
    assert "raw_internal_metadata" in decision.issues

    _print_decision(
        "TEST 4 — Un volcado Aster key=value queda bloqueado",
        decision,
    )


def test_internal_control_lines_are_removed_when_natural_answer_survives():
    state = CognitiveState(
        user_query="¿Qué pasó con las reservas?"
    )

    guard = OutputGuard()

    decision = guard.inspect(
        state,
        (
            "intent: long_range_reasoning\n"
            "tools: long_range_retriever, verifier\n"
            "Después de la avería quedan 30000 kg de reservas."
        ),
    )

    assert decision.accepted is True
    assert decision.action == "accept_sanitized"
    assert decision.sanitized is True
    assert "intent:" not in decision.output
    assert "tools:" not in decision.output
    assert decision.output == (
        "Después de la avería quedan 30000 kg de reservas."
    )

    _print_decision(
        "TEST 5 — Metadata interna se limpia si queda respuesta natural",
        decision,
    )


def test_too_much_internal_leakage_requires_regeneration():
    state = CognitiveState(
        user_query="Explícame el resultado."
    )

    guard = OutputGuard(
        max_internal_lines=2,
    )

    decision = guard.inspect(
        state,
        (
            "intent: chat\n"
            "route: memory_lookup\n"
            "tools: verifier\n"
            "controller_version: 1.0\n"
            "La respuesta correcta es 42."
        ),
    )

    assert decision.accepted is False
    assert decision.action == "regenerate"
    assert "excessive_internal_leakage" in decision.issues

    _print_decision(
        "TEST 6 — Demasiada filtración interna no se arregla silenciosamente",
        decision,
    )


def test_explicit_raw_request_can_allow_key_value_output():
    state = CognitiveState(
        user_query=(
            "Muéstrame literalmente la metadata exacta de Aster."
        )
    )

    guard = OutputGuard()

    decision = guard.inspect(
        state,
        (
            "proyecto = Aster\n"
            "lenguaje = Python\n"
            "estado = segunda fase"
        ),
    )

    assert decision.accepted is True
    assert decision.action == "accept"

    _print_decision(
        "TEST 7 — Si el usuario pide metadata literal, no se bloquea",
        decision,
    )


def main():
    test_clean_spanish_answer_passes()
    test_portuguese_drift_requires_regeneration()
    test_isolated_internal_token_requires_regeneration()
    test_raw_aster_metadata_dump_is_blocked()
    test_internal_control_lines_are_removed_when_natural_answer_survives()
    test_too_much_internal_leakage_requires_regeneration()
    test_explicit_raw_request_can_allow_key_value_output()

    print("=" * 100)
    print("OUTPUT GUARD 1.0: 7/7 pruebas superadas.")


if __name__ == "__main__":
    main()
