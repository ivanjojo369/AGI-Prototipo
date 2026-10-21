from cognition.cognitive_state import CognitiveState
from cognition.working_world_state import WorkingWorldState


def _print_state(title, world):
    print("=" * 92)
    print(title)
    detailed = world.export_detailed_state()

    print("CURRENT:")
    for key, value in detailed["current"].items():
        print(f"  {key} = {value}")

    print("ACTIVE CONFLICTS:")
    if detailed["active_conflicts"]:
        for key, items in detailed["active_conflicts"].items():
            print(f"  {key}:")
            for item in items:
                print("   -", item["value"], "source=", item["source"])
    else:
        print("  none")


def test_colony_reserves_are_updated_without_destroying_history():
    world = WorkingWorldState()

    world.set_fact(
        "colony.population.initial",
        100,
        source="user",
    )

    world.set_fact(
        "colony.food_reserves_kg",
        40000,
        source="user",
        metadata={"phase": "initial"},
    )

    result = world.apply_numeric_update(
        "colony.food_reserves_kg",
        "percent_loss",
        25,
        source="reasoner",
        metadata={"cause": "averia"},
    )

    assert result.value == 30000
    assert world.get_current("colony.food_reserves_kg") == 30000

    history = world.get_history("colony.food_reserves_kg")
    assert len(history) == 1
    assert history[0].value == 40000

    assert result.derived_from[0] == "colony.food_reserves_kg=40000"

    _print_state(
        "TEST 1 — 40,000 kg - 25% => 30,000 kg, conservando historial",
        world,
    )


def test_authoritative_user_update_supersedes_old_fact():
    world = WorkingWorldState()

    world.set_fact(
        "planet.climate.dominant",
        "frio y seco",
        source="user",
    )

    world.set_fact(
        "planet.climate.dominant",
        "calido humedo",
        source="user",
    )

    assert world.get_current("planet.climate.dominant") == "calido humedo"

    history = world.get_history("planet.climate.dominant")
    assert len(history) == 1
    assert history[0].value == "frio y seco"
    assert history[0].status == "historical"

    _print_state(
        "TEST 2 — Un dato nuevo del usuario reemplaza al anterior sin borrarlo",
        world,
    )


def test_non_authoritative_inference_cannot_silently_override_user_fact():
    world = WorkingWorldState()

    world.set_fact(
        "planet.oceans.salinity",
        "casi sin sales",
        source="user",
    )

    conflict = world.set_fact(
        "planet.oceans.salinity",
        "alta salinidad",
        source="model",
        confidence=0.65,
        authoritative=False,
    )

    assert conflict.status == "conflict"
    assert world.get_current("planet.oceans.salinity") == "casi sin sales"
    assert world.has_conflict("planet.oceans.salinity")

    _print_state(
        "TEST 3 — Una inferencia del modelo no pisa silenciosamente un hecho del usuario",
        world,
    )


def test_unknown_is_explicit_and_not_invented():
    world = WorkingWorldState()

    fact = world.mark_unknown(
        "colony.population.year_100",
        reason="Falta una tasa o modelo demografico.",
    )

    assert fact.status == "unknown"
    assert fact.value is None
    assert world.get_current("colony.population.year_100") is None

    # No debe existir una inferencia inventada.
    assert world.get_inferred("colony.population.year_100") == []

    _print_state(
        "TEST 4 — Un dato faltante queda UNKNOWN en lugar de inventarse",
        world,
    )


def test_world_state_syncs_into_cognitive_state_and_exports_conflicts():
    world = WorkingWorldState()

    world.set_fact(
        "colony.population.initial",
        100,
        source="user",
    )
    world.set_fact(
        "colony.food_reserves_kg",
        40000,
        source="user",
    )
    world.apply_numeric_update(
        "colony.food_reserves_kg",
        "percent_loss",
        25,
        source="reasoner",
    )

    world.set_fact(
        "planet.climate.dominant",
        "calido humedo",
        source="user",
    )
    world.set_fact(
        "planet.climate.dominant",
        "desertico",
        source="model",
        authoritative=False,
    )

    state = CognitiveState(
        user_query="¿Alcanzan las reservas actuales para la colonia?"
    )

    world.sync_to_cognitive_state(state)

    assert state.world_state["colony.population.initial"] == 100
    assert state.world_state["colony.food_reserves_kg"] == 30000
    assert state.world_state["planet.climate.dominant"] == "calido humedo"
    assert any(
        "planet.climate.dominant" in conflict
        for conflict in state.conflicts
    )

    print("=" * 92)
    print("TEST 5 — WorkingWorldState se sincroniza con CognitiveState")
    print("COGNITIVE STATE WORLD:")
    for key, value in state.world_state.items():
        print(f"  {key} = {value}")
    print("CONFLICTS:", state.conflicts)


def main():
    test_colony_reserves_are_updated_without_destroying_history()
    test_authoritative_user_update_supersedes_old_fact()
    test_non_authoritative_inference_cannot_silently_override_user_fact()
    test_unknown_is_explicit_and_not_invented()
    test_world_state_syncs_into_cognitive_state_and_exports_conflicts()

    print("=" * 92)
    print("WORKING WORLD STATE 1.0: 5/5 pruebas superadas.")


if __name__ == "__main__":
    main()
