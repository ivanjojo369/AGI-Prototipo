from cognition.cognitive_state import CognitiveState, MemoryCandidate
from cognition.long_range_retriever import LongRangeRetriever
from cognition.working_world_state import WorkingWorldState


def _print_result(title, result):
    print("=" * 100)
    print(title)

    print("ACCEPTED:")
    for item in result.accepted_memories:
        print("  +", item.content)

    print("REJECTED:")
    for item in result.rejected_memories:
        print("  -", item.content)

    print("RECONSTRUCTED STATE:")
    for key, value in sorted(result.reconstructed_state.items()):
        print(f"  {key} = {value}")

    print("MISSING:", result.missing_keys)
    print("PROVENANCE:", result.provenance)
    print("WARNINGS:", result.warnings)


def _candidate(
    content,
    *,
    memory_type="episodic",
    tags=None,
    entities=None,
    retrieval_score=0.8,
    block_index=None,
    world_facts=None,
):
    metadata = {
        "tags": list(tags or []),
        "entities": list(entities or []),
        "retrieval_score": retrieval_score,
    }

    if block_index is not None:
        metadata["block_index"] = block_index

    if world_facts is not None:
        metadata["world_facts"] = world_facts

    return MemoryCandidate(
        content=content,
        memory_type=memory_type,
        confidence=0.95,
        metadata=metadata,
    )


def test_reconstructs_colony_from_distant_partial_memories():
    retriever = LongRangeRetriever()

    candidates = [
        _candidate(
            "La colonia comenzó con 100 habitantes.",
            tags=["colonia", "habitantes", "poblacion"],
            block_index=10,
            world_facts=[
                {
                    "kind": "set",
                    "key": "colony.population.initial",
                    "value": 100,
                    "source": "user",
                }
            ],
        ),
        _candidate(
            "proyecto = Aster; lenguaje = Python; estado = segunda fase",
            memory_type="semantic",
            tags=["Aster", "Python", "vision artificial"],
            entities=["Aster"],
            retrieval_score=0.99,
            block_index=20,
            world_facts=[
                {
                    "kind": "set",
                    "key": "project.aster.language",
                    "value": "Python",
                    "source": "memory",
                }
            ],
        ),
        _candidate(
            "La colonia tenía 40000 kg de reservas al inicio.",
            tags=["colonia", "reservas", "alimentos", "40000 kg"],
            block_index=40,
            world_facts=[
                {
                    "kind": "set",
                    "key": "colony.food_reserves_kg",
                    "value": 40000,
                    "source": "user",
                }
            ],
        ),
        _candidate(
            "Cada persona consume 0.8 kg de alimento al día en este escenario.",
            tags=["colonia", "alimento", "consumo", "persona", "dia"],
            block_index=65,
            world_facts=[
                {
                    "kind": "set",
                    "key": "colony.food_consumption_kg_per_person_day",
                    "value": 0.8,
                    "source": "user",
                }
            ],
        ),
        _candidate(
            "Antes de la primera cosecha se perdió el 25% de las reservas.",
            tags=["colonia", "reservas", "alimento", "25%", "averia"],
            block_index=90,
            world_facts=[
                {
                    "kind": "numeric_update",
                    "key": "colony.food_reserves_kg",
                    "operation": "percent_loss",
                    "amount": 25,
                    "source": "user",
                }
            ],
        ),
        _candidate(
            {
                "objetivo": "migrar Quipu a la workstation nueva",
                "estado": "pendiente",
            },
            memory_type="prospective",
            tags=["Quipu", "workstation", "migracion", "Memory 3.0"],
            entities=["Quipu"],
            retrieval_score=0.99,
            block_index=120,
            world_facts=[
                {
                    "kind": "set",
                    "key": "quipu.goal",
                    "value": "migrar a workstation",
                    "source": "memory",
                }
            ],
        ),
    ]

    result = retriever.reconstruct(
        (
            "Con toda la información anterior, ¿las reservas actuales "
            "alcanzan para alimentar a la colonia durante el primer año?"
        ),
        candidates,
        required_keys=[
            "colony.population.initial",
            "colony.food_reserves_kg",
            "colony.food_consumption_kg_per_person_day",
        ],
    )

    assert result.reconstructed_state["colony.population.initial"] == 100
    assert result.reconstructed_state["colony.food_reserves_kg"] == 30000
    assert (
        result.reconstructed_state[
            "colony.food_consumption_kg_per_person_day"
        ]
        == 0.8
    )

    assert "project.aster.language" not in result.reconstructed_state
    assert "quipu.goal" not in result.reconstructed_state
    assert result.missing_keys == []
    assert result.complete is True

    _print_result(
        "TEST 1 — Reconstrucción longitudinal de la colonia con ruido Aster/workstation",
        result,
    )


def test_applies_events_in_temporal_order():
    retriever = LongRangeRetriever()

    candidates = [
        _candidate(
            "Después recibieron 5000 kg adicionales.",
            tags=["colonia", "reservas", "alimentos", "5000"],
            block_index=90,
            world_facts=[
                {
                    "kind": "numeric_update",
                    "key": "colony.food_reserves_kg",
                    "operation": "add",
                    "amount": 5000,
                    "source": "user",
                }
            ],
        ),
        _candidate(
            "Reservas iniciales: 40000 kg.",
            tags=["colonia", "reservas", "alimentos", "40000"],
            block_index=10,
            world_facts=[
                {
                    "kind": "set",
                    "key": "colony.food_reserves_kg",
                    "value": 40000,
                    "source": "user",
                }
            ],
        ),
        _candidate(
            "Se perdió el 25% de las reservas por una avería.",
            tags=["colonia", "reservas", "alimentos", "25%", "averia"],
            block_index=50,
            world_facts=[
                {
                    "kind": "numeric_update",
                    "key": "colony.food_reserves_kg",
                    "operation": "percent_loss",
                    "amount": 25,
                    "source": "user",
                }
            ],
        ),
    ]

    result = retriever.reconstruct(
        "¿Cuántas reservas de alimentos tiene ahora la colonia?",
        candidates,
        required_keys=["colony.food_reserves_kg"],
    )

    # 40,000 - 25% = 30,000; luego + 5,000 = 35,000.
    assert result.reconstructed_state["colony.food_reserves_kg"] == 35000
    assert result.missing_keys == []

    _print_result(
        "TEST 2 — Los recuerdos se aplican en orden temporal, no en orden de llegada",
        result,
    )


def test_reports_missing_required_information_instead_of_inventing():
    retriever = LongRangeRetriever()

    candidates = [
        _candidate(
            "La colonia comenzó con 100 habitantes.",
            tags=["colonia", "habitantes", "poblacion"],
            block_index=10,
            world_facts=[
                {
                    "kind": "set",
                    "key": "colony.population.initial",
                    "value": 100,
                    "source": "user",
                }
            ],
        ),
        _candidate(
            "La colonia conserva 30000 kg de reservas.",
            tags=["colonia", "reservas", "alimentos"],
            block_index=20,
            world_facts=[
                {
                    "kind": "set",
                    "key": "colony.food_reserves_kg",
                    "value": 30000,
                    "source": "user",
                }
            ],
        ),
    ]

    result = retriever.reconstruct(
        "¿Alcanzan las reservas para alimentar a la colonia todo el año?",
        candidates,
        required_keys=[
            "colony.population.initial",
            "colony.food_reserves_kg",
            "colony.food_consumption_kg_per_person_day",
        ],
    )

    assert result.complete is False
    assert result.missing_keys == [
        "colony.food_consumption_kg_per_person_day"
    ]

    _print_result(
        "TEST 3 — Falta consumo diario: LongRangeRetriever no lo inventa",
        result,
    )


def test_syncs_result_into_cognitive_state():
    retriever = LongRangeRetriever()

    state = CognitiveState(
        user_query=(
            "Con toda la información anterior, reconstruye el estado actual "
            "de las reservas de la colonia."
        )
    )

    state.memory_candidates = [
        _candidate(
            "Había 40000 kg de reservas.",
            tags=["colonia", "reservas", "alimentos"],
            block_index=1,
            world_facts=[
                {
                    "kind": "set",
                    "key": "colony.food_reserves_kg",
                    "value": 40000,
                    "source": "user",
                }
            ],
        ),
        _candidate(
            "Se perdió el 25% de las reservas.",
            tags=["colonia", "reservas", "alimentos", "25%"],
            block_index=2,
            world_facts=[
                {
                    "kind": "numeric_update",
                    "key": "colony.food_reserves_kg",
                    "operation": "percent_loss",
                    "amount": 25,
                    "source": "user",
                }
            ],
        ),
    ]

    result = retriever.reconstruct_into_cognitive_state(
        state,
        required_keys=["colony.food_reserves_kg"],
    )

    assert state.world_state["colony.food_reserves_kg"] == 30000
    assert state.missing_information == []
    assert result.complete is True

    print("=" * 100)
    print("TEST 4 — Resultado sincronizado con CognitiveState")
    print("WORLD STATE:", state.world_state)
    print("MISSING:", state.missing_information)
    print(
        "META:",
        state.turn_metadata.get("long_range_retriever"),
    )


def test_irrelevant_high_score_memory_is_still_rejected():
    retriever = LongRangeRetriever()

    candidates = [
        _candidate(
            "proyecto = Aster; lenguaje = Python; estado = segunda fase",
            memory_type="semantic",
            tags=["Aster", "Python", "vision artificial"],
            entities=["Aster"],
            retrieval_score=1.0,
            block_index=1,
            world_facts=[
                {
                    "kind": "set",
                    "key": "project.aster.language",
                    "value": "Python",
                    "source": "memory",
                }
            ],
        ),
        _candidate(
            "La colonia tiene 100 habitantes.",
            tags=["colonia", "habitantes", "poblacion"],
            retrieval_score=0.55,
            block_index=2,
            world_facts=[
                {
                    "kind": "set",
                    "key": "colony.population.initial",
                    "value": 100,
                    "source": "user",
                }
            ],
        ),
    ]

    result = retriever.reconstruct(
        "¿Cuál era la población inicial de la colonia?",
        candidates,
        required_keys=["colony.population.initial"],
    )

    assert result.reconstructed_state["colony.population.initial"] == 100
    assert "project.aster.language" not in result.reconstructed_state

    rejected = " ".join(str(x.content) for x in result.rejected_memories)
    assert "Aster" in rejected

    _print_result(
        "TEST 5 — Retrieval score alto no permite que Aster secuestre la reconstrucción",
        result,
    )


def main():
    test_reconstructs_colony_from_distant_partial_memories()
    test_applies_events_in_temporal_order()
    test_reports_missing_required_information_instead_of_inventing()
    test_syncs_result_into_cognitive_state()
    test_irrelevant_high_score_memory_is_still_rejected()

    print("=" * 100)
    print("LONG RANGE RETRIEVER 1.0.1: 5/5 pruebas superadas.")


if __name__ == "__main__":
    main()
