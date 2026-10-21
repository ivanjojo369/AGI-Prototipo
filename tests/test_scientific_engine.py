from __future__ import annotations

from scientific_engine import ScientificEngine
from scientific_engine.physics.relativity import (
    SOLAR_MASS_KG,
    schwarzschild_radius,
)
from scientific_engine.physics.symbolic_math import (
    SYMPY_AVAILABLE,
    solve_equation,
)


def test_1_router_recognizes_schwarzschild_structure():
    engine = ScientificEngine()
    route = engine.router.route(
        "¿Qué es un horizonte de sucesos y dónde está respecto a "
        "la singularidad en un agujero negro de Schwarzschild?"
    )

    assert route.matched is True
    assert route.intent == "schwarzschild_structure"
    assert route.deterministic is True


def test_2_structure_answer_is_deterministic_and_correct():
    engine = ScientificEngine()
    result = engine.handle(
        "¿Qué es un horizonte de sucesos y dónde está respecto "
        "a la singularidad en Schwarzschild?"
    )

    assert result.handled is True
    assert result.deterministic is True
    assert "r_s = 2GM/c^2" in result.answer
    assert "r = 0" in result.answer
    assert "regiones distintas" in result.answer
    assert result.facts["event_horizon_is_singularity"] is False


def test_3_one_solar_mass_radius_is_about_2_95_km():
    radius_m = schwarzschild_radius(SOLAR_MASS_KG)
    assert 2900.0 < radius_m < 3000.0


def test_4_natural_language_mass_routes_to_calculator():
    engine = ScientificEngine()
    result = engine.handle(
        "Calcula el radio de Schwarzschild de un agujero negro "
        "de 10 masas solares."
    )

    assert result.handled is True
    assert result.intent == "schwarzschild_radius"
    assert 29.0 < result.facts["radius_km"] < 30.5


def test_5_general_chat_is_not_hijacked():
    engine = ScientificEngine()
    result = engine.handle("Hola, ¿cómo estás?")
    assert result.handled is False


def test_6_registry_reports_scientific_tools():
    engine = ScientificEngine()
    names = {row["name"] for row in engine.status()["tools"]}

    assert "physics.schwarzschild_structure" in names
    assert "physics.schwarzschild_radius" in names
    assert "math.solve" in names
    assert "math.simplify" in names
    assert "math.derivative" in names
    assert "units.convert" in names


def test_7_sympy_solver_works_when_available():
    if not SYMPY_AVAILABLE:
        return

    roots = set(solve_equation("x**2 - 4 = 0", "x"))
    assert roots == {"-2", "2"}


def test_8_natural_language_equation_is_solved():
    engine = ScientificEngine()
    result = engine.handle("Resuelve x² - 4 = 0")

    assert result.handled is True
    assert result.deterministic is True
    assert result.intent == "solve_equation"
    assert set(result.facts["solutions"]) == {"-2", "2"}
    assert "x = -2" in result.answer
    assert "x = 2" in result.answer


def test_9_simplification_preserves_original_domain_restriction():
    engine = ScientificEngine()
    result = engine.handle("Simplifica (x² - 1)/(x - 1)")

    assert result.handled is True
    assert result.intent == "simplify_expression"
    assert result.facts["simplified"] == "x + 1"
    assert "x != 1" in result.facts["domain_exclusions"]
    assert "x ≠ 1" in result.answer


def test_10_derivative_accepts_unicode_and_implicit_multiplication():
    engine = ScientificEngine()
    result = engine.handle("Deriva x³ + 2x")

    assert result.handled is True
    assert result.intent == "derivative"
    assert result.facts["variable"] == "x"
    assert result.facts["derivative"] == "3*x**2 + 2"


def test_11_spanish_unit_conversion_is_deterministic():
    engine = ScientificEngine()
    result = engine.handle("Convierte 5 kilómetros a metros")

    assert result.handled is True
    assert result.intent == "unit_conversion"
    assert result.facts["output_value"] == 5000.0
    assert result.facts["to_unit"] == "meter"


def test_12_speed_unit_conversion_works_without_llm():
    engine = ScientificEngine()
    result = engine.handle("Convierte 72 km/h a m/s")

    assert result.handled is True
    assert result.intent == "unit_conversion"
    assert abs(result.facts["output_value"] - 20.0) < 1e-9


def test_13_router_does_not_treat_plain_arithmetic_as_symbolic_science():
    engine = ScientificEngine()
    result = engine.handle("Calcula 2 + 2")

    # Aritmética general sigue perteneciendo al Orchestrator/Calculator.
    assert result.handled is False


def test_14_versions_are_1_1():
    engine = ScientificEngine()
    status = engine.status()

    assert status["version"] == "1.1.0"
    assert status["router_version"] == "1.1.0"
    assert status["registry_version"] == "1.1.0"


def main():
    tests = [
        test_1_router_recognizes_schwarzschild_structure,
        test_2_structure_answer_is_deterministic_and_correct,
        test_3_one_solar_mass_radius_is_about_2_95_km,
        test_4_natural_language_mass_routes_to_calculator,
        test_5_general_chat_is_not_hijacked,
        test_6_registry_reports_scientific_tools,
        test_7_sympy_solver_works_when_available,
        test_8_natural_language_equation_is_solved,
        test_9_simplification_preserves_original_domain_restriction,
        test_10_derivative_accepts_unicode_and_implicit_multiplication,
        test_11_spanish_unit_conversion_is_deterministic,
        test_12_speed_unit_conversion_works_without_llm,
        test_13_router_does_not_treat_plain_arithmetic_as_symbolic_science,
        test_14_versions_are_1_1,
    ]

    for index, test in enumerate(tests, 1):
        test()
        print(f"TEST {index:02d} — OK — {test.__name__}")

    print("=" * 100)
    print("QUIPU SCIENTIFIC ENGINE 1.1.0: 14/14 pruebas superadas.")


if __name__ == "__main__":
    main()
