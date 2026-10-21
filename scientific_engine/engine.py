from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from .science_router import ScienceRouter, ScienceRoute
from .tool_registry import ToolRegistry
from .physics.relativity import (
    format_schwarzschild_structure,
    format_schwarzschild_radius,
    schwarzschild_radius_solar_masses,
    solar_masses_to_kg,
    schwarzschild_structure,
)
from .physics.symbolic_math import (
    SYMPY_AVAILABLE,
    simplify_expression,
    solve_equation,
    derivative,
    expression_domain_exclusions,
)
from .physics.units import (
    PINT_AVAILABLE,
    convert as convert_units,
    normalize_unit,
)
from .astronomy.astronomy_tools import (
    ASTROPY_AVAILABLE,
    dependency_status as astronomy_dependency_status,
)


@dataclass
class ScientificResult:
    handled: bool
    domain: Optional[str] = None
    intent: Optional[str] = None
    deterministic: bool = False
    answer: Optional[str] = None
    facts: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.facts is None:
            self.facts = {}
        if self.metadata is None:
            self.metadata = {}

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


class ScientificEngine:
    VERSION = "1.1.0"

    def __init__(self) -> None:
        self.router = ScienceRouter()
        self.registry = ToolRegistry()
        self._register_tools()

    def _register_tools(self) -> None:
        self.registry.register(
            "physics.schwarzschild_structure",
            domain="physics",
            callable=schwarzschild_structure,
            description="Estructura causal Schwarzschild: horizonte vs singularidad.",
        )
        self.registry.register(
            "physics.schwarzschild_radius",
            domain="physics",
            callable=self._schwarzschild_radius_tool,
            description="Calcula r_s = 2GM/c^2.",
        )
        self.registry.register(
            "math.simplify",
            domain="mathematics",
            callable=simplify_expression,
            description="Simplificación simbólica con SymPy.",
            available=SYMPY_AVAILABLE,
            dependency="sympy",
        )
        self.registry.register(
            "math.solve",
            domain="mathematics",
            callable=solve_equation,
            description="Resolución simbólica de ecuaciones con SymPy.",
            available=SYMPY_AVAILABLE,
            dependency="sympy",
        )
        self.registry.register(
            "math.derivative",
            domain="mathematics",
            callable=derivative,
            description="Derivación simbólica con SymPy.",
            available=SYMPY_AVAILABLE,
            dependency="sympy",
        )
        # Pint es preferido, pero units.convert dispone de un fallback SI
        # determinista para unidades comunes si Pint no está instalado.
        self.registry.register(
            "units.convert",
            domain="units",
            callable=convert_units,
            description="Conversión dimensional con Pint o fallback SI determinista.",
            available=True,
            dependency="pint (opcional)",
        )

    @staticmethod
    def _schwarzschild_radius_tool(
        *,
        mass_kg: Optional[float] = None,
        solar_masses: Optional[float] = None,
    ) -> Dict[str, float]:
        if mass_kg is None and solar_masses is None:
            raise ValueError("Se requiere mass_kg o solar_masses.")

        if mass_kg is None:
            mass_kg = solar_masses_to_kg(float(solar_masses))

        radius_m = (
            schwarzschild_radius_solar_masses(float(solar_masses))
            if solar_masses is not None
            else __import__(
                "scientific_engine.physics.relativity",
                fromlist=["schwarzschild_radius"],
            ).schwarzschild_radius(float(mass_kg))
        )

        return {
            "mass_kg": float(mass_kg),
            "solar_masses": (
                float(solar_masses)
                if solar_masses is not None
                else float(mass_kg) / solar_masses_to_kg(1.0)
            ),
            "radius_m": float(radius_m),
            "radius_km": float(radius_m) / 1000.0,
        }

    @staticmethod
    def _format_number(value: float) -> str:
        number = float(value)
        if number == 0.0:
            return "0"
        if abs(number) >= 1e9 or abs(number) < 1e-6:
            return f"{number:.8g}"
        rounded = round(number)
        if abs(number - rounded) <= max(1e-12, abs(number) * 1e-12):
            return str(int(rounded))
        return f"{number:.10g}"

    def _tool_failure(self, route: ScienceRoute, exc: Exception) -> ScientificResult:
        """
        Una ruta científica determinista reconocida no debe degradarse a una
        respuesta inventada por el LLM si la herramienta falla. El fallo de la
        herramienta se devuelve de forma explícita y trazable.
        """
        return ScientificResult(
            handled=True,
            domain=route.domain,
            intent=route.intent,
            deterministic=True,
            answer=(
                "No pude completar el cálculo científico determinista: "
                f"{type(exc).__name__}: {exc}"
            ),
            facts={},
            metadata={
                "engine_version": self.VERSION,
                "router_version": self.router.VERSION,
                "route": route.snapshot(),
                "tool": route.tool_name,
                "tool_error": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )

    @staticmethod
    def _split_compound_science(user_text: str) -> list[str]:
        """Divide solo instrucciones científicas explícitas; no divide lenguaje natural arbitrario."""
        import re

        text = str(user_text or "").strip()
        action = (
            r"(?:resuelve|resolver|soluciona|solve|"
            r"calcula\s+(?:la\s+)?derivada|deriva|derivar|differentiate|"
            r"convierte|convertir|convert|simplifica|simplificar|simplify)"
        )
        if not re.match(rf"^(?:{action})\b", text, flags=re.I):
            return []

        parts = re.split(
            rf"(?:,\s*|;\s*|\s+y\s+)(?={action}\b)".replace(r"\s", r"\s"),
            text,
            flags=re.I,
        )
        if len(parts) < 2 or len(parts) > 6:
            return []
        parts[-1] = re.sub(
            r"(?:[.;]\s*|\s+)(?:explica(?:\s+brevemente)?\s+(?:cada|los)\s+"
            r"resultado(?:s)?|explica\s+brevemente\s+cada\s+resultado)\.?\s*$".replace(r"\s", r"\s"),
            "", parts[-1], flags=re.I,
        ).strip()
        parts = [part.strip() for part in parts]
        if not all(parts):
            return []
        return parts

    def handle(self, user_text: str) -> ScientificResult:
        subtasks = self._split_compound_science(user_text)
        if subtasks:
            routes = [self.router.route(part) for part in subtasks]
            if all(route.matched and route.deterministic for route in routes):
                results = []
                for route in routes:
                    try:
                        results.append(self._execute_route(route))
                    except Exception as exc:
                        results.append(self._tool_failure(route, exc))
                answers = [
                    f"{index}. {result.answer}"
                    for index, result in enumerate(results, 1)
                ]
                return ScientificResult(
                    handled=True,
                    domain="multi_science",
                    intent="multi_operation",
                    deterministic=True,
                    answer="\n".join(answers).replace(r"\n", "\n"),
                    facts={"tasks": [result.snapshot() for result in results]},
                    metadata={
                        "engine_version": self.VERSION,
                        "router_version": self.router.VERSION,
                        "tool": "scientific_engine.multi_operation",
                        "tool_error": any(bool(r.metadata.get("tool_error")) for r in results),
                        "count": len(results),
                    },
                )
            # Una solicitud mixta no debe llegar entera a math.solve.
            return ScientificResult(
                handled=True, domain="multi_science", intent="multi_operation",
                deterministic=True,
                answer="No pude analizar todas las operaciones científicas por separado; reformula las instrucciones no reconocidas.",
                metadata={"engine_version": self.VERSION, "tool_error": True,
                          "unrecognized": [text for text, route in zip(subtasks, routes) if not route.matched]},
            )

        route = self.router.route(user_text)

        if not route.matched:
            return ScientificResult(
                handled=False,
                metadata={
                    "engine_version": self.VERSION,
                    "router_version": self.router.VERSION,
                    "route": route.snapshot(),
                },
            )

        try:
            return self._execute_route(route)
        except Exception as exc:
            return self._tool_failure(route, exc)

    def _execute_route(self, route: ScienceRoute) -> ScientificResult:
        if not route.tool_name:
            return ScientificResult(
                handled=False,
                domain=route.domain,
                intent=route.intent,
                metadata={
                    "engine_version": self.VERSION,
                    "router_version": self.router.VERSION,
                    "route": route.snapshot(),
                    "reason": "matched_route_without_tool",
                },
            )

        if route.intent == "schwarzschild_structure":
            structure = self.registry.execute(route.tool_name).snapshot()
            return self._success(
                route,
                answer=format_schwarzschild_structure(),
                facts=structure,
            )

        if route.intent == "schwarzschild_radius":
            values = self.registry.execute(
                route.tool_name,
                **route.parameters,
            )
            return self._success(
                route,
                answer=format_schwarzschild_radius(
                    mass_kg=values["mass_kg"]
                ),
                facts=values,
            )

        if route.intent == "solve_equation":
            solutions = self.registry.execute(
                route.tool_name,
                **route.parameters,
            )
            variable = str(route.parameters.get("variable") or "x")
            if solutions:
                rendered = ", ".join(
                    f"{variable} = {value}" for value in solutions
                )
                answer = f"Soluciones: {rendered}."
            else:
                answer = f"No se encontraron soluciones para {variable}."
            return self._success(
                route,
                answer=answer,
                facts={
                    "equation": route.parameters.get("equation"),
                    "variable": variable,
                    "solutions": list(solutions),
                },
            )

        if route.intent == "simplify_expression":
            simplified = self.registry.execute(
                route.tool_name,
                **route.parameters,
            )
            expression = str(route.parameters.get("expression") or "")
            exclusions = expression_domain_exclusions(expression)
            answer = f"La expresión se simplifica a: {simplified}."
            if exclusions:
                rendered = ", ".join(
                    item.replace("!=", "≠") for item in exclusions
                )
                answer += f" Restricción del dominio original: {rendered}."
            return self._success(
                route,
                answer=answer,
                facts={
                    "expression": expression,
                    "simplified": simplified,
                    "domain_exclusions": exclusions,
                },
            )

        if route.intent == "derivative":
            result = self.registry.execute(
                route.tool_name,
                **route.parameters,
            )
            variable = str(route.parameters.get("variable") or "x")
            expression = str(route.parameters.get("expression") or "")
            return self._success(
                route,
                answer=f"La derivada respecto a {variable} es: {result}.",
                facts={
                    "expression": expression,
                    "variable": variable,
                    "derivative": result,
                },
            )

        if route.intent == "unit_conversion":
            value = float(route.parameters["value"])
            source = str(route.parameters["from_unit"])
            target = str(route.parameters["to_unit"])
            converted = float(
                self.registry.execute(
                    route.tool_name,
                    value=value,
                    from_unit=source,
                    to_unit=target,
                )
            )
            source_normalized = normalize_unit(source)
            target_normalized = normalize_unit(target)
            answer = (
                f"{self._format_number(value)} {source_normalized} = "
                f"{self._format_number(converted)} {target_normalized}."
            )
            return self._success(
                route,
                answer=answer,
                facts={
                    "input_value": value,
                    "from_unit": source_normalized,
                    "output_value": converted,
                    "to_unit": target_normalized,
                },
            )

        return ScientificResult(
            handled=False,
            domain=route.domain,
            intent=route.intent,
            metadata={
                "engine_version": self.VERSION,
                "router_version": self.router.VERSION,
                "route": route.snapshot(),
                "reason": "unsupported_scientific_intent",
            },
        )

    def _success(
        self,
        route: ScienceRoute,
        *,
        answer: str,
        facts: Dict[str, Any],
    ) -> ScientificResult:
        return ScientificResult(
            handled=True,
            domain=route.domain,
            intent=route.intent,
            deterministic=True,
            answer=str(answer),
            facts=dict(facts or {}),
            metadata={
                "engine_version": self.VERSION,
                "router_version": self.router.VERSION,
                "registry_version": self.registry.VERSION,
                "route": route.snapshot(),
                "tool": route.tool_name,
                "tool_error": False,
            },
        )

    def status(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "router_version": self.router.VERSION,
            "registry_version": self.registry.VERSION,
            "sympy": SYMPY_AVAILABLE,
            "pint": PINT_AVAILABLE,
            "astropy": ASTROPY_AVAILABLE,
            "astronomy": astronomy_dependency_status(),
            "tools": self.registry.list_tools(),
        }
