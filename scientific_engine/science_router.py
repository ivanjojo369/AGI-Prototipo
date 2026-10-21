from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import re
import unicodedata


def _normalize(text: str) -> str:
    value = str(text or "").strip().lower()
    superscript_digits = str.maketrans({"⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9"})
    value = re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+",
        lambda m: "^" + m.group(0).translate(superscript_digits),
        value,
    )
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace("×", "*").replace("·", "*").replace("÷", "/")
    return re.sub(r"\s+", " ", value)


def _strip_terminal_punctuation(text: str) -> str:
    return str(text or "").strip().strip(" ¿?¡!;:")


def _infer_variable_from_expression(expression: str, default: str = "x") -> str:
    # Detecta símbolos de una letra sin confundirlos con nombres de funciones.
    names = re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", expression)
    ignored = {"e", "i"}
    for name in names:
        if name.lower() not in ignored:
            return name
    return default


@dataclass
class ScienceRoute:
    matched: bool
    domain: Optional[str] = None
    intent: Optional[str] = None
    confidence: float = 0.0
    deterministic: bool = False
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.parameters is None:
            self.parameters = {}

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


class ScienceRouter:
    VERSION = "1.1.0"

    BLACK_HOLE_MARKERS = (
        "agujero negro",
        "black hole",
        "schwarzschild",
        "horizonte de sucesos",
        "event horizon",
        "singularidad",
        "singularity",
    )

    def route(self, user_text: str) -> ScienceRoute:
        text = _normalize(user_text)

        if not text:
            return ScienceRoute(False, reason="empty_query")

        # 1) Física determinista conocida: prioridad máxima para evitar que una
        # expresión numérica dentro de la pregunta se desvíe al cálculo genérico.
        route = self._route_black_hole(text)
        if route is not None:
            return route

        # 2) Conversión dimensional explícita.
        route = self._route_unit_conversion(text)
        if route is not None:
            return route

        # 3) Matemática simbólica explícita. No capturamos aritmética general
        # porque esa capacidad ya pertenece al Orchestrator de Quipu.
        route = self._route_derivative(text)
        if route is not None:
            return route

        route = self._route_solve(text)
        if route is not None:
            return route

        route = self._route_simplify(text)
        if route is not None:
            return route

        return ScienceRoute(False, reason="no_scientific_tool_route")

    def _route_black_hole(self, text: str) -> Optional[ScienceRoute]:
        if not any(marker in text for marker in self.BLACK_HOLE_MARKERS):
            return None

        mass = self._extract_black_hole_mass(text)

        if (
            mass is not None
            and any(
                marker in text
                for marker in (
                    "radio de schwarzschild",
                    "schwarzschild radius",
                    "radio del horizonte",
                    "horizon radius",
                )
            )
        ):
            return ScienceRoute(
                True,
                domain="physics",
                intent="schwarzschild_radius",
                confidence=0.99,
                deterministic=True,
                tool_name="physics.schwarzschild_radius",
                parameters=mass,
                reason="black_hole_mass_and_radius_request",
            )

        if any(
            marker in text
            for marker in (
                "horizonte de sucesos",
                "event horizon",
                "singularidad",
                "singularity",
                "schwarzschild",
            )
        ):
            return ScienceRoute(
                True,
                domain="physics",
                intent="schwarzschild_structure",
                confidence=0.98,
                deterministic=True,
                tool_name="physics.schwarzschild_structure",
                reason="schwarzschild_structure_question",
            )

        return None

    @staticmethod
    def _route_unit_conversion(text: str) -> Optional[ScienceRoute]:
        patterns = (
            r"^(?:convierte|convertir|convert|pasa|transforma)\s+"
            r"([+-]?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?)\s+(.+?)\s+"
            r"(?:a|en|to)\s+(.+?)$",
            r"^(?:cuanto|cuantos|cuanto es|cuanto son|how much is)\s+"
            r"([+-]?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?)\s+(.+?)\s+"
            r"(?:en|a|to)\s+(.+?)$",
        )

        for pattern in patterns:
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            value_text, source, target = match.groups()
            source = _strip_terminal_punctuation(source)
            target = _strip_terminal_punctuation(target)
            if not source or not target:
                continue
            return ScienceRoute(
                True,
                domain="units",
                intent="unit_conversion",
                confidence=0.99,
                deterministic=True,
                tool_name="units.convert",
                parameters={
                    "value": float(value_text.replace(",", ".")),
                    "from_unit": source,
                    "to_unit": target,
                },
                reason="explicit_unit_conversion",
            )

        return None

    @staticmethod
    def _split_explicit_variable(payload: str) -> tuple[str, Optional[str]]:
        match = re.search(
            r"\s+(?:respecto\s+a|con\s+respecto\s+a|para|with\s+respect\s+to)\s+"
            r"([A-Za-z][A-Za-z0-9]*)\s*$",
            payload,
            flags=re.IGNORECASE,
        )
        if not match:
            return payload.strip(), None
        variable = match.group(1)
        return payload[:match.start()].strip(), variable

    @classmethod
    def _route_derivative(cls, text: str) -> Optional[ScienceRoute]:
        patterns = (
            r"^(?:deriva|derivar|differentiate)\s+(.+)$",
            r"^(?:calcula\s+)?(?:la\s+)?derivada\s+de\s+(.+)$",
            r"^derivative\s+of\s+(.+)$",
        )
        for pattern in patterns:
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            expression, explicit_variable = cls._split_explicit_variable(
                _strip_terminal_punctuation(match.group(1))
            )
            if not expression:
                continue
            variable = explicit_variable or _infer_variable_from_expression(expression)
            return ScienceRoute(
                True,
                domain="mathematics",
                intent="derivative",
                confidence=0.99,
                deterministic=True,
                tool_name="math.derivative",
                parameters={"expression": expression, "variable": variable},
                reason="explicit_symbolic_derivative",
            )
        return None

    @classmethod
    def _route_solve(cls, text: str) -> Optional[ScienceRoute]:
        patterns = (
            r"^(?:resuelve|resolver|soluciona|solve)\s+(?:la\s+ecuacion\s+)?(.+)$",
            r"^(?:encuentra\s+las?\s+soluciones?\s+de)\s+(.+)$",
        )
        for pattern in patterns:
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            equation, explicit_variable = cls._split_explicit_variable(
                _strip_terminal_punctuation(match.group(1))
            )
            if not equation:
                continue
            variable = explicit_variable or _infer_variable_from_expression(equation)
            return ScienceRoute(
                True,
                domain="mathematics",
                intent="solve_equation",
                confidence=0.99,
                deterministic=True,
                tool_name="math.solve",
                parameters={"equation": equation, "variable": variable},
                reason="explicit_symbolic_equation",
            )
        return None

    @staticmethod
    def _route_simplify(text: str) -> Optional[ScienceRoute]:
        patterns = (
            r"^(?:simplifica|simplificar|simplify)\s+(.+)$",
            r"^(?:reduce\s+la\s+expresion)\s+(.+)$",
        )
        for pattern in patterns:
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            expression = _strip_terminal_punctuation(match.group(1))
            if not expression:
                continue
            return ScienceRoute(
                True,
                domain="mathematics",
                intent="simplify_expression",
                confidence=0.99,
                deterministic=True,
                tool_name="math.simplify",
                parameters={"expression": expression},
                reason="explicit_symbolic_simplification",
            )
        return None

    @staticmethod
    def _extract_black_hole_mass(text: str) -> Optional[Dict[str, float]]:
        solar = re.search(
            r"([+-]?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?)\s*"
            r"(?:masas?\s+solares?|m_sun|msun)",
            text,
            flags=re.IGNORECASE,
        )
        if solar:
            return {
                "solar_masses": float(solar.group(1).replace(",", ".")),
            }

        kg = re.search(
            r"([+-]?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?)\s*kg\b",
            text,
            flags=re.IGNORECASE,
        )
        if kg:
            return {
                "mass_kg": float(kg.group(1).replace(",", ".")),
            }

        return None
