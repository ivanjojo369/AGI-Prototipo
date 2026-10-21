from __future__ import annotations

import re
from typing import Iterable

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
    SYMPY_AVAILABLE = True
except Exception:
    sp = None
    parse_expr = None
    standard_transformations = ()
    implicit_multiplication_application = None
    convert_xor = None
    SYMPY_AVAILABLE = False


_TRANSFORMATIONS = (
    standard_transformations
    + ((implicit_multiplication_application,) if implicit_multiplication_application else ())
    + ((convert_xor,) if convert_xor else ())
)

_SUPERSCRIPT_DIGITS = str.maketrans({
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
})


_ALLOWED_EXPRESSION = re.compile(r"^[0-9A-Za-z+\-*/^().,=\s]+$")


def _require_sympy() -> None:
    if not SYMPY_AVAILABLE:
        raise RuntimeError("SymPy no está instalado.")


def normalize_expression(expression: str) -> str:
    """Normaliza notación matemática común sin intentar interpretar lenguaje natural."""
    value = str(expression or "").strip()
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace("×", "*").replace("·", "*").replace("÷", "/")

    # x², x³, x¹² -> x**2, x**3, x**12
    value = re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+",
        lambda m: "**" + m.group(0).translate(_SUPERSCRIPT_DIGITS),
        value,
    )
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _validate_expression_text(expression: str) -> None:
    if not expression:
        raise ValueError("La expresión matemática está vacía.")
    if len(expression) > 500:
        raise ValueError("La expresión matemática es demasiado larga.")
    if not _ALLOWED_EXPRESSION.fullmatch(expression):
        raise ValueError("La expresión contiene caracteres no permitidos.")


def _parse_expression(expression: str):
    _require_sympy()
    normalized = normalize_expression(expression)
    _validate_expression_text(normalized)
    if "=" in normalized:
        raise ValueError("Se esperaba una expresión, no una ecuación.")
    return parse_expr(
        normalized,
        transformations=_TRANSFORMATIONS,
        evaluate=True,
    )


def infer_variable(expression: str, default: str = "x") -> str:
    """Devuelve la primera variable simbólica de una expresión, de forma estable."""
    _require_sympy()
    raw = normalize_expression(expression)
    equation_text = raw.replace("=", "-")
    _validate_expression_text(equation_text)
    expr = parse_expr(
        equation_text,
        transformations=_TRANSFORMATIONS,
        evaluate=False,
    )
    symbols = sorted(str(symbol) for symbol in expr.free_symbols)
    return symbols[0] if symbols else str(default)


def simplify_expression(expression: str) -> str:
    expr = _parse_expression(expression)
    return str(sp.simplify(expr))


def solve_equation(equation: str, variable: str) -> list[str]:
    _require_sympy()
    variable_name = str(variable or "").strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", variable_name):
        raise ValueError("La variable de la ecuación no es válida.")

    symbol = sp.Symbol(variable_name)
    raw = normalize_expression(equation)
    _validate_expression_text(raw)

    if "=" in raw:
        left, right = raw.split("=", 1)
        expr = sp.Eq(
            parse_expr(left.strip(), transformations=_TRANSFORMATIONS, evaluate=True),
            parse_expr(right.strip(), transformations=_TRANSFORMATIONS, evaluate=True),
        )
    else:
        expr = sp.Eq(
            parse_expr(raw, transformations=_TRANSFORMATIONS, evaluate=True),
            0,
        )

    return [str(value) for value in sp.solve(expr, symbol)]


def derivative(expression: str, variable: str) -> str:
    _require_sympy()
    variable_name = str(variable or "").strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", variable_name):
        raise ValueError("La variable de derivación no es válida.")
    symbol = sp.Symbol(variable_name)
    expr = _parse_expression(expression)
    return str(sp.diff(expr, symbol))


def expression_domain_exclusions(
    expression: str,
    variable: str | None = None,
) -> list[str]:
    """Devuelve valores excluidos por ceros del denominador de la expresión original."""
    _require_sympy()
    expr = _parse_expression(expression)

    if variable:
        symbols: Iterable = [sp.Symbol(str(variable))]
    else:
        symbols = sorted(expr.free_symbols, key=lambda item: str(item))

    denominator = sp.denom(expr)
    if denominator == 1:
        return []

    exclusions: list[str] = []
    for symbol in symbols:
        try:
            roots = sp.solve(sp.Eq(denominator, 0), symbol)
        except Exception:
            continue
        for root in roots:
            rendered = f"{symbol} != {root}"
            if rendered not in exclusions:
                exclusions.append(rendered)

    return exclusions
