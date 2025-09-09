from __future__ import annotations
"""
Calculator tool (safe AST evaluator)
- Operators: +, -, *, /, //, %, ^ (as power), unary +/-, parentheses
- Functions: sqrt, log, ln, log10, exp, sin, cos, tan, asin, acos, atan, degrees, radians, abs, round, pow
- Constants: pi, e
- Safety: AST whitelist + exponent clamp + input length cap
- API: 
    - calculate(expr: str) -> str
    - run(prompt: str = "", expr: str | None = None, **kw) -> str  (wrapper for selector)
"""

import ast
import math
import operator as op
import re
from typing import Any, Callable, Dict

# -----------------
# Limits / Safety
# -----------------
MAX_EXPR_LEN = 1000
MAX_ABS_EXPONENT = 100  # clamp exponent in a**b (|b| <= this)

# Allowed operators
_ALLOWED_OPS: Dict[type, Callable[..., Any]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

# Allowed math functions and constants
_ALLOWED_FUNCS: Dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "log": math.log,      # natural log
    "ln": math.log,       # alias of log
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "degrees": math.degrees,
    "radians": math.radians,
    "abs": abs,
    "round": round,
    "pow": pow,          # prefer operator **, but function allowed
}
_ALLOWED_CONSTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

_MATH_EXTRACT = re.compile(r"[0-9\.\(\)\+\-\*/\^eE\s]+")


class CalcError(Exception):
    pass


def _ensure_bounds_pow(base: Any, exp: Any) -> None:
    """Limit exponent magnitude to avoid DoS (e.g., 10**100000)."""
    try:
        exp_val = float(exp)
    except Exception as _:
        raise CalcError("Exponente no numérico en potencia.")
    if abs(exp_val) > MAX_ABS_EXPONENT:
        raise CalcError(f"Exponente demasiado grande (>|{MAX_ABS_EXPONENT}|).")


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalcError("Constante no numérica.")

    if isinstance(node, ast.Num):  # pragma: no cover (py<3.8)
        return node.n

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTS:
            return _ALLOWED_CONSTS[node.id]
        raise CalcError(f"Identificador no permitido: {node.id}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand)
        fn = _ALLOWED_OPS.get(type(node.op))
        if not fn:
            raise CalcError("Operación unaria no permitida.")
        return fn(operand)

    if isinstance(node, ast.BinOp):
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Pow):
            _ensure_bounds_pow(left, right)
        fn = _ALLOWED_OPS.get(type(node.op))
        if not fn:
            raise CalcError("Operación no permitida.")
        # Handle true division by zero gracefully
        if isinstance(node.op, ast.Div) and float(right) == 0.0:
            raise CalcError("División entre cero.")
        return fn(left, right)

    if isinstance(node, ast.Call):
        # Only support simple function names, no attrs
        if not isinstance(node.func, ast.Name):
            raise CalcError("Llamadas no permitidas.")
        fname = node.func.id
        f = _ALLOWED_FUNCS.get(fname)
        if not f:
            raise CalcError(f"Función no permitida: {fname}")
        # Evaluate arguments
        args = [_eval(a) for a in node.args]
        # Arity checks for common funcs (optional, most builtins will raise anyway)
        if fname in {"sqrt", "exp", "sin", "cos", "tan", "asin", "acos", "atan", "degrees", "radians", "abs"}:
            if len(args) != 1:
                raise CalcError(f"{fname} espera 1 argumento.")
        if fname in {"log", "ln"}:
            if not (1 <= len(args) <= 2):
                raise CalcError("log/ln acepta 1 o 2 argumentos (valor[, base]).")
        if fname in {"round"}:
            if not (1 <= len(args) <= 2):
                raise CalcError("round acepta 1 o 2 argumentos.")
        if fname in {"pow"}:
            if len(args) != 2:
                raise CalcError("pow espera 2 argumentos.")
            _ensure_bounds_pow(args[0], args[1])
        return f(*args)

    raise CalcError("Expresión inválida o no soportada.")


def calculate(expr: str) -> str:
    """Evalúa una expresión matemática segura. Devuelve el resultado como string.
    Soporta '^' como potencia y constantes pi/e.
    """
    if not expr:
        return "Expresión vacía."
    expr = expr.strip()
    if len(expr) > MAX_EXPR_LEN:
        return "Expresión demasiado larga."
    expr = expr.replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval(tree)
        # Formateo compacto; evita notación científica para enteros
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except CalcError as e:
        return f"Error: {e}"
    except Exception as e:  # errores de sintaxis u otros
        return f"Error: {e}"


def _extract_expr(prompt: str) -> str | None:
    if not prompt:
        return None
    # toma la secuencia matemática más larga con operadores
    cands = [m.group(0) for m in _MATH_EXTRACT.finditer(prompt)]
    cands = [c.strip() for c in cands if re.search(r"[\+\-\*/\^]", c)]
    if cands:
        return max(cands, key=len)
    # fallback: texto tras gatillos comunes
    m = re.search(r"(calcula|resultado|cu[aá]nto es|cuanto es)\s*[:\-]?\s*(.+)$", prompt, re.IGNORECASE)
    if m:
        return m.group(2)
    return None


def run(prompt: str = "", expr: str | None = None, **kw) -> str:
    """Wrapper estándar para el selector/tool runner.
    Prioriza `expr`; si no existe, intenta extraer la expresión del `prompt`.
    """
    expr = expr or _extract_expr(prompt) or ""
    return calculate(expr)


__all__ = ["calculate", "run"]
