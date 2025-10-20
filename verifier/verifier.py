# -*- coding: utf-8 -*-
"""
verifier/verifier.py
Verificador ligero + autocorrección heurística.
API:
    verify(output: str, ctx: dict = None) -> dict
    attempt_self_correction(output: str, ctx: dict = None) -> dict
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .rules import DEFAULT_RULES, Rule, MIN_CHARS, PUNCTUATION


def _normalize_text(s: str) -> str:
    return (s or "").strip()


def _run_rules(output: str, rules: List[Rule]) -> Dict[str, Any]:
    issues: List[str] = []
    suggestions: List[str] = []
    for r in rules:
        try:
            if not r.check(output):
                issues.append(r.issue)
                suggestions.append(r.suggestion)
        except Exception as e:
            issues.append(f"Error evaluando regla '{r.name}': {e}")
    ok = len(issues) == 0
    return {"ok": ok, "issues": issues, "suggestions": suggestions}


def verify(output: str, ctx: Optional[Dict[str, Any]] = None,
           rules: Optional[List[Rule]] = None) -> Dict[str, Any]:
    """
    Verifica una salida de modelo con reglas simples.
    ctx puede incluir: {"min_score": float, "rag_hits": int, ...} (se preserva en la respuesta)
    """
    ctx = ctx or {}
    rules = rules or DEFAULT_RULES
    text = _normalize_text(output)

    res = _run_rules(text, rules)
    res["autocorrected"] = False
    res["used_model"] = False
    res["trace"] = []
    res["hint"] = ctx.get("hint") or "Prueba con min_score ligeramente menor (p.ej., 0.36)."
    res["stats"] = {k: ctx[k] for k in ("rag_hits", "mem_hits", "min_score", "top_k") if k in ctx}
    return res


def _heuristic_fix(output: str) -> str:
    """
    Pequeña autocorrección offline:
    - añade puntuación final si falta
    - si es muy corto, añade una ampliación con pasos concretos
    """
    t = _normalize_text(output)

    if not t.endswith(PUNCTUATION):
        t += "."

    # si queda demasiado corto, dale un empujón útil
    no_spaces = sum(1 for c in t if not c.isspace())
    if no_spaces < MIN_CHARS:
        t += (
            "\n\nPasos propuestos:\n"
            "1) Resume el objetivo en 1 línea.\n"
            "2) Da 2-3 pasos accionables (imperativo, concisos).\n"
            "3) Cierra con una frase que sintetice el resultado esperado."
        )
    return t


def attempt_self_correction(output: str, ctx: Optional[Dict[str, Any]] = None,
                            rules: Optional[List[Rule]] = None) -> Dict[str, Any]:
    """
    Intenta mejorar la salida sin depender de un LLM externo (heurístico).
    Si tienes un adaptador de LLM, podrías integrarlo aquí.
    """
    rules = rules or DEFAULT_RULES
    fixed = _heuristic_fix(output)
    check = _run_rules(fixed, rules)
    return {
        "ok": check["ok"],
        "autocorrected": True,
        "output": fixed,
        "issues": check["issues"],
        "suggestions": check["suggestions"],
        "used_model": False,   # no usamos LLM aquí
        "trace": ["heuristic:punctuation_and_minlen"],
    }
