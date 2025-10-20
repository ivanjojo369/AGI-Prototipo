# -*- coding: utf-8 -*-
"""
verifier/rules.py
Reglas básicas de verificación para la AGI doméstica.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    check: Callable[[str], bool]
    issue: str
    suggestion: str


MIN_CHARS = 80
PUNCTUATION = (".", "!", "?")


def _len_ok(s: str) -> bool:
    # Cuenta caracteres no-espaciado para ser un poco más estrictos
    n = sum(1 for c in s if not c.isspace())
    return n >= MIN_CHARS


def _ends_with_punct(s: str) -> bool:
    s = s.strip()
    return len(s) > 0 and s.endswith(PUNCTUATION)


DEFAULT_RULES: List[Rule] = [
    Rule(
        name="longitud_minima",
        description=f"Debe tener al menos {MIN_CHARS} caracteres útiles.",
        check=_len_ok,
        issue=f"Salida demasiado corta (<{MIN_CHARS} caracteres).",
        suggestion="Amplía con pasos accionables y, si aplica, un ejemplo breve.",
    ),
    Rule(
        name="cierre_con_puntuacion",
        description="Debe terminar con punto, signo de exclamación o interrogación.",
        check=_ends_with_punct,
        issue="La respuesta no termina con puntuación.",
        suggestion="Añade un cierre gramatical (punto u otro signo adecuado).",
    ),
]
