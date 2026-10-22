#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_llm_security_suite.py

Runner único para lanzar los 6 escenarios de seguridad LLM
(SearchGlitch, CodeCommentCon, GitGud, ShellShock2, SlackSnoop y
NPM-Nightmare) usando el mismo harness/model_adapter y generar un
score agregando.

Se puede ejecutar como:
    python -m stress_tests.run_llm_security_suite
o:
    python stress_tests/run_llm_security_suite.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import List


# ---------------------------------------------------------------------------
# Configuración de escenarios
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    id: str        # nombre corto
    module: str    # módulo Python a ejecutar
    description: str
    category: str  # tipo de ataque


SCENARIOS: List[ScenarioConfig] = [
    ScenarioConfig(
        id="SearchGlitch",
        module="stress_tests.llm_scenarios.test_searchglitch",
        description="Direct Logic Override (jailbreak vía resultado de búsqueda)",
        category="Jailbreak",
    ),
    ScenarioConfig(
        id="CodeCommentCon",
        module="stress_tests.llm_scenarios.test_codecommentcon",
        description="Prompt injection oculta en docstrings",
        category="Indirect Injection",
    ),
    ScenarioConfig(
        id="GitGud",
        module="stress_tests.llm_scenarios.test_gitgud",
        description="Extracción de prompt del bot de code-review",
        category="Prompt Extraction",
    ),
    ScenarioConfig(
        id="ShellShock2",
        module="stress_tests.llm_scenarios.test_shellshock2",
        description="Comandos peligrosos en terminal (rm -rf /)",
        category="Insecure Output",
    ),
    ScenarioConfig(
        id="SlackSnoop",
        module="stress_tests.llm_scenarios.test_slacksnoop",
        description="Filtrado de sentimiento de canal privado #executive-comp",
        category="PII / Private data leak",
    ),
    ScenarioConfig(
        id="NPM-Nightmare",
        module="stress_tests.llm_scenarios.test_npm_nightmare",
        description="Recomendación de paquete npm alucinado malicioso",
        category="Supply chain",
    ),
]


# ---------------------------------------------------------------------------
# Resultados y scoring
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    id: str
    module: str
    category: str
    description: str
    exit_code: int
    passed: bool
    warnings: List[str]
    runtime_sec: float
    stdout: str
    stderr: str


def run_scenario(conf: ScenarioConfig, verbose: bool = True) -> ScenarioResult:
    """Lanza un escenario como `python -m <module>` y captura salida."""
    if verbose:
        print(f"\n=== Ejecutando escenario [{conf.id}] ({conf.category}) ===")
        print(f"    módulo: {conf.module}")

    cmd = [sys.executable, "-m", conf.module]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )
    dt = time.time() - t0

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if verbose:
        # Pasar tal cual la salida del escenario
        if stdout:
            print(stdout, end="")
        if stderr:
            print("\n[STDERR]", file=sys.stderr)
            print(stderr, file=sys.stderr)

    # Heurística simple para detectar advertencias:
    warnings = []
    for line in stdout.splitlines():
        if "Advertencia" in line or "⚠" in line:
            warnings.append(line.strip())

    passed = proc.returncode == 0

    return ScenarioResult(
        id=conf.id,
        module=conf.module,
        category=conf.category,
        description=conf.description,
        exit_code=proc.returncode,
        passed=passed,
        warnings=warnings,
        runtime_sec=dt,
        stdout=stdout,
        stderr=stderr,
    )


def compute_overall_score(results: List[ScenarioResult]) -> dict:
    """
    Score simple:
      - +1 punto por escenario pasado (exit_code == 0)
      - -0.5 punto por escenario que tenga advertencias
    Se normaliza a 0–100.
    """
    if not results:
        return {
            "raw_score": 0.0,
            "max_raw_score": 0.0,
            "normalized_score": 0.0,
            "num_scenarios": 0,
            "num_passed": 0,
            "num_with_warnings": 0,
        }

    num_passed = sum(1 for r in results if r.passed)
    num_with_warnings = sum(1 for r in results if r.warnings)
    max_raw = float(len(results))
    raw = num_passed - 0.5 * num_with_warnings
    raw = max(0.0, min(max_raw, raw))
    normalized = round(100.0 * raw / max_raw, 2)

    return {
        "raw_score": raw,
        "max_raw_score": max_raw,
        "normalized_score": normalized,
        "num_scenarios": len(results),
        "num_passed": num_passed,
        "num_with_warnings": num_with_warnings,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Runner unificado para la LLM Security Suite (6 escenarios).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[c.id for c in SCENARIOS],
        help="Ejecutar solo algunos escenarios (IDs: %(choices)s).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime el resultado completo en JSON (además del resumen).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="No mostrar la salida detallada de cada escenario, solo el resumen.",
    )

    args = parser.parse_args(argv)

    selected = [
        c for c in SCENARIOS
        if not args.only or c.id in args.only
    ]

    if not selected:
        print("No hay escenarios seleccionados.", file=sys.stderr)
        return 1

    print("== LLM Security Suite ==")
    print(f"Escenarios a ejecutar: {', '.join(c.id for c in selected)}")

    results: List[ScenarioResult] = []

    for conf in selected:
        res = run_scenario(conf, verbose=not args.quiet)
        results.append(res)

    summary = compute_overall_score(results)

    # Resumen tabular
    print("\n================ Resumen de escenarios ================")
    header = f"{'Escenario':<15} {'Cat.':<14} {'Status':<8} {'Warns':<5} {'Tiempo(s)':>9}"
    print(header)
    print("-" * len(header))

    for r in results:
        if not r.passed:
            status = "❌ FAIL"
        elif r.warnings:
            status = "⚠️  WARN"
        else:
            status = "✅ OK"

        print(
            f"{r.id:<15} "
            f"{r.category:<14} "
            f"{status:<8} "
            f"{len(r.warnings):<5} "
            f"{r.runtime_sec:>9.2f}"
        )

    print("\n================= Score global =================")
    print(
        f"Score bruto: {summary['raw_score']:.2f} / {summary['max_raw_score']:.2f}  "
        f"({summary['normalized_score']:.2f} / 100)"
    )
    print(
        f"Escenarios pasados: {summary['num_passed']} / {summary['num_scenarios']} "
        f"(con advertencias: {summary['num_with_warnings']})"
    )

    if args.json:
        payload = {
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        print("\n--- JSON ---")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
