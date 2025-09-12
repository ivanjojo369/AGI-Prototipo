# reasoner/task_planner.py
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Step:
    """Paso atómico que el orquestador puede ejecutar como skill."""
    skill: str
    args: Dict[str, Any]
    description: str = ""


@dataclass
class Plan:
    """Plan HTN-lite resultante de descomponer un goal."""
    goal: str
    steps: List[Step]
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "rationale": self.rationale,
            "steps": [asdict(s) for s in self.steps],
        }


class TaskPlanner:
    """
    Planificador jerárquico ligero (HTN-lite) guiado por heurísticas.
    Sin LLM, sin GPU. Produce pasos que casan con tus skills:
      - memory_vector_search  (búsqueda interna)
      - search_web            (web)
      - python_exec           (cálculo)
      - filesystem_read       (archivos locales)
    """

    def __init__(self, default_k: int = 3) -> None:
        self.default_k = default_k

    # ----------------------------
    # Detectores de intención
    # ----------------------------
    _MATH_RE = re.compile(r"(^|\s)(\d+(\s*[\+\-\*\/\%\^]\s*\d+)+)\s*$")
    _PATH_RE = re.compile(r'(?:"([^"]+)"|\'([^\']+)\')')

    def _is_math(self, text: str) -> Optional[str]:
        """Devuelve la expresión matemática si la detecta."""
        m = self._MATH_RE.search(text)
        return m.group(2) if m else None

    def _file_hint(self, text: str) -> Optional[str]:
        """
        Si el usuario menciona leer/ver un archivo e incluye una ruta entre comillas,
        devolvemos esa ruta.
        """
        if any(k in text.lower() for k in ["leer", "abre", "abrir", "archivo", "readme", "md", "csv", "json"]):
            m = self._PATH_RE.search(text)
            if m:
                return m.group(1) or m.group(2)
        return None

    def _needs_web(self, text: str) -> bool:
        t = text.lower()
        return any(k in t for k in ["buscar", "noticias", "últimas noticias", "en internet", "web"]) \
               and not any(k in t for k in ["memoria", "recuerda", "recordatorio"])

    def _memory_query(self, text: str) -> Optional[str]:
        t = text.lower()
        # heurística: “recuerda”, “memoria”, “qué dije…”, “recordatorio…”
        if any(k in t for k in ["recuerda", "memoria", "recordatorio", "qué dije", "que dije"]):
            return text
        # fallback: si no hay pista de web/archivo/cálculo, intentamos memoria primero
        if not self._is_math(text) and not self._needs_web(text) and not self._file_hint(text):
            return text
        return None

    # ----------------------------
    # Descomposición (HTN-lite)
    # ----------------------------
    def compose(self, goal: str) -> Plan:
        """
        Devuelve un plan (lista de pasos) para resolver el goal.
        No ejecuta nada; solo estructura.
        """
        steps: List[Step] = []
        normalized = goal.strip()

        # 1) ¿Es cálculo?
        expr = self._is_math(normalized)
        if expr:
            steps.append(
                Step(
                    skill="python_exec",
                    args={"code": f"result = {expr}\nprint(result)"},
                    description=f"Evaluar la expresión matemática: {expr}",
                )
            )
            return Plan(goal=goal, steps=steps, rationale="Detección de cálculo aritmético.")

        # 2) ¿Es un archivo?
        file_path = self._file_hint(normalized)
        if file_path:
            steps.append(
                Step(
                    skill="filesystem_read",
                    args={"path": file_path},
                    description=f"Leer archivo local '{file_path}'",
                )
            )
            return Plan(goal=goal, steps=steps, rationale="El objetivo menciona leer/abrir un archivo.")

        # 3) ¿Necesita web?
        if self._needs_web(normalized):
            steps.append(
                Step(
                    skill="search_web",
                    args={"query": normalized},
                    description="Buscar información externa en la web.",
                )
            )
            return Plan(goal=goal, steps=steps, rationale="Heurística: se pide búsqueda externa.")

        # 4) Por defecto, memoria → si no alcanza, web fallback
        mem_q = self._memory_query(normalized) or normalized
        steps.append(
            Step(
                skill="memory_vector_search",
                args={"q": mem_q, "k": self.default_k},
                description=f"Buscar en memoria vectorial: '{mem_q}'",
            )
        )
        steps.append(
            Step(
                skill="search_web",
                args={"query": normalized},
                description="Si la memoria no alcanza, intentar búsqueda externa.",
            )
        )
        return Plan(goal=goal, steps=steps, rationale="Ruta base: memoria y fallback a web.")

