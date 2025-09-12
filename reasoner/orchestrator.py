# reasoner/orchestrator.py — Orquestador Fase 2 con alias memory_vector / memory_search
from __future__ import annotations

import ast
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Decision:
    skill: str
    score: float
    reason: str


class SkillOrchestrator:
    """
    Orquestador de habilidades con selección heurística.
    Requiere un proveedor de memoria vectorial con firma:
        memory_vector_provider(query: str, k: int) -> List[Dict[str, Any]]
    """

    def __init__(
        self,
        memory_vector_provider: Callable[[str, int], List[Dict[str, Any]]],
        *,
        fs_root: str = ".",
        score_threshold: float = 0.45,
    ) -> None:
        self._memv = memory_vector_provider
        self._fs_root = os.path.abspath(fs_root)
        self._score_threshold = float(score_threshold)

        # Registro de skills: alias 'memory_vector' y 'memory_search' apuntan al mismo handler
        self.skills: Dict[str, Callable[..., Dict[str, Any]]] = {
            "memory_vector": self._skill_memory_vector,   # alias
            "memory_search": self._skill_memory_vector,   # nombre preferido para tests
            "python_exec": self._skill_python_exec,
            "filesystem_read": self._skill_filesystem_read,
            "search_web": self._skill_search_web,
        }

    # ------------------------------------------------------------------ Utils
    def available_skills(self) -> Tuple[str, ...]:
        """Lista de skills disponibles (incluye alias)."""
        return tuple(self.skills.keys())

    def _extract_math(self, s: str) -> str:
        """Extrae la parte matemática y convierte '^' en '**'."""
        s = (s or "").strip()
        for p in ("calcula ", "calcular ", "resuelve ", "resolver ", "compute ", "calc "):
            if s.lower().startswith(p):
                s = s[len(p):].strip()
                break
        m = re.findall(r"([0-9\.\s\+\-\*\/\%\^\(\)]+)", s)
        if m:
            s = max(m, key=len).strip()
        return s.replace("^", "**")

    def _infer_path_from_query(self, q: str) -> Optional[str]:
        q = (q or "").strip()
        # entre comillas
        if "'" in q or '"' in q:
            quote = "'" if "'" in q else '"'
            try:
                return q.split(quote)[1]
            except Exception:
                pass
        # heurística simple
        for key in ["leer ", "lee ", "open ", "archivo ", "file "]:
            if key in q.lower():
                return q.lower().split(key, 1)[-1].strip()
        return None

    # ---------------------------------------------------------- Skill choice
    def choose_skill(self, query: str) -> Decision:
        q = (query or "").strip()
        ql = q.lower()

        # python_exec: hay operadores y no es sólo número
        expr = self._extract_math(q)
        has_op = any(op in expr for op in "+-*/%^")
        only_number = bool(re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", expr))

        py_score = 0.0
        if has_op and not only_number:
            try:
                ast.parse(expr, mode="eval")
                py_score = 0.8
            except Exception:
                py_score = 0.0

        fs_score = 0.8 if any(k in ql for k in ["archivo", "file", "leer", "lee ", ".txt", ".md", ".json", ".py"]) else 0.0
        web_score = 0.6 if any(k in ql for k in ["buscar", "web", "google", "wikipedia", "http://", "https://"]) else 0.0

        mem_score = 0.45
        try:
            peek = self._memv(q, 1)
            top = float(peek[0].get("score", 0.0)) if peek else 0.0
            mem_score = max(mem_score, top)
            if top >= 0.25 and (not has_op or only_number):
                mem_score = max(mem_score, py_score + 0.15)
        except Exception:
            pass

        # Importante: el candidato de memoria se etiqueta como 'memory_search' (no 'memory_vector')
        candidates = [
            Decision("python_exec",     py_score,  "expresión matemática/cálculo"),
            Decision("filesystem_read", fs_score,  "lectura de archivo"),
            Decision("search_web",      web_score, "información externa"),
            Decision("memory_search",   mem_score, "contexto interno (vector)"),
        ]
        candidates.sort(key=lambda d: d.score, reverse=True)
        top = candidates[0]
        if top.score < self._score_threshold:
            return Decision("memory_search", mem_score, "umbral bajo; fallback a memoria")
        return top

    # -------------------------------------------------------------- Execute
    def choose_and_execute(self, name: str, **kwargs) -> Dict[str, Any]:
        """Ejecuta una skill por nombre (usado en tests)."""
        if name not in self.skills:
            raise KeyError(f"Skill desconocida: {name}")
        return self.skills[name](**kwargs)

    def execute(self, query: str, *, k: int = 5) -> Dict[str, Any]:
        """Elige skill automáticamente y la ejecuta."""
        decision = self.choose_skill(query)

        if decision.skill == "python_exec":
            out = self._skill_python_exec(code=query)
        elif decision.skill == "filesystem_read":
            out = self._skill_filesystem_read(path=self._infer_path_from_query(query))
        elif decision.skill == "search_web":
            out = self._skill_search_web(query=query)
        else:
            # memoria (acepta ambos nombres en el registro)
            out = self._skill_memory_vector(query=query, k=k)

        if not out.get("ok"):
            out = self._skill_memory_vector(query=query, k=k)
            decision = Decision("memory_search", 0.5, "fallback por salida vacía")

        return {
            "ok": True,
            "skill": decision.skill,  # será 'memory_search' para memoria
            "confidence": out.get("confidence", decision.score),
            "reason": decision.reason,
            "output": out.get("output"),
            "citations": out.get("citations", []),
            "raw": out.get("raw"),
        }

    # ---------------------------------------------------------- Skill impls
    def _skill_memory_vector(self, *, query: str, k: int = 5) -> Dict[str, Any]:
        results = self._memv(query, k)
        if not results:
            return {"ok": False, "output": "Sin resultados", "citations": [], "confidence": 0.0}
        top = results[0]
        citations = [{"id": r.get("citation_id"), "score": r.get("score")} for r in results]
        return {
            "ok": True,
            "output": top.get("text"),
            "citations": citations,
            "confidence": float(top.get("score", 0.0)),
            "raw": results,
        }

    def _skill_search_web(self, *, query: str, k: int = 5) -> Dict[str, Any]:
        # Stub sin red: reemplázalo por tu conector real cuando lo tengas
        return {"ok": True, "output": f"(web) Buscar: '{query}'", "citations": [], "confidence": 0.35}

    def _skill_python_exec(self, *, code: str) -> Dict[str, Any]:
        expr_clean = self._extract_math(code)
        has_op = any(op in expr_clean for op in "+-*/%^")
        only_number = bool(re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", expr_clean))
        if not has_op or only_number:
            # Evita casos como "Kant 1781" (número suelto sin operación)
            return {"ok": False, "output": "no-op", "citations": [], "confidence": 0.0}

        allowed = {
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
            "pow": pow, "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
            "sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi, "e": math.e,
            "floor": math.floor, "ceil": math.ceil,
        }

        class Safe(ast.NodeTransformer):
            ALLOWED = (
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Load,
                ast.Tuple, ast.List,
            )

            def generic_visit(self, node):
                if not isinstance(node, self.ALLOWED):
                    raise ValueError(f"Nodo no permitido: {type(node).__name__}")
                return super().generic_visit(node)

        try:
            tree = ast.parse(expr_clean, mode="eval")
            Safe().visit(tree)
            val = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, allowed)
            return {"ok": True, "output": str(val), "citations": [], "confidence": 0.9}
        except Exception as e:
            return {"ok": False, "output": f"Error: {e}", "citations": [], "confidence": 0.0}

    def _skill_filesystem_read(
        self, *, path: Optional[str], encoding: str = "utf-8", max_bytes: int = 2_000_000
    ) -> Dict[str, Any]:
        if not path:
            return {"ok": False, "output": "No encontré ruta de archivo", "citations": [], "confidence": 0.0}

        abs_path = os.path.abspath(path)
        if not abs_path.startswith(self._fs_root):
            return {"ok": False, "output": "Ruta fuera de la zona permitida", "citations": [], "confidence": 0.0}
        if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
            return {"ok": False, "output": "Archivo no encontrado", "citations": [], "confidence": 0.0}

        size = os.path.getsize(abs_path)
        if size > max_bytes:
            return {"ok": False, "output": "Archivo demasiado grande", "citations": [], "confidence": 0.0}

        with open(abs_path, "r", encoding=encoding, errors="ignore") as f:
            data = f.read(min(size, 64_000))

        return {"ok": True, "output": data, "citations": [{"id": f"file://{abs_path}", "score": 1.0}], "confidence": 0.7}
