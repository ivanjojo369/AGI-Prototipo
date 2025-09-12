# reasoner/orchestrator.py — Orquestador Fase 2 (final)
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import ast, math, os, re

@dataclass
class Decision:
    skill: str
    score: float
    reason: str

class SkillOrchestrator:
    """
    memory_vector_provider(query, k) -> List[Dict] con claves:
      text, score (0..1), citation_id, confidence?, metadata?
    """
    def __init__(
        self,
        memory_vector_provider: Callable[[str, int], List[Dict[str, Any]]],
        *,
        fs_root: str = ".",
        score_threshold: float = 0.45,
    ) -> None:
        self.memv = memory_vector_provider
        self.fs_root = os.path.abspath(fs_root)
        self.score_threshold = float(score_threshold)

    # ------------ selección ------------
    def choose_skill(self, query: str) -> Decision:
        q = (query or "").strip()
        ql = q.lower()

        # ---- python_exec: solo si hay operador y no es número suelto
        expr = self._extract_math(q)
        has_op = any(op in expr for op in "+-*/%^")
        only_number = bool(re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", expr))

        py_score = 0.0
        if has_op and not only_number:
            try:
                ast.parse(expr, mode="eval")
                py_score = 0.8
            except Exception:
                py_score = 0.0  # si no parsea, no la consideramos

        # ---- filesystem
        fs_score = 0.0
        if any(k in ql for k in ["archivo", "file", "leer", "lee", "open ", ".txt", ".md", ".json", ".py"]):
            fs_score = 0.8

        # ---- web (stub)
        web_score = 0.0
        if any(k in ql for k in ["buscar", "web", "google", "wikipedia", "noticias", "http://", "https://"]):
            web_score = 0.6

        # ---- memoria vectorial (peek del top-1)
        mem_score = 0.45
        try:
            peek = self.memv(q, 1)
            top = float(peek[0].get("score", 0.0)) if peek else 0.0
            mem_score = max(mem_score, top)
            # Si hay número pero SIN operadores, y la memoria tiene señal,
            # prioriza memoria sobre python_exec.
            if top >= 0.25 and (not has_op or only_number):
                mem_score = max(mem_score, py_score + 0.15)
        except Exception:
            pass

        candidates = [
            Decision("python_exec",      py_score,  "expresión matemática/cálculo"),
            Decision("filesystem_read",  fs_score,  "consulta de archivo"),
            Decision("search_web",       web_score, "información externa"),
            Decision("memory_vector",    mem_score, "contexto interno"),
        ]
        candidates.sort(key=lambda d: d.score, reverse=True)
        top = candidates[0]
        if top.score < self.score_threshold:
            return Decision("memory_vector", mem_score, "umbral bajo; fallback a memoria")
        return top

    # ------------ ejecución ------------
    def execute(self, query: str, *, k: int = 5) -> Dict[str, Any]:
        decision = self.choose_skill(query)
        if decision.skill == "python_exec":
            out = self._run_python_exec(query)
        elif decision.skill == "filesystem_read":
            out = self._run_filesystem_read(query)
        elif decision.skill == "search_web":
            out = self._run_search_web(query)
        else:
            out = self._run_memory_vector(query, k=k)

        if not out.get("ok"):
            out = self._run_memory_vector(query, k=k)
            decision = Decision("memory_vector", 0.5, "fallback por salida vacía")

        return {
            "ok": True,
            "skill": decision.skill,
            "confidence": out.get("confidence", decision.score),
            "reason": decision.reason,
            "output": out.get("output"),
            "citations": out.get("citations", []),
            "raw": out.get("raw", None),
        }

    # ------------ skills ------------
    def _run_memory_vector(self, query: str, *, k: int) -> Dict[str, Any]:
        results = self.memv(query, k)
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

    def _run_search_web(self, query: str) -> Dict[str, Any]:
        # Stub sin red (reemplazar por conector real cuando lo tengas)
        return {"ok": True, "output": f"(web) Buscar: '{query}'", "citations": [], "confidence": 0.35}

    # ---- python_exec seguro con extracción y ^ -> ** ----
    def _extract_math(self, s: str) -> str:
        s = s.strip()
        for p in ("calcula ", "calcular ", "resuelve ", "resolver ", "compute ", "calc "):
            if s.lower().startswith(p):
                s = s[len(p):].strip()
                break
        # tomar bloque con dígitos/operadores (más largo)
        m = re.findall(r'([0-9\.\s\+\-\*\/\%\^\(\)]+)', s)
        if m:
            s = max(m, key=len).strip()
        return s.replace("^", "**")  # potencia

    def _run_python_exec(self, expr: str) -> Dict[str, Any]:
        expr_clean = self._extract_math(expr)
        has_op = any(op in expr_clean for op in "+-*/%^")
        only_number = bool(re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", expr_clean))
        if not has_op or only_number:
            # evita casos como "Crítica 1781" (número suelto)
            return {"ok": False, "output": "no-op", "citations": [], "confidence": 0.0}

        allowed = {
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
            "pow": pow, "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
            "sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi, "e": math.e,
            "floor": math.floor, "ceil": math.ceil
        }

        class Safe(ast.NodeTransformer):
            ALLOWED = (
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Load,
                ast.Tuple, ast.List
            )
            def generic_visit(self, node):
                if not isinstance(node, self.ALLOWED):
                    raise ValueError(f"Nodo no permitido: {type(node).__name__}")
                return super().generic_visit(node)

        try:
            tree = ast.parse(expr_clean, mode="eval")
            Safe().visit(tree)
            code = compile(tree, "<expr>", "eval")
            val = eval(code, {"__builtins__": {}}, allowed)
            return {"ok": True, "output": str(val), "citations": [], "confidence": 0.9}
        except Exception as e:
            return {"ok": False, "output": f"Error: {e}", "citations": [], "confidence": 0.0}

    def _run_filesystem_read(self, query: str) -> Dict[str, Any]:
        q = query.strip()
        path: Optional[str] = None
        if "'" in q or '"' in q:
            quote = "'" if "'" in q else '"'
            try:
                path = q.split(quote)[1]
            except Exception:
                path = None
        if not path:
            for key in ["leer ", "lee ", "open ", "archivo ", "file "]:
                if key in q.lower():
                    path = q.lower().split(key, 1)[-1].strip()
                    break
        if not path:
            return {"ok": False, "output": "No encontré ruta de archivo", "citations": [], "confidence": 0.0}

        abs_path = os.path.abspath(path)
        if not abs_path.startswith(self.fs_root):
            return {"ok": False, "output": "Ruta fuera de la zona permitida", "citations": [], "confidence": 0.0}
        if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
            return {"ok": False, "output": "Archivo no encontrado", "citations": [], "confidence": 0.0}

        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read(64_000)
        return {"ok": True, "output": data, "citations": [{"id": f"file://{abs_path}", "score": 1.0}], "confidence": 0.7}
