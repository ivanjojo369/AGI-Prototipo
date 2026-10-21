# reasoner/orchestrator.py
from __future__ import annotations

import ast
import importlib
import math
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Decision:
    skill: str
    score: float
    reason: str


@dataclass
class OrchestrationResult:
    ok: bool
    skill: str
    confidence: float
    reason: str
    output: Any = None
    citations: List[Dict[str, Any]] | None = None
    raw: Any = None
    error_code: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "skill": self.skill,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "output": self.output,
            "citations": list(self.citations or []),
            "raw": self.raw,
            "error_code": self.error_code,
        }


class SkillOrchestrator:
    """
    Orchestrator 3.0 para Quipu.

    Responsabilidades:
    - elegir una capacidad sin sustituir a UnifiedMemory;
    - ejecutar habilidades deterministas seguras;
    - preparar planes de proyecto mediante un planner inyectado;
    - mantener compatibilidad con memory_search/memory_vector;
    - usar opcionalmente la capa ExecutionLoop de AegisForge si está disponible.

    UnifiedMemory sigue siendo la autoridad sobre qué memoria concreta recuperar.
    """

    VERSION = "3.0"

    MEMORY_MARKERS = (
        "recuerda",
        "recordar",
        "memoria",
        "recupera",
        "recuperar",
        "qué sabes sobre",
        "que sabes sobre",
        "qué recuerdas",
        "que recuerdas",
        "historial",
        "metamemoria",
        "objetivo pendiente",
        "objetivos pendientes",
        "procedimiento",
        "esquema del conocimiento",
        "conocimientos asociados",
        "conceptos relacionados",
        "cómo se relacionan",
        "como se relacionan",
        "quién estaba antes",
        "quien estaba antes",
    )

    PROJECT_MARKERS = (
        "haz un plan",
        "crea un plan",
        "planifica",
        "planea",
        "divide en subtareas",
        "divide en sub-tareas",
        "divide en pasos",
        "organiza en etapas",
        "ejecuta por etapas",
        "resuelve por etapas",
        "descompón la tarea",
        "descompon la tarea",
        "proyecto completo",
    )

    FILE_MARKERS = (
        "leer archivo",
        "lee el archivo",
        "abre el archivo",
        "open file",
        "read file",
    )

    WEB_MARKERS = (
        "busca en la web",
        "buscar en la web",
        "busca en internet",
        "buscar en internet",
        "consulta internet",
        "google",
        "wikipedia",
        "http://",
        "https://",
    )

    def __init__(
        self,
        memory_vector_provider: Callable[[str, int], List[Dict[str, Any]]],
        *,
        fs_root: str = ".",
        score_threshold: float = 0.50,
        project_planner: Optional[Callable[[str], Dict[str, Any]]] = None,
        enable_web: bool = False,
        use_aegis_execution: bool = True,
    ) -> None:
        self._memv = memory_vector_provider
        self._fs_root = os.path.abspath(fs_root)
        self._score_threshold = float(score_threshold)
        self._project_planner = project_planner
        self._enable_web = bool(enable_web)

        self._aegis = None
        if use_aegis_execution:
            self._aegis = self._try_build_aegis_bridge()

        self.skills: Dict[str, Callable[..., Dict[str, Any]]] = {
            "memory_vector": self._skill_memory_vector,
            "memory_search": self._skill_memory_vector,
            "python_exec": self._skill_python_exec,
            "filesystem_read": self._skill_filesystem_read,
            "search_web": self._skill_search_web,
            "project_plan": self._skill_project_plan,
            "chat": self._skill_chat,
        }

    # ---------------------------------------------------------- status

    def get_status(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "version": self.VERSION,
            "skills": list(self.skills.keys()),
            "web_enabled": self._enable_web,
            "project_planner": self._project_planner is not None,
            "aegis_execution_available": self._aegis is not None,
            "fs_root": self._fs_root,
        }

    def available_skills(self) -> Tuple[str, ...]:
        return tuple(self.skills.keys())

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _normalize(text: str) -> str:
        value = (text or "").strip().lower()
        value = (
            value.replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("ü", "u")
            .replace("ñ", "n")
        )
        return re.sub(r"\s+", " ", value)

    @classmethod
    def _contains_any(cls, text: str, markers: Tuple[str, ...]) -> bool:
        normalized = cls._normalize(text)
        return any(cls._normalize(marker) in normalized for marker in markers)

    def _extract_math(self, text: str) -> str:
        s = (text or "").strip()
        for prefix in (
            "calcula ",
            "calcular ",
            "resuelve ",
            "resolver ",
            "compute ",
            "calc ",
        ):
            if s.lower().startswith(prefix):
                s = s[len(prefix):].strip()
                break

        matches = re.findall(
            r"([0-9\.\s\+\-\*\/\%\^\(\),]+)",
            s,
        )
        if matches:
            s = max(matches, key=len).strip()

        return s.replace("^", "**")

    def _infer_path_from_query(self, query: str) -> Optional[str]:
        q = (query or "").strip()

        quoted = re.findall(r"""["']([^"']+)["']""", q)
        if quoted:
            return quoted[0].strip()

        match = re.search(
            r"(?:archivo|file)\s+([^\n]+)$",
            q,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        return None

    def _memory_peek_score(self, query: str) -> float:
        try:
            results = self._memv(query, 1)
            if not results:
                return 0.0
            return float(results[0].get("score", 0.0) or 0.0)
        except Exception:
            return 0.0

    # ---------------------------------------------------------- routing

    def choose_skill(self, query: str) -> Decision:
        q = (query or "").strip()
        qn = self._normalize(q)

        expr = self._extract_math(q)
        has_op = any(op in expr for op in "+-*/%")
        if "**" in expr:
            has_op = True

        py_score = 0.0
        if has_op:
            try:
                ast.parse(expr, mode="eval")
                py_score = 0.97
            except Exception:
                py_score = 0.0

        file_score = 0.0
        path = self._infer_path_from_query(q)
        if (
            self._contains_any(q, self.FILE_MARKERS)
            or (
                path
                and re.search(
                    r"\.(?:txt|md|json|py|csv|log|yaml|yml)$",
                    path,
                    flags=re.IGNORECASE,
                )
            )
        ):
            file_score = 0.93

        project_score = (
            0.91
            if self._contains_any(q, self.PROJECT_MARKERS)
            else 0.0
        )

        web_score = (
            0.88
            if self._contains_any(q, self.WEB_MARKERS)
            else 0.0
        )

        memory_intent = self._contains_any(q, self.MEMORY_MARKERS)
        memory_peek = self._memory_peek_score(q)
        memory_score = max(
            0.32,
            min(0.86, memory_peek),
        )
        if memory_intent:
            memory_score = max(memory_score, 0.90)

        # General conversation must remain a valid route; old Orchestrator Fase 2
        # forced almost everything into memory_search.
        chat_score = 0.55

        candidates = [
            Decision("python_exec", py_score, "expresión matemática verificable"),
            Decision("filesystem_read", file_score, "lectura explícita de archivo local"),
            Decision("project_plan", project_score, "tarea explícitamente multi-etapa"),
            Decision("search_web", web_score, "solicitud explícita de información externa"),
            Decision("memory_search", memory_score, "consulta de conocimiento/memoria"),
            Decision("chat", chat_score, "conversación o tarea general"),
        ]

        # Stable priority on exact ties.
        priority = {
            "python_exec": 6,
            "filesystem_read": 5,
            "project_plan": 4,
            "search_web": 3,
            "memory_search": 2,
            "chat": 1,
        }
        candidates.sort(
            key=lambda d: (d.score, priority.get(d.skill, 0)),
            reverse=True,
        )
        top = candidates[0]

        if top.score < self._score_threshold:
            return Decision(
                "chat",
                chat_score,
                "ninguna capacidad especializada superó el umbral",
            )

        return top

    # ---------------------------------------------------------- execution

    def choose_and_execute(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        if name not in self.skills:
            raise KeyError(f"Skill desconocida: {name}")
        return self.skills[name](**kwargs)

    def execute(self, query: str, *, k: int = 5) -> Dict[str, Any]:
        decision = self.choose_skill(query)

        if decision.skill == "chat":
            return OrchestrationResult(
                ok=True,
                skill="chat",
                confidence=decision.score,
                reason=decision.reason,
                output=None,
            ).as_dict()

        def direct_call() -> Dict[str, Any]:
            if decision.skill == "python_exec":
                return self._skill_python_exec(code=query)
            if decision.skill == "filesystem_read":
                return self._skill_filesystem_read(
                    path=self._infer_path_from_query(query)
                )
            if decision.skill == "search_web":
                return self._skill_search_web(query=query, k=k)
            if decision.skill == "project_plan":
                return self._skill_project_plan(goal=query)
            return self._skill_memory_vector(query=query, k=k)

        out = self._run_with_optional_aegis(
            query=query,
            skill=decision.skill,
            call=direct_call,
        )

        return OrchestrationResult(
            ok=bool(out.get("ok")),
            skill=decision.skill,
            confidence=float(
                out.get("confidence", decision.score) or 0.0
            ),
            reason=decision.reason,
            output=out.get("output"),
            citations=list(out.get("citations") or []),
            raw=out.get("raw"),
            error_code=out.get("error_code"),
        ).as_dict()

    # ---------------------------------------------------------- AegisForge bridge

    @staticmethod
    def _try_build_aegis_bridge() -> Any:
        """
        Intenta localizar AegisForge sin asumir que ``src`` sea un paquete.

        En este proyecto AegisForge normalmente vive en:
            AGI-Prototipo/AegisForge_agent/src/aegisforge

        Primero intenta importar el paquete instalado. Si no está instalado,
        añade dinámicamente ``AegisForge_agent/src`` al sys.path.
        """
        module = None

        try:
            module = importlib.import_module(
                "aegisforge.orchestration"
            )
        except Exception:
            module = None

        if module is None:
            try:
                project_root = (
                    Path(__file__)
                    .resolve()
                    .parents[1]
                )

                candidates = (
                    project_root
                    / "AegisForge_agent"
                    / "src",
                    project_root
                    / "src",
                )

                for candidate in candidates:
                    if (
                        candidate.exists()
                        and str(candidate)
                        not in sys.path
                    ):
                        sys.path.insert(
                            0,
                            str(candidate),
                        )

                importlib.invalidate_caches()

                module = importlib.import_module(
                    "aegisforge.orchestration"
                )
            except Exception:
                return None

        required = (
            "AdapterRequest",
            "AdapterResult",
            "EpisodeState",
            "ExecutionLoop",
            "TaskContextBuilder",
        )

        if not all(
            hasattr(module, name)
            for name in required
        ):
            return None

        return {
            "AdapterRequest":
                getattr(
                    module,
                    "AdapterRequest",
                ),
            "AdapterResult":
                getattr(
                    module,
                    "AdapterResult",
                ),
            "EpisodeState":
                getattr(
                    module,
                    "EpisodeState",
                ),
            "ExecutionLoop":
                getattr(
                    module,
                    "ExecutionLoop",
                )(),
            "TaskContextBuilder":
                getattr(
                    module,
                    "TaskContextBuilder",
                )(),
        }

    def _run_with_optional_aegis(
        self,
        *,
        query: str,
        skill: str,
        call: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        bridge = self._aegis
        if bridge is None:
            return call()

        try:
            context = bridge["TaskContextBuilder"].from_text(
                query,
                task_id="quipu-orchestrator",
                track_hint="quipu",
                metadata={"skill": skill},
            )
            episode = bridge["EpisodeState"](context=context)
            request = bridge["AdapterRequest"](
                track="quipu",
                task_text=query,
                prompt_profile="quipu-core",
                policy_profile="default",
                tool_mode=skill,
                metadata={"skill": skill},
            )

            def adapter_call(_request: Any) -> Any:
                result = call()
                return bridge["AdapterResult"](
                    ok=bool(result.get("ok")),
                    response_text=str(result.get("output") or ""),
                    artifacts={"orchestration_result": result},
                    warnings=[],
                    error_code=result.get("error_code"),
                    error_message=(
                        None
                        if result.get("ok")
                        else str(result.get("output") or "skill failed")
                    ),
                )

            finished = bridge["ExecutionLoop"].run(
                episode,
                request,
                adapter_call,
            )

            stored = dict(
                finished.final_artifacts or {}
            ).get("orchestration_result")
            if isinstance(stored, dict):
                return stored

            return {
                "ok": bool(finished.final_response),
                "output": finished.final_response or "",
                "citations": [],
                "confidence": 0.5,
            }
        except Exception:
            # Orchestration should never make Quipu unusable.
            return call()

    # ---------------------------------------------------------- skills

    def _skill_chat(self, **_: Any) -> Dict[str, Any]:
        return {
            "ok": True,
            "output": None,
            "citations": [],
            "confidence": 0.55,
        }

    def _skill_memory_vector(
        self,
        *,
        query: str,
        k: int = 5,
    ) -> Dict[str, Any]:
        results = self._memv(query, k)
        if not results:
            return {
                "ok": False,
                "output": "Sin resultados de memoria.",
                "citations": [],
                "confidence": 0.0,
                "error_code": "empty_memory",
            }

        top = results[0]
        citations = [
            {
                "id": item.get("citation_id"),
                "score": item.get("score"),
            }
            for item in results
        ]

        return {
            "ok": True,
            "output": top.get("text"),
            "citations": citations,
            "confidence": float(top.get("score", 0.0) or 0.0),
            "raw": results,
        }

    def _skill_search_web(
        self,
        *,
        query: str,
        k: int = 5,
    ) -> Dict[str, Any]:
        if not self._enable_web:
            return {
                "ok": False,
                "output": (
                    "La búsqueda web no está conectada al Orchestrator 3.0 local."
                ),
                "citations": [],
                "confidence": 0.0,
                "error_code": "web_unavailable",
            }

        return {
            "ok": False,
            "output": "No hay proveedor web configurado.",
            "citations": [],
            "confidence": 0.0,
            "error_code": "web_provider_missing",
        }

    def _skill_project_plan(
        self,
        *,
        goal: str,
    ) -> Dict[str, Any]:
        if self._project_planner is None:
            return {
                "ok": False,
                "output": "No hay planner de proyecto conectado.",
                "citations": [],
                "confidence": 0.0,
                "error_code": "planner_unavailable",
            }

        try:
            plan = self._project_planner(goal)
        except Exception as exc:
            return {
                "ok": False,
                "output": f"Error del planner: {exc}",
                "citations": [],
                "confidence": 0.0,
                "error_code": "planner_error",
            }

        subgoals = []
        if isinstance(plan, dict):
            subgoals = list(plan.get("subgoals") or [])
        elif isinstance(plan, list):
            subgoals = list(plan)

        clean = []
        for item in subgoals[:6]:
            if isinstance(item, dict):
                desc = str(item.get("desc") or "").strip()
                done_when = str(
                    item.get("done_when")
                    or "sub-tarea completada"
                ).strip()
            else:
                desc = str(item).strip()
                done_when = "sub-tarea completada"

            if desc:
                clean.append(
                    {
                        "desc": desc,
                        "done_when": done_when,
                    }
                )

        if not clean:
            return {
                "ok": False,
                "output": "El planner no produjo subtareas utilizables.",
                "citations": [],
                "confidence": 0.0,
                "error_code": "empty_plan",
            }

        return {
            "ok": True,
            "output": {"subgoals": clean},
            "citations": [],
            "confidence": 0.88,
            "raw": plan,
        }

    def _skill_python_exec(
        self,
        *,
        code: str,
    ) -> Dict[str, Any]:
        expr_clean = self._extract_math(code)
        has_op = any(op in expr_clean for op in "+-*/%")
        if "**" in expr_clean:
            has_op = True

        if not has_op:
            return {
                "ok": False,
                "output": "No se detectó una expresión matemática.",
                "citations": [],
                "confidence": 0.0,
                "error_code": "not_math",
            }

        allowed = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "log": math.log,
            "exp": math.exp,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
            "floor": math.floor,
            "ceil": math.ceil,
        }

        class Safe(ast.NodeTransformer):
            ALLOWED = (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Num,
                ast.Constant,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.Mod,
                ast.Pow,
                ast.USub,
                ast.UAdd,
                ast.Call,
                ast.Name,
                ast.Load,
                ast.Tuple,
                ast.List,
            )

            def generic_visit(self, node: ast.AST) -> Any:
                if not isinstance(node, self.ALLOWED):
                    raise ValueError(
                        f"Nodo no permitido: {type(node).__name__}"
                    )
                return super().generic_visit(node)

        try:
            tree = ast.parse(expr_clean, mode="eval")
            Safe().visit(tree)

            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id not in allowed:
                    raise ValueError(
                        f"Nombre no permitido: {node.id}"
                    )

            value = eval(
                compile(tree, "<quipu-math>", "eval"),
                {"__builtins__": {}},
                allowed,
            )

            return {
                "ok": True,
                "output": str(value),
                "citations": [],
                "confidence": 1.0,
            }
        except Exception as exc:
            return {
                "ok": False,
                "output": f"Error de cálculo: {exc}",
                "citations": [],
                "confidence": 0.0,
                "error_code": "math_error",
            }

    def _skill_filesystem_read(
        self,
        *,
        path: Optional[str],
        encoding: str = "utf-8",
        max_bytes: int = 2_000_000,
    ) -> Dict[str, Any]:
        if not path:
            return {
                "ok": False,
                "output": "No encontré una ruta de archivo.",
                "citations": [],
                "confidence": 0.0,
                "error_code": "path_missing",
            }

        candidate = os.path.abspath(
            path
            if os.path.isabs(path)
            else os.path.join(self._fs_root, path)
        )

        try:
            common = os.path.commonpath(
                [self._fs_root, candidate]
            )
        except Exception:
            common = ""

        if common != self._fs_root:
            return {
                "ok": False,
                "output": "Ruta fuera de la zona permitida.",
                "citations": [],
                "confidence": 0.0,
                "error_code": "path_outside_root",
            }

        if not os.path.isfile(candidate):
            return {
                "ok": False,
                "output": "Archivo no encontrado.",
                "citations": [],
                "confidence": 0.0,
                "error_code": "file_not_found",
            }

        size = os.path.getsize(candidate)
        if size > max_bytes:
            return {
                "ok": False,
                "output": "Archivo demasiado grande.",
                "citations": [],
                "confidence": 0.0,
                "error_code": "file_too_large",
            }

        with open(
            candidate,
            "r",
            encoding=encoding,
            errors="ignore",
        ) as handle:
            data = handle.read(min(size, 64_000))

        return {
            "ok": True,
            "output": data,
            "citations": [
                {
                    "id": f"file://{candidate}",
                    "score": 1.0,
                }
            ],
            "confidence": 1.0,
            "raw": {
                "path": candidate,
                "size": size,
            },
        }
