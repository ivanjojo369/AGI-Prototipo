# agents/agent.py (actualizado)
# --------------------------------------------------------------------------------------
# Agente definitivo: plan->retrieve->tool(optional)->generate->verify->(reflect->retry)
# - T02 tolerante (primer JSON válido)
# - Memoria semántica (vector + episodios)
# - Tool-use integrado (selector + ejecución + evidencia en contexto)
# - Citas [doc:ID] en el contexto para medir `used_fact`
# - Métricas de memoria (compute_and_log_memory_metrics)
# --------------------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

# --------------------------------- dependencias opcionales ---------------------------------
# Validadores (si existen). Se usan primero; si no, cae al T02 interno.
try:
    from agents.validators import validate_json_output as _validate_json_output  # type: ignore
except Exception:  # pragma: no cover
    _validate_json_output = None  # fallback interno

# Motor de reflexión opcional
try:
    from agents.reflection_engine import ReflectionEngine as _ReflectionEngine  # type: ignore
except Exception:  # pragma: no cover
    _ReflectionEngine = None

# Métricas de memoria (si existe el módulo que agregamos)
try:
    from memory.metrics import MemoryTrace, compute_and_log_memory_metrics  # type: ignore
except Exception:  # pragma: no cover
    MemoryTrace = None  # type: ignore
    compute_and_log_memory_metrics = None  # type: ignore

# Memoria semántica (la implementes como la implementes)
try:
    from memory.semantic_memory import SemanticMemory  # type: ignore
except Exception:  # pragma: no cover
    SemanticMemory = None  # type: ignore

# Herramientas opcionales usadas directamente (no selector)
try:
    from tools.datetime_tool import get_datetime as _tool_datetime  # type: ignore
except Exception:  # pragma: no cover
    _tool_datetime = None

try:
    from tools.calculator import calc as _tool_calc  # type: ignore
except Exception:  # pragma: no cover
    _tool_calc = None

# Selector de herramientas (opcional)
try:
    from tools.tool_selector import choose_tool, suggest_args, route_and_execute  # type: ignore
except Exception:  # pragma: no cover
    choose_tool = None  # type: ignore
    suggest_args = None  # type: ignore
    route_and_execute = None  # type: ignore

# HTTP client (requests si está, si no urllib)
try:
    import requests as _requests  # type: ignore
except Exception:  # pragma: no cover
    _requests = None
    import urllib.request as _urllib_request  # type: ignore
    import urllib.error as _urllib_error  # type: ignore


# ======================================= configuración =======================================

@dataclass
class AgentConfig:
    # HTTP/API
    api_url: str = "http://localhost:8010/chat"
    api_key: Optional[str] = None
    system_prompt: Optional[str] = None
    timeout: float = 30.0

    # muestreo
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: int = 512

    # control de flujo
    strict: bool = False
    mode: str = "auto"            # "auto" o "manual" (libre)
    max_steps: int = 3

    # reintentos de red
    retry_http: int = 1
    retry_backoff: float = 0.5

    # control lógico y reflexión
    retry_count: int = 0          # reintento lógico si verify falla (stop-loss)
    reflect_budget: int = 0       # nº de rondas de reflexión (40–60 tks recomendadas)

    # métricas de memoria
    metrics_enabled: bool = True
    metrics_out_dir: str = "logs/benchmarks"
    metrics_k: int = 5            # k para retrieval@k (no es sampling top-k)

    # tool-use
    tool_threshold: float = 0.35  # umbral mínimo para ejecutar herramienta


# ===================================== utilidades internas ====================================

def _extract_first_json(s: str) -> Optional[Any]:
    """
    T02 tolerante: encuentra el PRIMER objeto/array JSON balanceado en s y lo parsea.
    No usa extensiones de regex; implementa un parser balanceado simple con soporte de
    strings y escapes. Aplica pequeños fixes (comillas “inteligentes”, coma colgante).
    """
    if not s:
        return None

    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if ch in "{[":
            start = i
            stack = [ch]
            in_str = False
            esc = False
            j = i + 1
            while j < n:
                c = s[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c in "{[":
                        stack.append(c)
                    elif c in "}]":
                        if not stack:
                            break
                        o = stack.pop()
                        if (o == "{" and c != "}") or (o == "[" and c != "]"):
                            return None
                        if not stack:
                            candidate = s[start : j + 1]
                            try:
                                return json.loads(candidate)
                            except Exception:
                                fixed = (
                                    candidate.replace("“", '"')
                                    .replace("”", '"')
                                    .replace("’", "'")
                                )
                                fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
                                try:
                                    return json.loads(fixed)
                                except Exception:
                                    return None
                j += 1
            break
        i += 1
    return None


def _post_json(url: str, payload: Dict[str, Any], timeout: float, api_key: Optional[str],
               retry_http: int = 1, backoff: float = 0.5) -> Dict[str, Any]:
    """
    POST JSON con 'requests' o 'urllib' con pequeños reintentos. Devuelve dict.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    attempts = max(1, int(retry_http))
    delay = max(0.0, float(backoff))

    if _requests is not None:
        last_exc: Optional[Exception] = None
        for a in range(attempts):
            try:
                r = _requests.post(url, json=payload, headers=headers, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                if a < attempts - 1:
                    time.sleep(delay * (2 ** a))
        raise RuntimeError(f"HTTP error after {attempts} attempts: {last_exc!r}")

    # urllib fallback
    req = _urllib_request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST"
    )
    last_exc = None
    for a in range(attempts):
        try:
            with _urllib_request.urlopen(req, timeout=timeout) as resp:
                data = resp.read().decode("utf-8", "ignore")
                return json.loads(data)
        except _urllib_error.HTTPError as e:  # type: ignore
            last_exc = e
        except _urllib_error.URLError as e:   # type: ignore
            last_exc = e
        if a < attempts - 1:
            time.sleep(delay * (2 ** a))
    raise RuntimeError(f"HTTP error after {attempts} attempts: {last_exc!r}")


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


# ======================================== clase Agent ========================================

class Agent:
    """
    Agente principal.
    - HTTP al LLM (endpoint estilo llama.cpp u otro servidor local).
    - Memoria semántica opcional.
    - Verifica salida (strict=True exige JSON válido) con T02 tolerante.
    - Stop-loss: reflexión + reintento si corresponde.
    - Tool-use integrado (selector + ejecución + evidencia).
    - Métricas de memoria si memory/metrics.py está disponible.

    Interfaz pública:
        Agent(cfg).run(goal: str, **overrides) -> str
    """

    def __init__(self, cfg: AgentConfig) -> None:
        self.cfg = cfg
        self.tools: Dict[str, Callable[..., Any]] = {}
        self._runs = 0  # útil para compactar cada N corridas

        self._init_tools()
        self.memory = self._init_memory()
        self.reflector = self._init_reflector()

    def _init_tools(self) -> None:
        """Registro de herramientas (solo si existen)."""
        if _tool_datetime:
            self.tools["datetime"] = _tool_datetime
        if _tool_calc:
            self.tools["calculator"] = _tool_calc

    def _init_memory(self):
        if SemanticMemory is None:
            return None
        try:
            return SemanticMemory()  # adapta si tu constructor necesita args
        except Exception as e:
            logging.warning("No se pudo inicializar SemanticMemory: %s", e)
            return None

    def _init_reflector(self):
        if _ReflectionEngine is None:
            return None
        try:
            return _ReflectionEngine()
        except Exception as e:
            logging.warning("No se pudo inicializar ReflectionEngine: %s", e)
            return None

    # --------------------------------- loop principal ---------------------------------

    def run(self, goal: str, **kwargs) -> str:
        """
        Ejecuta el loop E2E.
        overrides: mode, strict, max_steps, retry_count, reflect_budget, gold_doc_ids
        """
        # Overrides por llamada
        mode = kwargs.get("mode", self.cfg.mode)
        strict = bool(kwargs.get("strict", self.cfg.strict))
        max_steps = int(kwargs.get("max_steps", self.cfg.max_steps))
        retry_count = int(kwargs.get("retry_count", self.cfg.retry_count))
        reflect_budget = int(kwargs.get("reflect_budget", self.cfg.reflect_budget))

        run_id = uuid.uuid4().hex[:12]
        self._runs += 1

        context_docs: List[str] = []
        top_ids: List[str] = []
        top_texts: List[str] = []
        top_scores: List[float] = []

        # 1) PLAN (simple)
        plan = f"Objetivo: {goal}\nResponde útil y preciso."
        if mode == "auto":
            plan += "\nSi necesitas pasos, enuméralos; si debes llamar herramientas, explica brevemente su uso."

        # 2) RETRIEVE (memoria semántica) — inyecta [doc:ID] en el contexto
        trace = None
        if self.memory is not None:
            try:
                results = None
                if hasattr(self.memory, "retrieve"):
                    results = self.memory.retrieve(goal, k=self.cfg.metrics_k)
                elif hasattr(self.memory, "search"):
                    results = self.memory.search(query=goal, k=self.cfg.metrics_k)
                elif hasattr(self.memory, "query"):
                    results = self.memory.query(goal, top_k=self.cfg.metrics_k)

                for r in _ensure_list(results):
                    rid = str(r.get("id") if isinstance(r, dict) else getattr(r, "id", None))
                    rtext = str(r.get("text") if isinstance(r, dict) else getattr(r, "text", ""))
                    rscore = float(r.get("score") if isinstance(r, dict) else getattr(r, "score", 0.0))
                    doc_id = rid if rid and rid != "None" else f"doc-{len(top_ids)+1}"
                    context_docs.append(f"[doc:{doc_id}] {rtext}")
                    top_ids.append(doc_id)
                    top_texts.append(rtext)
                    top_scores.append(rscore)
            except Exception as e:
                logging.warning("Fallo en retrieval de memoria: %s", e)

            if MemoryTrace is not None and top_texts:
                try:
                    trace = MemoryTrace(goal=goal)
                    trace.record_retrieval(ids=top_ids, texts=top_texts, scores=top_scores)
                except Exception as e:
                    logging.warning("No se pudo registrar la traza de memoria: %s", e)

        # 2.5) TOOL-USE (selección/ejecución + evidencia)
        selected_tool = None
        tool_reason, tool_score = "", 0.0
        if choose_tool is not None and route_and_execute is not None:
            try:
                selected_tool, tool_reason, tool_score = choose_tool(goal)
                if selected_tool and tool_score >= self.cfg.tool_threshold:
                    exec_res = route_and_execute(goal)
                    if exec_res.get("ok"):
                        snippet = f"[Tool:{exec_res['tool']}] args={exec_res.get('args')} result={exec_res.get('result')}"
                        context_docs.insert(0, snippet)
                        # Fast-path para deterministas
                        if exec_res['tool'] in ('calculator','datetime'):
                            return str(exec_res.get('result'))
            except Exception as e:
                logging.debug("Tool-use omitido por error: %s", e)

        # 3) GENERATE
        prompt = self._build_prompt(plan, context_docs)
        answer, raw = self._generate(prompt)

        # 4) VERIFY (T02/validadores) + STOP-LOSS (reflect + retry)
        ok, parsed = self._verify(answer, strict)
        retries_left = max(0, retry_count)
        reflections_left = max(0, reflect_budget)

        while not ok and (retries_left > 0 or reflections_left > 0):
            hint = ""
            if reflections_left > 0 and self.reflector is not None:
                try:
                    hint = self.reflector.reflect(
                        answer=answer,
                        goal=goal,
                        error="La salida no cumple el formato requerido." if strict else "Mejora claridad/estructura."
                    )
                    reflections_left -= 1
                except Exception as e:
                    logging.debug("Reflector falló: %s", e)

            prompt2 = self._build_prompt(plan + ("\n" + hint if hint else ""), context_docs)
            answer, raw = self._generate(prompt2)
            ok, parsed = self._verify(answer, strict)
            if not ok:
                retries_left -= 1
            else:
                break

        final_text = answer if not strict else (
            json.dumps(parsed, ensure_ascii=False) if parsed is not None else answer
        )

        # 5) MÉTRICAS DE MEMORIA
        if self.cfg.metrics_enabled and compute_and_log_memory_metrics and trace is not None:
            try:
                compute_and_log_memory_metrics(
                    run_id=run_id,
                    goal=goal,
                    answer=final_text,
                    trace=trace,
                    k=self.cfg.metrics_k,
                    gold_doc_ids=kwargs.get("gold_doc_ids"),
                    out_dir=self.cfg.metrics_out_dir,
                )
            except Exception as e:
                logging.warning("Error al registrar métricas de memoria: %s", e)

        # 6) EPISODIOS + COMPACTACIÓN
        try:
            if self.memory is not None:
                meta = {"run_id": run_id, "mode": mode}
                if selected_tool:
                    meta.update({"tool_pred": selected_tool, "tool_score": tool_score, "tool_reason": tool_reason})

                if hasattr(self.memory, "write_episode"):
                    self.memory.write_episode(
                        goal=goal,
                        answer=final_text,
                        retrieved_ids=top_ids,
                        meta=meta,
                    )
                elif hasattr(self.memory, "write"):
                    self.memory.write(
                        text=final_text,
                        meta={"goal": goal, "retrieved_ids": top_ids, **meta},
                    )

                if hasattr(self.memory, "maybe_compact"):
                    self.memory.maybe_compact(max_docs=200, topic_summarizer=True, budget_tokens=256)
                elif hasattr(self.memory, "compact") and (self._runs % 20 == 0):
                    self.memory.compact(strategy="topic", max_summary_tokens=256)
        except Exception as e:
            logging.warning("Memory write/compact failed: %s", e)

        return final_text

    # ----------------------------------- helpers -----------------------------------

    def _build_prompt(self, plan: str, context_docs: List[str]) -> str:
        sys = f"<<SYSTEM>> {self.cfg.system_prompt}\n" if self.cfg.system_prompt else ""
        ctx = ""
        if context_docs:
            ctx = "\n\n[Contexto recuperado]\n" + "\n---\n".join(context_docs[: self.cfg.metrics_k]) + "\n"
        instr = (
            "\n[Instrucciones]\n"
            "Responde útil y conciso. Si `strict=True`, devuelve un JSON VÁLIDO.\n"
            "Cuando uses información del contexto, **cita el doc** copiando la marca [doc:ID] junto al hecho correspondiente.\n"
            "Si usas herramientas, explica brevemente la llamada y el resultado.\n"
        )
        return sys + plan + ctx + instr

    def _generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        payload = {
            "prompt": prompt,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "top_k": self.cfg.top_k,
            "repeat_penalty": self.cfg.repeat_penalty,
            "max_new_tokens": self.cfg.max_new_tokens,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        resp = _post_json(
            self.cfg.api_url,
            payload,
            timeout=self.cfg.timeout,
            api_key=self.cfg.api_key,
            retry_http=self.cfg.retry_http,
            backoff=self.cfg.retry_backoff,
        )

        text = (
            resp.get("text")
            or resp.get("output")
            or resp.get("message")
            or resp.get("choices", [{}])[0].get("text", "")
        )
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        return text, resp

    def _verify(self, answer: str, strict: bool) -> Tuple[bool, Optional[Any]]:
        if not strict:
            return True, None

        # 1) validadores externos
        if _validate_json_output is not None:
            try:
                res = _validate_json_output(answer)
                if isinstance(res, tuple) and len(res) == 2:
                    return bool(res[0]), res[1]
                if res is not None:
                    return True, res
            except Exception:
                pass  # cae al T02 interno

        # 2) T02 interno
        parsed = _extract_first_json(answer)
        return (parsed is not None), parsed


# ===================================== AgentBackend =====================================

class AgentBackend:
    """
    Wrapper para integrarse con agi_initializer._ApiCfg (o equivalente).
    Expone .run(goal, **kwargs) y ajusta overrides comunes (mode, strict, etc.).
    """

    def __init__(self, api: Any) -> None:
        cfg = AgentConfig(
            api_url=getattr(api, "api_url", "http://127.0.0.1:8010/chat"),
            api_key=getattr(api, "api_key", None),
            system_prompt=getattr(api, "system_prompt", None),
            timeout=float(getattr(api, "timeout", 30.0)),
            temperature=getattr(api, "temperature", None),
            top_p=getattr(api, "top_p", None),
            max_new_tokens=int(getattr(api, "max_new_tokens", 512)),
        )
        self.agent = Agent(cfg)

    def run(self, goal: str, **kwargs) -> str:
        for k in ("mode", "strict", "max_steps", "retry_count", "reflect_budget"):
            if k in kwargs and hasattr(self.agent.cfg, k):
                setattr(self.agent.cfg, k, kwargs[k])
        return self.agent.run(goal, **kwargs)

# --------------------------------------------------------------------------------------
# fin de archivo
# --------------------------------------------------------------------------------------
