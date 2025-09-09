# benchmarks/benchmark_agent.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Optional

# ------------------------------------------------------------------------------------------
# RUTA RAÍZ (para importar agents/agent.py si existe)
# ------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Intentamos usar tu Agent unificado (backend preferido)
_HAVE_AGENT = False
Agent = AgentConfig = None  # type: ignore
try:
    from agents.agent import Agent, AgentConfig  # noqa: E402
    _HAVE_AGENT = True
except Exception:
    _HAVE_AGENT = False

# requests se usa en el backend HTTP directo (contingencia o forzado)
try:
    import requests  # noqa: E402
except Exception as e:
    # Solo imprescindible si se usa backend http. Si no está, lo avisamos más adelante.
    requests = None  # type: ignore


# ==========================================================================================
# Utilidades de robustez (coerción de texto, recorte, extracción de JSON top-level)
# ==========================================================================================

def _coerce_text(x: Any) -> str:
    """Convierte cualquier cosa en un texto legible y estable."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("output", "text", "content", "message"):
            if k in x:
                return _coerce_text(x[k])
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    if isinstance(x, list):
        return " ".join(_coerce_text(i) for i in x if i is not None)
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def snippet(s: Any, n: int = 160) -> str:
    t = _coerce_text(s).replace("\n", " ").strip()
    return t[:n] + ("…" if len(t) > n else "")


def parse_top_json(text: str) -> Optional[Any]:
    """
    Extrae el primer objeto/array JSON top-level de un texto potencialmente ruidoso.
    Combina la estrategia tolerante del runner alternativo con limpieza simple al final.
    """
    if not isinstance(text, str):
        text = _coerce_text(text)

    buf: List[str] = []
    stack: List[str] = []
    in_str = False
    esc = False
    start = -1

    def push(ch: str):
        nonlocal start
        if not stack:
            start = len(buf)
        stack.append(ch)

    def pop(ch: str) -> bool:
        if not stack:
            return False
        open_ch = stack.pop()
        return (open_ch == '{' and ch == '}') or (open_ch == '[' and ch == ']')

    for ch in text:
        buf.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch in '{[':
            push(ch)
        elif ch in '}]':
            if not pop(ch):
                stack.clear()
                start = -1
            elif not stack and start != -1:
                frag = "".join(buf[start:])
                try:
                    return json.loads(frag)
                except Exception:
                    # recorte simple hasta el último cierre visible
                    last_brace = max(frag.rfind('}'), frag.rfind(']'))
                    if last_brace != -1:
                        frag2 = frag[:last_brace+1]
                        try:
                            return json.loads(frag2)
                        except Exception:
                            pass
                start = -1
    return None


# ==========================================================================================
# Definición de tareas (T01..T05) y carga de prompts
# ==========================================================================================

@dataclass
class Task:
    task_id: str
    goal: str
    strict: bool  # si True reforzamos el formato y validación


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_tasks() -> List[Task]:
    """
    Prioridad de carga:
      1) <root>/benchmarks/golden_prompts.json
      2) <root>/golden_prompts.json
    Estructura esperada: [{"task_id":"T01","goal":"..."}, ...]
    Si no existen, usa fallback con metas afinadas.
    """
    candidates = [
        ROOT / "benchmarks" / "golden_prompts.json",
        ROOT / "golden_prompts.json",
    ]
    for fp in candidates:
        if fp.exists():
            arr = json.loads(fp.read_text(encoding="utf-8"))
            out: List[Task] = []
            for it in arr:
                tid = (it.get("task_id") or it.get("id") or "").strip()
                goal = (it.get("goal") or it.get("prompt") or "").strip()
                if tid and goal:
                    strict = any(s in goal.lower() for s in ["exactamente", "json", "4 líneas", "4 lineas"])
                    out.append(Task(tid, goal, strict))
            if out:
                return out

    # Fallbacks
    return [
        Task("T01", "Define Synchronexis en 3 frases claras.", strict=False),
        Task(
    "T02",
    "Devuelve un objeto JSON que contenga las claves: owner (igual a 'agent', sin importar mayúsculas) y subgoals (lista de elementos que pueden ser números o strings). Se evaluará el PRIMER JSON válido encontrado; se tolera texto fuera del objeto.",
    strict=False
),
        Task("T03", "Resume en 80-120 palabras la diferencia entre un plan HTN y un plan plano.", strict=False),
        Task("T04",
             "Propón 4 pruebas unitarias para validar un parser tolerante que extrae el top-level JSON de una respuesta ruidosa. "
             "SALIDA EXACTA: 4 líneas numeradas 1-4, sin texto extra. Cada línea: nombre_corto | assert: condición esperada.",
             strict=True),
        Task("T05",
             "Entrega 5 bullets de riesgos de LLM en producción y una mitigación por cada uno. "
             "Formato exacto: '- Riesgo: … | Mitigación: …'. Responde EXACTAMENTE con 5 bullets.",
             strict=True),
    ]


# ==========================================================================================
# Validadores para cada tarea
# ==========================================================================================

def v_T01(text: str) -> bool:
    # Heurística: al menos 3 oraciones (por puntos/exclamación/interrogación)
    s = _coerce_text(text).replace("\n", " ").strip()
    sentences = [p for p in re.split(r"[.!?]+", s) if p.strip()]
    return len(sentences) >= 3

def v_T02(text: str) -> bool:
    """
    Tolerante para T02:
    - Acepta owner con cualquier capitalización; valor 'agent' case-insensitive.
    - Acepta subgoals con variaciones de nombre (subgoals, subGoal, sub_goal...).
    - Acepta enteros o strings; no exige exactamente 3 (>=1 ok).
    - Tolera texto extra fuera del objeto si parse_top_json lo permite.
    """

    # 1) Primero intenta JSON puro (evita depender de parse_top_json)
    try:
        obj = json.loads(text)
    except Exception:
        obj = None

    # 2) Si no es JSON puro, intenta extraer el primer objeto top-level
    if obj is None:
        try:
            obj = parse_top_json(text)  # debe existir en el módulo
        except Exception:
            return False
        if isinstance(obj, tuple):  # por si devuelve (obj, rest)
            obj = obj[0]

    if not isinstance(obj, dict):
        return False

    # 3) owner (case-insensitive)
    owner_key = next((k for k in obj.keys() if str(k).lower() == "owner"), None)
    if owner_key is None:
        return False
    owner_val = str(obj[owner_key]).strip().lower()
    if owner_val != "agent":
        return False

    # 4) subgoals (nombre tolerante + tipos tolerantes)
    def looks_like_subgoals(k: str) -> bool:
        kl = re.sub(r"[^a-z]", "", str(k).lower())
        return kl in ("subgoals", "subgoal") or kl.startswith("subgoal")

    sub_key = next((k for k in obj.keys() if looks_like_subgoals(k)), None)
    if sub_key is None:
        return False

    subs = obj[sub_key]
    if not isinstance(subs, list):
        subs = [subs]

    # Normaliza a strings (pero no falla si eran ints)
    norm = []
    for x in subs:
        if isinstance(x, (int, float)):
            norm.append(str(int(x)))
        else:
            s = str(x).strip()
            m = re.match(r"^\s*(\d+)\s*$", s)
            norm.append(m.group(1) if m else s)

    # Requisito mínimo: al menos un subgoal
    return len(norm) >= 1

def v_T03(text: str) -> bool:
    s = _coerce_text(text)
    words = re.findall(r"\w+(?:'\w+)?", s, flags=re.UNICODE)
    return 80 <= len(words) <= 120


def v_T04(text: str) -> bool:
    # EXACTAMENTE 4 líneas: "1. nombre | assert: condición"
    s = _coerce_text(text).strip("\n")
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) != 4:
        return False
    pat = re.compile(r"^[1-4]\.\s+[^|]+?\s+\|\s+assert:\s+.+$", re.IGNORECASE)
    # Adicionalmente, checamos el orden 1..4
    for idx, ln in enumerate(lines, start=1):
        if not pat.match(ln):
            return False
        if not ln.lstrip().startswith(f"{idx}."):
            return False
    return True


def v_T05(text: str) -> bool:
    # EXACTAMENTE 5 bullets: "- Riesgo: … | Mitigación: …"
    s = _coerce_text(text).strip("\n")
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) != 5:
        return False
    pat = re.compile(r"^-+\s*Riesgo:\s+.+\s+\|\s+Mitigación:\s+.+$", re.IGNORECASE)
    return all(pat.match(ln) for ln in lines)


VALIDATORS: Dict[str, Callable[[str], bool]] = {
    "T01": v_T01,
    "T02": v_T02,
    "T03": v_T03,
    "T04": v_T04,
    "T05": v_T05,
}


# ==========================================================================================
# Construcción de mensajes (reglas reforzadas para tareas estrictas)
# ==========================================================================================

def build_messages(task: Task, prior_hint: Optional[str] = None) -> List[Dict[str, str]]:
    sys_base = "Eres un asistente útil. Responde exactamente lo pedido."
    usr_goal = task.goal
    if task.strict:
        rules = (
            "Sigue las reglas al pie de la letra. "
            "No agregues prefacios ni epílogos. "
            "No uses paréntesis, corchetes ni llaves fuera de lo pedido. "
            "Si se piden N líneas, responde EXACTAMENTE con esas N líneas, sin texto extra."
        )
        sys_base += " " + rules

    msgs = [
        {"role": "system", "content": sys_base},
        {"role": "user", "content": usr_goal},
    ]
    if prior_hint:
        msgs.append({"role": "user", "content": prior_hint})
    return msgs


# ==========================================================================================
# Backends: (A) Agent de tu proyecto, (B) HTTP directo /chat
# ==========================================================================================

@dataclass
class APIParams:
    api_url: str
    api_key: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: int = 110
    timeout_s: float = 60.0


class BackendBase:
    def run(self, messages: List[Dict[str, str]], api: APIParams) -> Tuple[str, int]:
        raise NotImplementedError

class AgentBackend(BackendBase):
    """Usa agents/agent.py si está disponible. Acepta APIParams opcional en __init__."""
    def __init__(self, api: APIParams | None = None):
        if not _HAVE_AGENT or Agent is None or AgentConfig is None:
            raise RuntimeError("Agent no disponible (agents/agent.py no encontrado o con errores)")
        self.api: APIParams | None = None
        self._agent = None
        if api is not None:
            self._agent = self._build_agent(api)
            self.api = api

    def _build_agent(self, api: APIParams):
        # Construye AgentConfig siendo tolerantes con 'timeout' vs 'timeout_s'
        kwargs = dict(
            api_url=api.api_url,
            api_key=api.api_key,
            temperature=api.temperature,
            top_p=api.top_p,
            repeat_penalty=api.repeat_penalty,
            top_k=api.top_k,
            max_new_tokens=api.max_new_tokens,
            max_steps=5,  # el loop interno del Agent puede ser <= que el del runner
        )
        try:
            cfg = AgentConfig(timeout=api.timeout_s, **kwargs)  # preferimos 'timeout'
        except TypeError:
            cfg = AgentConfig(timeout_s=api.timeout_s, **kwargs)  # compatibilidad
        return Agent(cfg)

    def run(self, messages: List[Dict[str, str]], api: APIParams) -> Tuple[str, int]:
        # Lazy build / rebuild si el APIParams cambió o aún no hay agente
        if self._agent is None or (self.api is None) or (
            (self.api.api_url != api.api_url) or (self.api.api_key != api.api_key)
            or (self.api.max_new_tokens != api.max_new_tokens)
            or (self.api.temperature != api.temperature) or (self.api.top_p != api.top_p)
            or (self.api.repeat_penalty != api.repeat_penalty) or (self.api.top_k != api.top_k)
            or (self.api.timeout_s != api.timeout_s)
        ):
            self._agent = self._build_agent(api)
            self.api = api

        # Nuestro Agent espera un goal (string). Concatenamos system+user.
        sys_txt = []
        usr_txt = []
        for m in messages:
            if m.get("role") == "system":
                sys_txt.append(m.get("content") or "")
            elif m.get("role") == "user":
                usr_txt.append(m.get("content") or "")
        goal = ("\n\n".join(sys_txt + usr_txt)).strip()

        t0 = time.perf_counter()
        res = self._agent.run(goal)  # {"status","output":{"text":...},"lat_ms":...}
        lat_ms = int((time.perf_counter() - t0) * 1000)
        if isinstance(res, dict) and isinstance(res.get("lat_ms"), int):
            lat_ms = int(res["lat_ms"])

        text = ""
        if isinstance(res, dict):
            out = res.get("output")
            if isinstance(out, dict):
                text = _coerce_text(out.get("text"))
            else:
                text = _coerce_text(out)
        else:
            text = _coerce_text(res)
        return text, lat_ms


class HTTPBackend(BackendBase):
    """Cliente HTTP directo a /chat (tolerante a distintas formas de respuesta)."""
    def __init__(self):
        if requests is None:
            raise RuntimeError("Necesitas 'requests' para backend http. Instala con: pip install requests")

    def run(self, messages: List[Dict[str, str]], api: APIParams) -> Tuple[str, int]:
        t0 = time.perf_counter()
        payload = {
            "messages": messages,
            "params": {
                "max_new_tokens": api.max_new_tokens,
                "temperature": api.temperature,
                "top_p": api.top_p,
                "stream": False,
            },
        }
        headers = {
            "X-API-Key": api.api_key,
            "Authorization": f"Bearer {api.api_key}",
            "Content-Type": "application/json",
        }
        r = requests.post(api.api_url, headers=headers, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), timeout=api.timeout_s)
        lat_ms = int((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        data = r.json()

        # Formas esperadas del servidor:
        # {"output": "..."} | {"output":{"text":"..."}} | {"text":"..."} | etc.
        if isinstance(data, dict):
            if "output" in data:
                return _coerce_text(data["output"]), lat_ms
            if "text" in data:
                return _coerce_text(data["text"]), lat_ms
            return _coerce_text(data), lat_ms
        return _coerce_text(data), lat_ms


# ==========================================================================================
# Ejecución de tareas con reintentos + validación
# ==========================================================================================

@dataclass
class RunResult:
    task_id: str
    goal: str
    status: str   # success|fail|error
    passed: int   # 1|0
    lat_ms: int
    steps: int
    text: str
    error: str


def run_task(task: Task, backend: BackendBase, api: APIParams, max_steps: int) -> RunResult:
    validator = VALIDATORS[task.task_id]
    steps = 0
    last_text = ""
    last_err = ""
    last_lat = 0

    hint = None
    while steps < max_steps:
        steps += 1
        try:
            messages = build_messages(task, prior_hint=hint)
            text, lat_ms = backend.run(messages, api)
            last_text = text
            last_lat = lat_ms
            if validator(text):
                return RunResult(task.task_id, task.goal, "success", 1, lat_ms, steps, text, "")
            # No pasó -> reforzamos y reintentamos
            hint = "(reintento) RESPONDE EXACTAMENTE; respeta formato / conteos / JSON requerido."
            if task.task_id.upper() == "T04":
                hint += " Devuelve EXACTAMENTE 4 líneas numeradas 1-4 con 'assert:' en cada línea, sin texto extra."
            elif task.task_id.upper() == "T05":
                hint += " Devuelve EXACTAMENTE 5 bullets con el formato '- Riesgo: … | Mitigación: …', sin nada más."
            elif task.task_id.upper() == "T02":
                hint += " Devuelve SOLO un JSON top-level válido con claves exactas: subgoals (3) y owner = 'agent'."
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            break

    status = "fail" if not last_err else "error"
    return RunResult(task.task_id, task.goal, status, 0, last_lat, steps, last_text, last_err)


# ==========================================================================================
# CLI principal
# ==========================================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark unificado T01–T05 (backend Agent o HTTP).")
    parser.add_argument("--api-url", default="http://127.0.0.1:8010/chat")
    parser.add_argument("--api-key", required=True)

    # Backend: agent (por defecto si existe) o http
    default_backend = "agent" if _HAVE_AGENT else "http"
    parser.add_argument("--backend", choices=["agent", "http"], default=default_backend,
                        help="Usa 'agent' (agents/agent.py) o 'http' (requests directo).")

    # Sampling / límites
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", dest="top_p", type=float, default=None)
    parser.add_argument("--repeat-penalty", dest="repeat_penalty", type=float, default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=110)
    parser.add_argument("--timeout", type=float, default=60.0)

    # Loop y selección
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--only", nargs="+", help="Ejecuta solo algunas tareas (p.ej. --only T04 T05)")

    # Salida
    parser.add_argument("--out", default=None,
                        help="Directorio salida (por defecto logs/benchmarks/agent_run_YYYYMMDD_HHMMSS)")

    args = parser.parse_args()

    out_dir = ROOT / "logs" / "benchmarks" / f"agent_run_{_now_stamp()}" if not args.out else Path(args.out)
    _ensure_dir(out_dir)
    results_csv = out_dir / "results.csv"
    infer_csv = out_dir / "infer_metrics.csv"

    tasks = load_tasks()
    if args.only:
        wanted = {t.upper() for t in args.only}
        tasks = [t for t in tasks if t.task_id.upper() in wanted]

    api = APIParams(
        api_url=args.api_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        timeout_s=args.timeout,
    )

    if args.backend == "agent":
        if not _HAVE_AGENT:
            print("[WARN] agents/agent.py no disponible. Cambiando a backend HTTP.")
            backend: BackendBase = HTTPBackend()
        else:
            backend = AgentBackend(api)
    else:
        backend = HTTPBackend()

    print(f"[Benchmark] Backend: {args.backend} | Tareas: {len(tasks)} | API: {args.api_url} | Out: {out_dir}")

    results: List[RunResult] = []
    for t in tasks:
        print(f"\n[{t.task_id}] {t.goal.splitlines()[0]}")
        r = run_task(t, backend, api, args.max_steps)
        results.append(r)
        print(f"{'✅' if r.passed else '❌'}  {r.status:7s}  {r.lat_ms:6d} ms  |  {snippet(r.text)}")
        if r.error:
            print(f"   ERROR: {r.error}")

    # CSVs
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "goal", "pass", "status", "lat_ms", "steps", "output_snippet", "error"])
        for r in results:
            w.writerow([r.task_id, r.goal.replace("\n", " "), r.passed, r.status, r.lat_ms, r.steps, snippet(r.text, 240), r.error])

    with infer_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "lat_ms"])
        for r in results:
            w.writerow([r.task_id, r.lat_ms])

    passed = sum(r.passed for r in results)
    avg_lat = int(sum(r.lat_ms for r in results) / max(1, len(results)))
    print(f"\n[Benchmark] PASS: {passed}/{len(results)} ({(100.0*passed/max(1,len(results))):.1f}%)  |  avg_lat: {avg_lat} ms")
    print("[Benchmark] CSVs:")
    print(f" - {results_csv}")
    print(f" - {infer_csv}")


if __name__ == "__main__":
    main()
