# agents/agent_loop.py
# Agent loop mínimo: plan → (tool) → observe → (opcional) reflect → final.
# Compatible con tu API FastAPI en 127.0.0.1:8010 (llama_server.py).

from __future__ import annotations
import os, re, json, time, math, requests
from typing import Any, Dict, List, Callable, Optional

API      = os.getenv("AGI_API", "http://127.0.0.1:8010")
STOP     = ["<|end_of_turn|>", "</s>"]
DEBUG    = True  # traza por consola

# Soporte opcional de API Key (si en el server pusiste REQUIRE_API_KEY=1)
API_KEY  = os.getenv("DEMO_KEY", "")
HEADERS  = {"Content-Type": "application/json", **({"X-API-Key": API_KEY} if API_KEY else {})}

# =================== Herramientas ===================
def tool_now(_: Dict[str, Any]) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def tool_calc(args: Dict[str, Any]) -> str:
    expr = str(args.get("expression", "")).strip()
    if not expr or any(c not in "0123456789+-*/(). " for c in expr):
        return "error: expresión inválida"
    try:
        return str(eval(expr, {"__builtins__": {}}, {"math": math}))
    except Exception as e:
        return f"error: {e}"

def tool_mem_search(args: Dict[str, Any]) -> str:
    q = str(args.get("q", ""))
    k = int(args.get("k", 3))
    r = requests.post(f"{API}/memory/semantic/search", headers=HEADERS, json={"q": q, "k": k}, timeout=15)
    r.raise_for_status()
    hits = [x.get("text","") for x in r.json().get("results", [])]
    return " | ".join(hits) if hits else "(sin resultados)"

def tool_mem_upsert(args: Dict[str, Any]) -> str:
    text = str(args.get("text","")).strip()
    tags = args.get("tags") or []
    ttl_days = int(args.get("ttl_days", 0))
    if not text:
        return "error: texto vacío"
    r = requests.post(f"{API}/memory/semantic/upsert", headers=HEADERS,
                      json={"facts":[{"text":text,"tags":tags,"ttl_days":ttl_days}]}, timeout=15)
    r.raise_for_status()
    return "ok"

TOOLS: Dict[str, Dict[str, Any]] = {
    "now":        {"desc": "Devuelve fecha-hora local.", "schema": {"type":"object","properties":{}}},
    "calc":       {"desc": "Evalúa expresión aritmética segura.", "schema":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}},
    "mem_search": {"desc": "Busca facts en memoria semántica.", "schema":{"type":"object","properties":{"q":{"type":"string"},"k":{"type":"integer"}},"required":["q"]}},
    "mem_upsert": {"desc": "Guarda fact con tags (TTL opcional).", "schema":{"type":"object","properties":{"text":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}},"ttl_days":{"type":"integer"}},"required":["text"]}},
}
TOOL_IMPL: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "now": tool_now, "calc": tool_calc, "mem_search": tool_mem_search, "mem_upsert": tool_mem_upsert,
}

# =================== LLM call ===================
def call_llm(messages: List[Dict[str,str]], max_new_tokens=110, temperature=0.55, stream=False) -> str:
    payload = {
        "messages": messages,
        "params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.9, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.08,
            "stop": STOP, "stream": bool(stream)
        }
    }
    r = requests.post(f"{API}/chat", headers=HEADERS, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    return (data.get("text") or data.get("content") or "").strip()

# =================== Prompts ===================
SYS = (
"Tú eres un agente técnico conciso y accionable.\n"
"Trabaja en pasos: piensa, decide si usar herramienta, observa, itera.\n"
"Si necesitas herramienta, responde SOLO con JSON:\n"
'  {\"tool\":\"<nombre>\",\"args\":{...}}\n'
"Si ya puedes responder al usuario, responde SOLO con JSON:\n"
'  {\"final\":\"<respuesta breve y clara>\"}\n'
"Nada de texto fuera del JSON."
)

def tool_catalog_str() -> str:
    lines = []
    for name, spec in TOOLS.items():
        lines.append(f"- {name}: {spec['desc']} (schema: {json.dumps(spec['schema'])})")
    return "\n".join(lines)

REFLECT_SYS = (
"Tú eres un revisor crítico. Si el borrador es confuso, incorrecto o incompleto,\n"
"reescríbelo en ≤ 5 viñetas, directo al grano."
)

# =================== Utilidades ===================
def extract_json(s: str) -> Optional[dict]:
    """Intenta parsear JSON; si viene con texto extra o ```bloques```, recorta."""
    s = s.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.IGNORECASE|re.MULTILINE).strip()
    try:
        return json.loads(s)
    except Exception:
        # heurística: toma el primer {...} balanceado
        start = s.find("{")
        end   = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                return None
    return None

def reflect_quick(draft: str, observations: List[str], max_tokens: int = 60) -> str:
    msgs = [
        {"role":"system","content": REFLECT_SYS},
        {"role":"user","content": f"Borrador: {draft}\nObservaciones: {' | '.join(observations[-3:]) if observations else '(ninguna)'}"}
    ]
    out = call_llm(msgs, max_new_tokens=max_tokens, temperature=0.4, stream=False)
    return out or draft

# =================== Agent loop ===================
def run_goal(
    goal: str,
    max_steps: int = 6,
    reflect_tokens: int = 60,
    timebox_sec: int = 90,
    *,
    llm_call: Optional[
        Callable[[List[Dict[str, str]], int, float, bool], str]
    ] = None,
    tool_impl: Optional[
        Dict[str, Callable[[Dict[str, Any]], str]]
    ] = None,
    tool_specs: Optional[
        Dict[str, Dict[str, Any]]
    ] = None,
    debug: Optional[bool] = None,
) -> str:
    """
    Agent loop 3.0-compatible.

    Sin inyección conserva el comportamiento legacy.
    Con llm_call/tool_impl puede ser ejecutado desde Quipu sin depender
    del endpoint 127.0.0.1:8010.
    """
    t0 = time.time()
    active_tools = (
        tool_impl
        if tool_impl is not None
        else TOOL_IMPL
    )
    active_specs = (
        tool_specs
        if tool_specs is not None
        else TOOLS
    )
    debug_enabled = (
        DEBUG
        if debug is None
        else bool(debug)
    )

    def call_model(
        messages: List[
            Dict[str, str]
        ],
        max_tokens: int,
        temperature: float,
    ) -> str:
        if llm_call is None:
            return call_llm(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )

        return llm_call(
            messages,
            max_tokens,
            temperature,
            False,
        )

    catalog_lines = []

    for name, spec in (
        active_specs.items()
    ):
        catalog_lines.append(
            f"- {name}: "
            f"{spec.get('desc', '')} "
            f"(schema: "
            f"{json.dumps(spec.get('schema') or {})})"
        )

    messages: List[
        Dict[str, str]
    ] = [
        {
            "role": "system",
            "content": SYS,
        },
        {
            "role": "system",
            "content":
                "Herramientas disponibles:\n"
                + "\n".join(
                    catalog_lines
                ),
        },
        {
            "role": "user",
            "content":
                f"Meta: {goal}",
        },
    ]

    observations: List[str] = []

    for step in range(
        1,
        max_steps + 1,
    ):
        if (
            time.time() - t0
            > timebox_sec
        ):
            if debug_enabled:
                print(
                    "[timebox] agotado"
                )
            break

        response = call_model(
            messages,
            110,
            0.55,
        )
        obj = extract_json(
            response
        )

        if obj is None:
            if debug_enabled:
                print(
                    "[warn] respuesta no JSON"
                )
            return response.strip()

        if "tool" in obj:
            name = str(
                obj.get(
                    "tool"
                )
                or ""
            )
            args = obj.get(
                "args"
            ) or {}

            if debug_enabled:
                print(
                    f"[tool] {name} "
                    f"args={args}"
                )

            if name not in active_tools:
                observation = (
                    f"[error] herramienta "
                    f"'{name}' desconocida"
                )
            else:
                try:
                    output = (
                        active_tools[
                            name
                        ](
                            args
                        )
                    )
                    observation = (
                        f"[{name}] "
                        f"{str(output)[:600]}"
                    )
                except Exception as exc:
                    observation = (
                        f"[error] excepción "
                        f"en {name}: {exc}"
                    )

            observations.append(
                observation
            )
            messages.append(
                {
                    "role":
                        "assistant",
                    "content":
                        json.dumps(
                            obj,
                            ensure_ascii=False,
                        ),
                }
            )
            messages.append(
                {
                    "role":
                        "user",
                    "content":
                        "Observación: "
                        + observation,
                }
            )
            continue

        if "final" in obj:
            return str(
                obj["final"]
            ).strip()

        return str(
            obj
        ).strip()

    if observations:
        return (
            " | ".join(
                observations[-3:]
            )
        )

    return "(sin final)"

# =================== CLI ===================
if __name__ == "__main__":
    import sys
    goal = ("1) dime la hora; 2) calcula (45+55)/2; 3) guarda el fact 'prefiere respuestas concisas' con tag 'pref'; "
            "4) busca en memoria 'concisas' y devuelve JSON {now, calc, mem_hits}.")
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])
    print(run_goal(goal))
