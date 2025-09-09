# planner/htn_simple.py
from __future__ import annotations
import os, re, json, requests
from typing import List, Dict, Any

API      = os.getenv("AGI_API", "http://127.0.0.1:8010")
API_KEY  = os.getenv("DEMO_KEY", "")
HEADERS  = {"Content-Type": "application/json", **({"X-API-Key": API_KEY} if API_KEY else {})}
STOP     = ["<|end_of_turn|>", "</s>"]

SYS = (
    "Divide una meta en sub-tareas concisas, ordenadas y accionables.\n"
    "Reglas:\n"
    "- 2 a 5 sub-tareas máximo.\n"
    "- Cada sub-tarea: ≤ 14 palabras.\n"
    '- Devuelve SOLO JSON con esta forma exacta: {"subgoals":[{"desc":"...","done_when":"..."}]}\n'
    "- Nada fuera del JSON."
)

def _extract_json(s: str):
    """Devuelve un objeto JSON o None. Tolera fences y texto extra."""
    if not s:
        return None
    s = s.strip()

    # elimina triples comillas/fences en cualquier parte
    s = re.sub(r"```(?:json)?\s*|```", "", s, flags=re.IGNORECASE)

    # intento directo
    try:
        return json.loads(s)
    except Exception:
        pass

    # intenta primer '{' y último '}' (tolerante a prefijos/sufijos)
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        frag = s[i:j+1]
        try:
            return json.loads(frag)
        except Exception:
            # intenta limpiar comas colgantes simples
            frag2 = re.sub(r",\s*([}\]])", r"\1", frag)
            try:
                return json.loads(frag2)
            except Exception:
                return None
    return None

def _call_llm(messages: List[Dict[str, str]], max_tokens=160, temperature=0.4) -> str:
    payload = {
        "messages": messages,
        "params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "repeat_penalty": 1.08,
            "stop": STOP,
            "stream": False,
        },
    }
    r = requests.post(f"{API}/chat", headers=HEADERS, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    return (data.get("text") or data.get("content") or "").strip()

def plan(goal: str, max_subgoals: int = 5) -> Dict[str, List[Dict[str, str]]]:
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user",   "content": f"Meta: {goal}"},
    ]
    out = _call_llm(msgs, max_tokens=140, temperature=0.35)

    obj = _extract_json(out) or {}
    subs = obj.get("subgoals") or []

    clean: List[Dict[str, str]] = []
    for sg in subs[:max_subgoals]:
        desc = str(sg.get("desc", "")).strip()
        done = str(sg.get("done_when", "resultado obtenido")).strip()
        if not desc:
            continue
        words = desc.split()
        if len(words) > 18:
            desc = " ".join(words[:18])
        clean.append({"desc": desc, "done_when": done or "resultado obtenido"})

    # Fallback si el modelo no dio nada utilizable
    if not clean:
        parts = re.split(r"(?:\d+\)|\d+\.)|[;•\n\r]+", goal)
        parts = [p.strip() for p in parts if p and p.strip()]
        for p in parts[:max_subgoals]:
            clean.append({"desc": p, "done_when": "sub-tarea ejecutada"})
        if not clean:
            clean = [{"desc": "Analizar la meta y preparar pasos breves",
                      "done_when": "lista de pasos creada"}]

    return {"subgoals": clean}

if __name__ == "__main__":
    import sys
    g = "Implementar FAISS en memoria semántica, evaluar retrieval@k y documentar setup."
    if len(sys.argv) > 1:
        g = " ".join(sys.argv[1:])
    print(json.dumps(plan(g), ensure_ascii=False, indent=2))
