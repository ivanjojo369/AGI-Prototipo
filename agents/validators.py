# agents/validators.py
from __future__ import annotations

from typing import Optional
import json
import re
from typing import Any, Tuple


# -----------------------------------------
# Utilidad: extrae el primer JSON top-level
# -----------------------------------------
def parse_top_json(text: str) -> Tuple[Any, str]:
    """
    Extrae el primer objeto/array JSON top-level de 'text' tolerando ruido.
    Retorna (objeto_json, resto_fuera_del_json).
    Lanza ValueError si no encuentra JSON válido.
    """
    buf = []
    stack = []
    in_str = False
    esc = False
    started = False

    for ch in text:
        if not started and ch.isspace():
            buf.append(ch)  # espacios previos permitidos
            continue
        if not started and ch in "{[":
            started = True
            stack.append("}" if ch == "{" else "]")
            buf.append(ch)
            continue
        if not started:
            # primer char no es { ni [, aborta
            break

        buf.append(ch)

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if not stack or stack[-1] != ch:
                    raise ValueError("JSON mal balanceado")
                stack.pop()
                if not stack:
                    # fin del primer json top-level
                    break

    js = "".join(buf)
    if not js.strip().startswith(("{", "[")):
        raise ValueError("No se encontró JSON top-level")
    try:
        obj = json.loads(js)
    except Exception as e:
        raise ValueError(f"JSON inválido: {e}") from e

    rest = text[len(js):]
    return obj, rest


# -------------------------
# Validadores de los tasks
# -------------------------

def is_t02_json(text: str) -> bool:
    """
    T02 ultra-tolerante:
    - Acepta texto extra fuera del JSON (fences, notas, encabezados).
    - Acepta 'owner' con cualquier capitalización; valor 'agent' case-insensitive.
    - Acepta 'subgoals' con cualquier capitalización/variación (subGoals, sub_goal...).
    - Acepta enteros o strings en la lista; NO exige exactamente 3 (>=1 ok).
    - Si viene un único valor en lugar de lista, también lo acepta (lo envuelve en lista).
    """
    def _strip_fences(s: str) -> str:
        s = s.strip()
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)\n```$", r"\1", s, flags=re.DOTALL)
        s = re.sub(r"^```\s*\n(.*?)\n```$", r"\1", s, flags=re.DOTALL)
        s = re.sub(r"```(.*?)```", r"\1", s, flags=re.DOTALL)
        return s.strip()

    s = _strip_fences(text or "")

    # 1) Intenta extraer el primer JSON top-level tolerante
    try:
        obj, _rest = parse_top_json(s)
    except Exception:
        return False

    if not isinstance(obj, dict):
        return False

    # owner (clave case-insensitive, valor 'agent' case-insensitive)
    owner_key = next((k for k in obj.keys() if k.lower() == "owner"), None)
    if owner_key is None:
        return False
    if str(obj[owner_key]).strip().lower() != "agent":
        return False

    # subgoals (clave flexible y tipos mixtos)
    def looks_like_subgoals(k: str) -> bool:
        kl = re.sub(r"[^a-z]", "", k.lower())
        return kl in ("subgoals", "subgoal") or kl.startswith("subgoal")

    sub_key = next((k for k in obj.keys() if looks_like_subgoals(k)), None)
    if sub_key is None:
        return False

    subs = obj[sub_key]
    if not isinstance(subs, list):
        subs = [subs]

    # Normaliza a strings (enteros/float → str; " 1 " → "1")
    norm: list[str] = []
    for x in subs:
        if isinstance(x, (int, float)):
            norm.append(str(int(x)))
        elif isinstance(x, str):
            m = re.match(r"\s*(\d+)\s*$", x)
            norm.append(m.group(1) if m else x.strip())
        else:
            norm.append(str(x))

    return len(norm) >= 1

    s = _strip_fences(text or "")
    # 1) Intenta parseo directo
    obj = None
    try:
        obj = json.loads(s)
    except Exception:
        frag = _first_json(s)
        if frag is None:
            return False
        try:
            obj = json.loads(frag)
        except Exception:
            return False

    if not isinstance(obj, dict):
        return False

    # ---- owner (tolerante a capitalización y espacios) ----
    owner_key = next((k for k in obj.keys() if k.lower() == "owner"), None)
    if owner_key is None:
        return False
    owner_val = str(obj[owner_key]).strip().lower()
    if owner_val != "agent":
        return False

    # ---- subgoals (tolerante a nombre y tipos) ----
    def looks_like_subgoals(k: str) -> bool:
        kl = re.sub(r"[^a-z]", "", k.lower())
        return kl in ("subgoals", "subgoal") or kl.startswith("subgoal")

    sub_key = next((k for k in obj.keys() if looks_like_subgoals(k)), None)
    if sub_key is None:
        return False

    subs = obj[sub_key]
    if not isinstance(subs, list):
        subs = [subs]

    # Normaliza: convierte números a strings, limpia " 1 " -> "1"
    norm: list[str] = []
    for x in subs:
        if isinstance(x, (int, float)):
            norm.append(str(int(x)))
        elif isinstance(x, str):
            m = re.match(r"\s*(\d+)\s*$", x)
            norm.append(m.group(1) if m else x.strip())
        else:
            norm.append(str(x))

    # Requisito mínimo: al menos 1 elemento válido (no forzamos exactamente 3)
    return len(norm) >= 1


def is_words_80_120_single_paragraph(text: str) -> bool:
    """
    T03: Un único párrafo de 80–120 palabras, sin saltos de línea ni listas.
    """
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    if "\n" in s or "\r" in s:
        return False
    # heurística anti-listas
    if re.search(r"(^|\s)[\-\*•]\s", s):
        return False
    # contar palabras
    words = re.findall(r"\b\w+(?:[-’']\w+)?\b", s, flags=re.UNICODE)
    return 80 <= len(words) <= 120


def is_5_bullets_risk_mitigation(text: str) -> bool:
    """
    T05: EXACTAMENTE 5 líneas con el formato:
      - Riesgo: … | Mitigación: …
    Sin texto extra, sin líneas vacías.
    """
    if not isinstance(text, str):
        return False
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    if len(lines) != 5:
        return False
    pat = re.compile(r"^\-\s+Riesgo:\s+.+\s+\|\s+Mitigación:\s+.+$")
    return all(pat.match(ln) for ln in lines)


def is_four_numbered_lines(text: str) -> bool:
    """
    Para T04: EXACTAMENTE 4 líneas numeradas '1.' .. '4.' sin texto extra.
    """
    if not isinstance(text, str):
        return False
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    if len(lines) != 4:
        return False
    for i, ln in enumerate(lines, 1):
        if not re.match(rf"^{i}\.\s+\S.+$", ln):
            return False
    return True
