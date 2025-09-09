from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import import_module
from inspect import signature
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "ReflectionConfig",
    "generate_reflective_response",
    "critique_and_improve",
    "summarize_context",
]

# --------------------------- Config ---------------------------

@dataclass
class ReflectionConfig:
    max_tokens: int = 256
    temperature: float = 0.2
    model: Optional[str] = None

# --------------------- LLM loader (opcional) ------------------

def _load_llm_callable() -> Optional[Callable[..., Any]]:
    """
    Intenta encontrar una función de generación en adapters/model_interface.py.
    Nombres probables: generate_text, generate, infer, complete, chat.
    """
    try:
        mod = import_module("adapters.model_interface")
    except Exception:
        return None
    for name in ("generate_text", "generate", "infer", "complete", "chat"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None

def _call_with_filtered_args(fn: Callable[..., Any], **kwargs) -> Any:
    """Llama a fn pasando solo kwargs que su firma acepte (evita TypeError)."""
    params = set(signature(fn).parameters.keys())
    safe = {k: v for k, v in kwargs.items() if k in params}
    return fn(**safe)

_LLM_FN = _load_llm_callable()

# ---------------------- Utilidades de formato -----------------

def _bullets(items: List[str]) -> str:
    """Formatea una lista como viñetas '- ' separadas por nueva línea."""
    if not items:
        return "- (sin elementos)"
    return "- " + "\n- ".join(items)

# ------------------------ Fallback local ----------------------

def _simple_reflection(prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Fallback determinista y rápido si no hay LLM disponible.
    """
    prompt_clean = re.sub(r"\s+", " ", (prompt or "").strip())
    ctx_keys = ", ".join(sorted((context or {}).keys())) if context else "—"

    observations: List[str] = []
    if prompt_clean:
        observations.append(
            "Resumen: "
            + (prompt_clean[:180] + ("…" if len(prompt_clean) > 180 else ""))
        )
    if context:
        observations.append(f"Contexto: claves disponibles → {ctx_keys}")

    risks: List[str] = []
    if any(k in prompt_clean.lower() for k in ("ambig", "ambigu", "incierto")):
        risks.append("Ambigüedad detectada: especificar entradas, salidas y restricciones.")
    else:
        risks.append("Riesgo: supuestos no validados. Confirmar datos de entrada y criterios de éxito.")

    next_steps = [
        "Definir objetivo en una única frase verificable.",
        "Listar 3–5 criterios de aceptación medibles.",
        "Enumerar pasos atómicos (1 acción por paso) y dependencias.",
    ]

    return (
        "[REFLEXIÓN]\n"
        f"{_bullets(observations) if observations else 'Sin observaciones.'}\n\n"
        "[RIESGOS]\n"
        f"{_bullets(risks)}\n\n"
        "[PRÓXIMOS PASOS]\n"
        f"{_bullets(next_steps)}"
    )

# ---------------------- API principal ------------------------

def generate_reflective_response(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    max_tokens: int = 256,
    temperature: float = 0.2,
    model: Optional[str] = None,
    **kw: Any,
) -> str:
    """
    Genera una respuesta reflexiva. Usa un LLM de adapters/model_interface si existe;
    si no, recurre a un fallback determinista.
    """
    cfg = ReflectionConfig(max_tokens=max_tokens, temperature=temperature, model=model)

    if _LLM_FN is not None:
        try:
            text = _call_with_filtered_args(
                _LLM_FN,
                prompt=prompt,
                context=context,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                model=cfg.model,
                **kw,
            )
            if isinstance(text, dict):
                for key in ("text", "content", "output"):
                    if key in text and isinstance(text[key], str):
                        return text[key]
                return str(text)
            if isinstance(text, str):
                return text
        except Exception:
            pass  # cae al fallback

    return _simple_reflection(prompt, context)

# ------------------- Utilidades opcionales -------------------

def summarize_context(context: Optional[Dict[str, Any]]) -> str:
    if not context:
        return "Contexto vacío."
    keys = ", ".join(sorted(context.keys()))
    return f"Contexto con claves: {keys}"

def critique_and_improve(
    plan: List[Dict[str, Any]],
    goal: str,
    max_suggestions: int = 3,
    **kw: Any,
) -> Dict[str, Any]:
    """
    Heurística simple: verifica que cada paso tenga 'action' y sugiere mejoras mínimas.
    Devuelve {"comments": [...], "suggestions": [...], "improved_plan": plan}
    """
    comments: List[str] = []
    improved: List[Dict[str, Any]] = []

    for i, step in enumerate(plan or []):
        s = dict(step)
        if "action" not in s:
            s["action"] = s.get("task") or s.get("name") or f"step_{i+1}"
            comments.append(f"Paso {i+1}: faltaba 'action', se normalizó.")
        if "args" not in s:
            s["args"] = {}
        improved.append(s)

    if not comments:
        comments.append("Plan verificado: estructura mínima OK.")
    comments.append(f"Objetivo: {goal}")

    suggestions = [
        "Agregar criterios de éxito por paso.",
        "Marcar dependencias explícitas (requires=[...]).",
        "Estimar costos/latencias si aplica.",
    ][: max(1, max_suggestions)]

    return {"comments": comments, "suggestions": suggestions, "improved_plan": improved}
