# -*- coding: utf-8 -*-
"""
Autocorrección verificable para QuipuLoop.

Cubre:
- Reglas por tipo de tarea (QA-RAG, general).
- Verificación base (longitud, frases prohibidas, puntuación).
- Reglas RAG: exigir citas si hubo hits y validar score mínimo.
- Segundo pase opcional (LLM-as-judge) para reescritura si falla la verificación.
- Plan de reintentos (sugerencias operativas para el loop/usuario).

Requisito del adaptador LLM opcional (módulo 'llama_cpp_adapter'):
    generate(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional, Protocol, cast
import importlib

# Settings robusto (funciona si ejecutas como paquete o script)
try:
    from .settings import VERIFY_CONF, MODEL_CONF, RAG_SCORE_THRESHOLD
except Exception:  # pragma: no cover
    try:
        from settings import VERIFY_CONF, MODEL_CONF, RAG_SCORE_THRESHOLD  # type: ignore
    except Exception:
        from root.settings import VERIFY_CONF, MODEL_CONF, RAG_SCORE_THRESHOLD  # type: ignore

# Para anexar citas si faltan
from rag.retriever import citations as rag_citations


# --------------------------------------------------------------------------- #
# Tipos / utilidades

class LLMAdapter(Protocol):
    MODEL_NAME: str
    def generate(self, prompt: str, max_tokens: int = ..., temperature: float = ...) -> str: ...

def _load_adapter() -> Optional[LLMAdapter]:
    try:
        mod = importlib.import_module("llama_cpp_adapter")
        return cast(LLMAdapter, mod)
    except Exception:
        return None

def _task_type(context: Dict[str, Any], output: str) -> str:
    """Heurística mínima de tipo de tarea."""
    if (context.get("rag_hits") or []) or "Fuentes:" in (output or ""):
        return "qa_rag"
    return "general"


# --------------------------------------------------------------------------- #
# Reglas

def _rule_min_len(output: str, ctx: Dict[str, Any], issues: List[str], suggestions: List[str]):
    min_chars = int(VERIFY_CONF.get("min_chars", 0))
    if len((output or "").strip()) < min_chars:
        issues.append(f"Salida demasiado corta (<{min_chars} caracteres).")
        suggestions.append("Amplía con pasos accionables y un ejemplo breve (si aplica).")

def _rule_banned(output: str, ctx: Dict[str, Any], issues: List[str], suggestions: List[str]):
    lower = (output or "").lower()
    for bad in VERIFY_CONF.get("banned_phrases", []):
        if bad and bad in lower:
            issues.append(f"Frase prohibida detectada: «{bad}».")
            suggestions.append("Evita lenguaje derrotista; ofrece alternativas o rutas de solución.")

def _rule_punctuation(output: str, ctx: Dict[str, Any], issues: List[str], suggestions: List[str]):
    if VERIFY_CONF.get("must_end_with_punctuation", False):
        s = (output or "").rstrip()
        if s and s[-1] not in ".!?…)”»]":
            issues.append("La respuesta no termina con puntuación.")
            suggestions.append("Añade un cierre gramatical (punto u otro signo adecuado).")

def _rule_rag_citations(output: str, ctx: Dict[str, Any], issues: List[str], suggestions: List[str]):
    """Si hubo hits de RAG, exigir al menos N citas y score >= τ."""
    hits = ctx.get("rag_hits") or []
    if not hits:
        return
    need_cites = bool(VERIFY_CONF.get("require_citation_on_rag", True))
    min_cites = int(VERIFY_CONF.get("min_rag_citations", 1))
    min_score = float(VERIFY_CONF.get("min_rag_score", RAG_SCORE_THRESHOLD))
    # ¿El texto contiene la sección de citas?
    has_cites_block = "Fuentes:" in (output or "")
    # ¿Hay suficientes hits y con score aceptable?
    valid_hits = [h for h in hits if float(h.get("score", 0.0)) >= min_score]
    if need_cites and (not has_cites_block or len(valid_hits) < min_cites):
        issues.append(f"Faltan citas RAG suficientes (≥{min_cites}) o score < {min_score:.2f}.")
        suggestions.append("Incluye 'Fuentes:' con al menos una cita relevante y ajusta el umbral si es necesario.")

def _rules_for(task: str):
    base = [_rule_min_len, _rule_banned, _rule_punctuation]
    if task == "qa_rag":
        base.append(_rule_rag_citations)
    return base


# --------------------------------------------------------------------------- #
# Verificación

def verify(output: str, context: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    suggestions: List[str] = []
    ttype = _task_type(context, output)
    for rule in _rules_for(ttype):
        rule(output, context, issues, suggestions)
    return {"passed": len(issues) == 0, "issues": issues, "suggestions": suggestions, "task_type": ttype}


# --------------------------------------------------------------------------- #
# Autocorrección

def _ensure_punctuation(text: str) -> str:
    s = (text or "").rstrip()
    if s and s[-1] not in ".!?…)”»]":
        s += "."
    return s

def _ensure_citations(text: str, ctx: Dict[str, Any]) -> str:
    hits = ctx.get("rag_hits") or []
    if not hits:
        return text
    if "Fuentes:" in (text or ""):
        return text
    cites = rag_citations(hits)
    if cites:
        return f"{text}\n\nFuentes:\n{cites}"
    return text

def _fallback_patch(original: str, suggestions: List[str], ctx: Dict[str, Any]) -> str:
    """Parche básico sin LLM."""
    text = _ensure_punctuation(original)
    text = _ensure_citations(text, ctx)
    if suggestions:
        text += "\n\nNotas de mejora aplicadas:\n- " + "\n- ".join(suggestions)
    return text

def attempt_self_correction(draft: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Intenta reescritura con LLM si hay adaptador; si no, aplica parche básico.
    """
    first = verify(draft, context)
    if first["passed"]:
        return {"output": draft, "used_model": False, "trace": ["Sin cambios: verificación aprobada."]}

    adapter = _load_adapter()
    if adapter is None or not hasattr(adapter, "generate"):
        return {
            "output": _fallback_patch(draft, first["suggestions"], context),
            "used_model": False,
            "trace": ["Sin adaptador disponible. Parche básico aplicado."]
        }

    prompt = (
        "Arregla la siguiente respuesta para cumplir las reglas:\n"
        f"- Tipo de tarea: {first['task_type']}\n"
        f"- Issues: {first['issues']}\n"
        f"- Sugerencias: {first['suggestions']}\n\n"
        "RESPUESTA ORIGINAL:\n"
        f"{draft}\n\n"
        "Devuelve solo la versión corregida, clara y completa en español. Si hubo RAG, añade 'Fuentes:' con las citas."
    )
    try:
        new_text = adapter.generate(  # type: ignore[arg-type]
            prompt=prompt,
            max_tokens=int(MODEL_CONF.get("max_tokens", 512)),
            temperature=float(MODEL_CONF.get("temperature", 0.7)),
        )
        # Asegura reglas mínimas por si el modelo no lo cumple al 100%
        new_text = _ensure_punctuation(new_text)
        new_text = _ensure_citations(new_text, context)
        return {"output": new_text, "used_model": True, "trace": ["Reescritura con modelo aplicada."]}
    except Exception as e:  # degradación segura
        return {
            "output": _fallback_patch(draft, first["suggestions"], context),
            "used_model": False,
            "trace": [f"Fallo del adaptador: {e}. Parche básico aplicado."]
        }


# --------------------------------------------------------------------------- #
# Plan de reintentos

def retry_plan(context: Dict[str, Any], verify_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Sugerencias operativas para un segundo intento (no ejecuta nada).
    """
    plan: List[Dict[str, Any]] = []
    ttype = verify_result.get("task_type", "general")
    issues = " ".join(verify_result.get("issues") or [])
    rag_hits = context.get("rag_hits") or []
    min_score = context.get("min_score")

    if ttype == "qa_rag":
        if not rag_hits:
            plan.append({"action": "reindex_or_broaden",
                         "hint": "Reindexa la carpeta correcta o intenta una consulta más específica."})
        elif "citas" in issues.lower():
            plan.append({"action": "add_citations",
                         "hint": "Asegura 'Fuentes:' con al menos una cita y score >= min_score."})
        if min_score is not None:
            plan.append({"action": "adjust_min_score",
                         "hint": f"Prueba con min_score ligeramente menor (p.ej., {float(min_score)*0.8:.2f})."})

    if "demasiado corta" in issues.lower():
        plan.append({"action": "expand_answer", "hint": "Añade pasos y un ejemplo breve."})

    return plan


# --------------------------------------------------------------------------- #
# CLI de prueba rápida

if __name__ == "__main__":  # pragma: no cover
    demo = "Ok"
    ctx = {"rag_hits": [], "min_score": RAG_SCORE_THRESHOLD}
    vr = verify(demo, ctx)
    print("[VERIFY]", vr)
    ac = attempt_self_correction(demo, ctx)
    print("\n[AUTOCORRECT used_model=", ac["used_model"], "]")
    print(ac["output"])
