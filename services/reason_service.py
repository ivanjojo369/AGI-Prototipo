# services/reason_service.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

def _last_user(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages or []):
        if (m or {}).get("role") == "user":
            return (m.get("content") or "").strip()
    return ""

def _avoid_echo(user: str, text: str) -> str:
    if not user:
        return text
    if text.strip().lower() == user.strip().lower():
        return f"{text}. Cuéntame un poco más para poder ayudarte mejor."
    return text

def _rule_based_chat(user: str) -> str:
    u = user.lower()
    if any(g in u for g in ["hola", "buenos dias", "buen día", "buenas"]):
        return "¡Hola! Estoy aquí. ¿Qué te gustaría lograr?"
    if u.endswith("?"):
        return "Buena pregunta. Puedo darte una explicación concisa o paso a paso, ¿cuál prefieres?"
    if len(u) <= 3:
        return "Te leo. Dime con más detalle qué necesitas."
    return "Entiendo. Dame un poco más de contexto (objetivo, restricciones, datos) para afinar la respuesta."

class ReplyService:
    """
    Servicio ligero de razonamiento:
      - Evita eco.
      - Usa orquestador si está disponible (modo 'reason').
      - Puede adjuntar citas de memoria vectorial si se provee un 'vector_provider'.
    """
    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        planner: Optional[Any] = None,
        curriculum: Optional[Any] = None,
        vector_provider: Optional[Any] = None,
        system_prompt: Optional[str] = None,
    ):
        self.orchestrator = orchestrator
        self.planner = planner
        self.curriculum = curriculum
        self.vector_provider = vector_provider
        self.system_prompt = system_prompt or (
            "Eres una AGI doméstica textual competente. Responde con precisión y brevedad, "
            "sin repetir literal al usuario y ofreciendo próximos pasos claros."
        )

    def reply(self, messages: List[Dict[str, Any]], mode: str = "reason", k: int = 3, **kwargs) -> Dict[str, Any]:
        user = _last_user(messages)
        if not user:
            return {"reply": "¿En qué te ayudo hoy?"}

        # Contexto vectorial opcional
        citations = []
        if callable(self.vector_provider):
            try:
                citations = self.vector_provider(user, k) or []
            except Exception as e:
                citations = [{"text": f"[vector error] {type(e).__name__}: {e}"}]

        # “simple”: social/chit-chat mínimo
        if (mode or "").lower() == "simple":
            out = _avoid_echo(user, _rule_based_chat(user))
            return {"reply": out, "citations": citations, "mode_used": "simple"}

        # “reason”: intenta orquestador si existe
        if self.orchestrator and hasattr(self.orchestrator, "execute"):
            try:
                ans = self.orchestrator.execute(user, k=k) or {}
                text = ans.get("answer") or ans.get("text") or str(ans)
                out = _avoid_echo(user, text)
                return {"reply": out, "citations": citations, "mode_used": "reason", "orchestrator": True}
            except Exception as e:
                fallback = (
                    f"Voy a desglosarlo en pasos. (Nota: orquestador falló: {type(e).__name__}). "
                    f"Primero, define el objetivo exacto; luego, recursos disponibles; por último, restricciones."
                )
                return {"reply": _avoid_echo(user, fallback), "citations": citations, "mode_used": "reason"}

        # Sin orquestador → plantilla razonada breve
        outline = [
            "1) Objetivo: entiende la meta del usuario.",
            "2) Plan: divide en 2–4 pasos ejecutables.",
            "3) Riesgos: anticipa bloqueos y mitigaciones.",
            "4) Siguiente acción: un paso concreto inmediato.",
        ]
        text = f"He entendido: «{user}».\n" + "\n".join(outline)
        return {"reply": _avoid_echo(user, text), "citations": citations, "mode_used": "reason"}

# Funciones comodín para compatibilidad con otros importadores
def generate_reply(*, messages: List[Dict[str, Any]], mode: str = "reason", k: int = 3, **kwargs) -> Dict[str, Any]:
    return ReplyService(
        orchestrator=kwargs.get("orchestrator"),
        planner=kwargs.get("planner"),
        curriculum=kwargs.get("curriculum"),
        vector_provider=kwargs.get("vector_provider"),
    ).reply(messages=messages, mode=mode, k=k)

def reply(*, messages: List[Dict[str, Any]], mode: str = "reason", k: int = 3, **kwargs) -> Dict[str, Any]:
    return generate_reply(messages=messages, mode=mode, k=k, **kwargs)
