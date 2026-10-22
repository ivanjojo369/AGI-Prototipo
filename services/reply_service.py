# services/reply_service.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import inspect


class ReplyService:
    """
    Servicio de respuestas para el chat.
    - mode="simple": responde de forma mínima (placeholder) sin LLM.
    - mode="orchestrator": delega en un orquestador (tu AGI/LLM) si está disponible.

    Acepta parámetros opcionales para evitar TypeError si el caller
    pasa kwargs adicionales (p. ej. 'orchestrator').
    """

    def __init__(
        self,
        mode: str = "simple",
        llm: Any = None,
        retriever: Any = None,
        orchestrator: Any = None,
        **_: Any,  # Absorbe kwargs desconocidos para evitar TypeError
    ) -> None:
        self.mode = (mode or "simple").lower()
        self.llm = llm
        self.retriever = retriever
        self.orchestrator = orchestrator

    async def run(
        self,
        messages: List[Dict[str, str]],
        k: int = 3,
        node: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta la respuesta según el modo.
        Devuelve un dict JSON–serializable con las claves 'reply', 'text' y 'mode_used'.
        """
        try:
            if self.mode == "simple":
                text = self._simple_reply(messages)
            elif self.mode in ("orchestrator", "agi", "llm"):
                text = await self._orchestrator_reply(messages, k=k, node=node)
            else:
                text = f"[reply_service] Modo no soportado: {self.mode}. Usando 'simple'."
                text = self._simple_reply(messages)

            return {
                "reply": text,
                "text": text,
                "mode_used": self.mode,
            }
        except Exception as e:
            err = f"[reply_service error] {type(e).__name__}: {e}"
            return {
                "reply": err,
                "text": err,
                "mode_used": self.mode,
            }

    # ----------------- helpers -----------------

    def _simple_reply(self, messages: List[Dict[str, str]]) -> str:
        """Placeholder básico: toma el último mensaje del usuario y responde amigablemente."""
        user_last = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_last = m.get("content", "").strip()
                break

        if not user_last:
            return "¿En qué puedo ayudarte?"

        # Respuesta mínima (no eco literal)
        return f"Te escucho. Me dijiste: «{user_last}». ¿Quieres que haga algo con ello?"

    async def _orchestrator_reply(
        self, messages: List[Dict[str, str]], k: int = 3, node: Optional[str] = None
    ) -> str:
        """
        Invoca al orquestador si existe. Soporta:
        - métodos async: 'achat' o 'chat'
        - métodos sync: 'chat' que devuelvan str o dict
        """
        if self.orchestrator is None:
            return "[reply_service] No hay orquestador disponible."

        # Elegir método del orquestador
        fn = getattr(self.orchestrator, "achat", None) or getattr(self.orchestrator, "chat", None)
        if fn is None:
            return "[reply_service] El orquestador no expone métodos 'achat' ni 'chat'."

        # Llamar respetando sync/async
        try:
            if inspect.iscoroutinefunction(fn):
                result = await fn(messages=messages, k=k, node=node)
            else:
                result = fn(messages=messages, k=k, node=node)

            # Normalizar a texto
            if isinstance(result, dict):
                # intenta usar alguna clave típica
                for key in ("reply", "text", "content", "message"):
                    if key in result:
                        return str(result[key])
                return str(result)
            return str(result)
        except Exception as e:
            return f"[reply_service] error invocando orquestador: {type(e).__name__}: {e}"
