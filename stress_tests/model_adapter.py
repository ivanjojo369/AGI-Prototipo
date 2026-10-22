"""
Adaptador de modelo para los stress tests de LLM.

Usa el mismo estilo que chat_local.py:
- openai.ChatCompletion.create(...)
- servidor local compatible con la API de OpenAI
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

try:
    import openai  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "No se pudo importar el paquete 'openai'.\n"
        "Asegúrate de instalarlo en tu entorno 'ab-py313' con:\n"
        "    pip install openai\n"
        "o con la librería compatible que uses para tu servidor local."
    ) from e


class ModelAdapter:
    """
    Pequeña capa de abstracción para llamar al LLM desde los tests.

    - Por defecto apunta a tu servidor local (mismo estilo que chat_local.py).
    - Todos los escenarios (SearchGlitch, NPM-Nightmare, etc.) solo tienen que
      construir `messages` y llamar a `adapter.chat(messages, ...)`.
    """

    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        # Defaults inspirados en tu chat_local.py
        api_base = api_base or os.getenv("LLM_API_BASE", "http://localhost:7860/v1")
        api_key = api_key or os.getenv("OPENAI_API_KEY", "sk-fake")
        model = model or os.getenv("LLM_MODEL_NAME", "openchat-3.5-1210.Q4_K_M.gguf")

        # Config global de la librería openai (estilo 0.28 / llama-cpp-openai)
        openai.api_base = api_base  # type: ignore[attr-defined]
        openai.api_key = api_key    # type: ignore[attr-defined]

        self.model = model

    # --------------------------------------------------------------------- #
    #  Interfaz principal que usan los tests
    # --------------------------------------------------------------------- #
    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """
        Envía un chat al modelo y devuelve solo el 'content' del assistant.

        `messages` es la lista estándar de OpenAI:
        [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."},
          ...
        ]
        """
        resp = openai.ChatCompletion.create(  # type: ignore[attr-defined]
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Estilo clásico de la API: choices[0].message["content"]
        choice = resp.choices[0]
        # Algunos clientes devuelven .message.content, otros dict:
        msg = getattr(choice, "message", None)
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if content is None:
            # Fallback agresivo por si cambia el formato
            content = str(resp)

        return content


# Helper cómodo para los tests
def get_default_adapter() -> ModelAdapter:
    return ModelAdapter()

# ---------------------------------------------------------------------------
# Helper de alto nivel usado por stress_tests/harness.py
# ---------------------------------------------------------------------------

from typing import List, Dict, Any  # (si no lo tienes ya arriba)

_default_adapter: ModelAdapter | None = None


def _get_or_create_adapter() -> ModelAdapter:
    """
    Devuelve un singleton de ModelAdapter para reutilizar la misma
    instancia entre llamadas.
    """
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = get_default_adapter()
    return _default_adapter


def ask_model(
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    Función sencilla que usa harness.py.

    - `messages` es la lista de mensajes estilo OpenAI:
      [{"role": "...", "content": "..."}, ...]
    - Devuelve siempre un string con la respuesta del modelo.
    """
    adapter = _get_or_create_adapter()
    reply = adapter.chat(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return str(reply)
