# -*- coding: utf-8 -*-
"""
Adaptador del runtime llama.cpp (llama_cpp_python) con interfaz de chat.
- Lee hiperparámetros desde variables de entorno (N_CTX, N_THREADS, N_BATCH, N_GPU_LAYERS).
- Exponemos chat(...) no-stream y stream(...) para streaming token a token.
- Extrae texto de respuestas tanto en .message.content como en .text (compat).
"""

from __future__ import annotations
import os
from typing import Dict, Iterable, List, Optional, Union, Generator, Any

from llama_cpp import Llama


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


class LlamaCppChat:
    """
    Pequeño wrapper de Llama para chat.
    """

    def __init__(self, model_path: str):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"MODEL_PATH no válido: {model_path}")

        n_ctx = _env_int("N_CTX", 4096)                # puedes subir a 8192 en el entorno
        n_threads = _env_int("N_THREADS", os.cpu_count() or 4)
        n_batch = _env_int("N_BATCH", 256)
        n_gpu_layers = _env_int("N_GPU_LAYERS", 0)     # -1 = todo en GPU si build lo permite
        seed = _env_int("LLAMA_SEED", 0)
        verbose = os.getenv("LLAMA_VERBOSE", "0") in ("1", "true", "True")
        use_mlock = os.getenv("LLAMA_MLOCK", "0") in ("1", "true", "True")
        use_mmap  = os.getenv("LLAMA_MMAP", "1") in ("1", "true", "True")

        # Stops extra por entorno (override mínimo): EXTRA_STOP="</s>|||<|eot_id|>"
        extra_stop = os.getenv("EXTRA_STOP", "").strip()
        self._default_stops: List[str] = [s for s in (x.strip() for x in extra_stop.split("|||")) if s]

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=verbose,
            use_mlock=use_mlock,
            use_mmap=use_mmap,
        )

    # ------------------------------- Helpers ------------------------------- #
    @staticmethod
    def _extract_text(obj: Dict[str, Any]) -> str:
        # Forma chat moderna
        try:
            ch = obj.get("choices")
            if isinstance(ch, list) and ch:
                c0 = ch[0]
                if isinstance(c0, dict):
                    # message.content
                    msg = c0.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg["content"]
                    # text (fallback)
                    if isinstance(c0.get("text"), str):
                        return c0["text"]
        except Exception:
            pass
        # Fallbacks comunes
        for k in ("text", "response", "content", "output", "message"):
            v = obj.get(k)
            if isinstance(v, str):
                return v
        return ""

    @staticmethod
    def _extract_delta(obj: Dict[str, Any]) -> str:
        # Stream: delta.content o text
        try:
            ch = obj.get("choices")
            if isinstance(ch, list) and ch:
                c0 = ch[0]
                if isinstance(c0, dict):
                    delta = c0.get("delta")
                    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                        return delta["content"]
                    if isinstance(c0.get("text"), str):
                        return c0["text"]
        except Exception:
            pass
        return ""

    @staticmethod
    def _merge_stops(user_stop: Optional[List[str]], default_stops: List[str]) -> Optional[List[str]]:
        # Combina stops del usuario con los del entorno, sin duplicados y sin strings vacíos.
        merged = [s for s in (user_stop or []) + default_stops if s]
        # Si queda vacía, devolvemos None para NO pasar 'stop' y evitar cortes inesperados.
        return merged or None

    # ------------------------------- Public API ---------------------------- #
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 128,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, str]:
        """Llamada no-stream; devuelve {'text': '...'}."""
        kwargs: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_new_tokens,
        }
        stop_final = self._merge_stops(stop, self._default_stops)
        if stop_final is not None:
            kwargs["stop"] = stop_final
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)

        out = self._llm.create_chat_completion(**kwargs)
        return {"text": self._extract_text(out)}

    def stream(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 128,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """Streaming token a token: genera texto plano incremental."""
        kwargs: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_new_tokens,
            "stream": True,
        }
        stop_final = self._merge_stops(stop, self._default_stops)
        if stop_final is not None:
            kwargs["stop"] = stop_final
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)

        for chunk in self._llm.create_chat_completion(**kwargs):
            piece = self._extract_delta(chunk)
            if piece:
                yield piece

    # Alias para compatibilidad con el server (sin tocar el server)
    def chat_stream(self, messages: List[Dict[str, str]], **kw) -> Generator[str, None, None]:
        return self.stream(messages, **kw)

    def stream_chat(self, messages: List[Dict[str, str]], **kw) -> Generator[str, None, None]:
        return self.stream(messages, **kw)
