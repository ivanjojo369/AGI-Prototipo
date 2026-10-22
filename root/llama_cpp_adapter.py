# root/llama_cpp_adapter.py
from __future__ import annotations
import os
from typing import List, Dict, Any
from llama_cpp import Llama

_LLM = None

def _get_llm() -> Llama:
    global _LLM
    if _LLM is None:
        model = os.getenv("LLAMA_MODEL", r".\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf")
        n_ctx = int(os.getenv("LLAMA_CTX", "8192"))
        n_gpu_layers = int(os.getenv("LLAMA_N_GPU_LAYERS", "0"))
        _LLM = Llama(
            model_path=model,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            # chat_format="qwen2",  # normalmente no hace falta; el GGUF trae plantilla
        )
    return _LLM

def generate(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Modo prompt llano (no chat)."""
    out = _get_llm()(prompt, max_tokens=max_tokens, temperature=temperature, echo=False)
    return out["choices"][0]["text"]

def chat(messages: List[Dict[str, str]], max_tokens: int = 256, temperature: float = 0.7) -> str:
    """
    Modo chat (role-based). messages = [{"role":"system"/"user"/"assistant","content": "..."}]
    """
    out = _get_llm().create_chat_completion(
        messages=messages,
        temperature=temperature, top_p=0.95,
        max_tokens=max_tokens, repeat_penalty=1.1
    )
    return out["choices"][0]["message"]["content"]
