# adapters/chat_templates.py
from typing import List, Dict, Any

QWEN_STOP = ["<|im_end|>"]

def build_qwen_messages(messages: List[Dict[str, str]], system: str | None = None) -> List[Dict[str, Any]]:
    """
    Normaliza mensajes para --chat_format qwen.
    messages: [{"role": "user"|"assistant"|"system", "content": "..."}]
    Si hay system, lo antepone; asegura orden y tipos válidos.
    """
    out: List[Dict[str, str]] = []
    if system:
        out.append({"role": "system", "content": system})

    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role not in ("user", "assistant", "system"):
            # fallback conservador
            role = "user"
        if role == "system" and system:
            # evita duplicar system si ya lo pasamos por parámetro
            continue
        out.append({"role": role, "content": content})
    return out

def stops_for_model(model_name: str) -> List[str]:
    # Extensible a otros modelos (openchat/nous-hermes con plantillas distintas)
    return QWEN_STOP
