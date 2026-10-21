# rag/pipeline.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag import retriever
from root.settings import RAG_SCORE_THRESHOLD, RAG_STORE_JSON, RAG_TOPK_DEFAULT


REASON_PREAMBLE = (
    "Eres un asistente técnico y cuidadoso.\n"
    "Usa el contexto documental solo como evidencia y no inventes datos ausentes.\n"
)


class RagPipeline:
    """Adaptador único de RAG usado por QuipuLoop.

    La autoridad de recuperación vive en :mod:`rag.retriever`; esta clase solo
    expone el contrato estable que QuipuLoop necesita: ``retrieve`` y
    ``render_context``. Así evitamos que el bucle central caiga silenciosamente
    en un stub cuando el pipeline funcional existe pero no había una clase.
    """

    VERSION = "1.1.0"

    def __init__(
        self,
        *,
        top_k: int = RAG_TOPK_DEFAULT,
        min_score: float = RAG_SCORE_THRESHOLD,
    ) -> None:
        self.top_k = max(1, int(top_k))
        self.min_score = float(min_score)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        return retriever.search(
            query=str(query or ""),
            top_k=max(1, int(top_k or self.top_k)),
            min_score=(
                self.min_score
                if min_score is None
                else float(min_score)
            ),
        )

    @staticmethod
    def render_context(docs: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for index, doc in enumerate(docs or [], 1):
            if not isinstance(doc, dict):
                continue
            text = " ".join(str(doc.get("text") or "").split()).strip()
            if not text:
                continue
            meta = dict(doc.get("meta") or {})
            source = str(meta.get("path") or meta.get("source") or doc.get("id") or "documento")
            score = float(doc.get("score") or 0.0)
            lines.append(f"[{index}] {source} | score={score:.3f}\n{text}")
        return "\n\n".join(lines)

    def status(self) -> Dict[str, Any]:
        path = Path(RAG_STORE_JSON)
        count = 0
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    count = len(data)
            except Exception:
                count = 0
        return {
            "version": self.VERSION,
            "store": str(path),
            "store_exists": path.exists(),
            "documents": count,
            "top_k": self.top_k,
            "min_score": self.min_score,
        }


def build_context(
    query: str,
    top_k: int = RAG_TOPK_DEFAULT,
    min_score: float = RAG_SCORE_THRESHOLD,
) -> Tuple[str, List[Dict[str, Any]]]:
    pipe = RagPipeline(top_k=top_k, min_score=min_score)
    hits = pipe.retrieve(query)
    return pipe.render_context(hits), hits


def augment_messages(
    user_message: str,
    top_k: int = RAG_TOPK_DEFAULT,
    min_score: float = RAG_SCORE_THRESHOLD,
) -> Tuple[List[Dict[str, str]], str]:
    ctx_text, hits = build_context(user_message, top_k, min_score)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": REASON_PREAMBLE},
    ]
    if ctx_text:
        messages.append({"role": "system", "content": "Contexto RAG:\n" + ctx_text})
    messages.append({"role": "user", "content": user_message})
    return messages, retriever.citations(hits)


def answer_via_llama(
    user_message: str,
    server_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compatibilidad para callers antiguos; QuipuLoop no usa esta ruta."""
    import requests

    server_url = (
        server_url
        or os.getenv("LLAMA_SERVER_URL")
        or "http://127.0.0.1:8123"
    ).rstrip("/")
    api_key = api_key or os.getenv("LLAMA_API_KEY") or ""
    model_id = (
        model_id
        or os.getenv("LLAMA_MODEL_ID")
        or os.getenv("LLAMA_REMOTE_MODEL_ID")
        or "qwen2.5-3b-instruct"
    )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-api-key"] = api_key

    messages, cits = augment_messages(user_message)
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 512,
    }
    r = requests.post(
        f"{server_url}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return {"reply": text, "citations": cits, "route": "chat"}


__all__ = [
    "RagPipeline",
    "build_context",
    "augment_messages",
    "answer_via_llama",
]
