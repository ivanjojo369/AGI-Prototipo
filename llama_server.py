#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llama_server.py — FastAPI para servir un modelo GGUF con llama-cpp-python.

Características:
- Carga de modelo GGUF local (llama-cpp-python).
- Auth flexible (Bearer, X-API-Key, query ?api_key=, body api_key) — si no hay API_KEY, auth desactivada.
- Endpoint /chat tolerante a varios esquemas de payload:
    • prompt | input | text | query | message
    • messages (estilo OpenAI)
    • max_new_tokens | max_tokens | n_predict
- /health para chequeo rápido.
- CLI con --model, --ctx, --threads, --batch, --gpu_layers, --host, --port.
- Variables de entorno de respaldo: MODEL_PATH, API_KEY, HOST, PORT, CTX, THREADS, BATCH, GPU_LAYERS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ----------------------------- Config y Modelo ----------------------------- #

@dataclass
class ServerConfig:
    model_path: str
    ctx: int = 4096
    threads: int = 8
    batch: int = 256
    gpu_layers: int = 0
    temperature: float = 0.8
    top_p: float = 0.9
    host: str = "127.0.0.1"
    port: int = 8010
    api_key: Optional[str] = None
    chat_join_template: str = "{role}: {content}"  # cómo unir messages -> prompt (simple y universal)


LLM = None  # instancia global de llama_cpp.Llama
CFG: ServerConfig  # instancia global de configuración


def _load_llama(cfg: ServerConfig):
    """Carga el modelo GGUF usando llama-cpp-python."""
    global LLM
    try:
        from llama_cpp import Llama
    except Exception as e:
        print(
            "\nERROR: llama-cpp-python no está instalado.\n"
            "Instala con: pip install llama-cpp-python\n"
        )
        raise

    if not cfg.model_path or not os.path.exists(cfg.model_path):
        raise RuntimeError(f"MODEL_PATH no válido: {cfg.model_path!r}")

    print(
        f"INFO llama_server: Cargando modelo: {cfg.model_path} "
        f"(ctx={cfg.ctx}, threads={cfg.threads}, batch={cfg.batch}, gpu_layers={cfg.gpu_layers})"
    )

    LLM = Llama(
        model_path=cfg.model_path,
        n_ctx=cfg.ctx,
        n_threads=cfg.threads,
        n_batch=cfg.batch,
        n_gpu_layers=cfg.gpu_layers,
        verbose=False,
    )
    print("INFO llama_server: Modelo cargado OK.")


# ----------------------------- Auth Flexible ------------------------------ #

def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


async def auth_dependency(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Auth flexible: Bearer, X-API-Key, query ?api_key=, body.api_key."""
    key_required = CFG.api_key or os.getenv("API_KEY")  # Runtime override si cambiara el env
    if not key_required:
        return  # Auth desactivada

    # 1) Authorization: Bearer <token>
    token = _extract_bearer_token(authorization)

    # 2) X-API-Key
    if not token and x_api_key:
        token = x_api_key.strip()

    # 3) Query param ?api_key=
    if not token:
        token = request.query_params.get("api_key")

    # 4) api_key en el body
    if not token:
        try:
            body_bytes = await request.body()
            if body_bytes:
                data = json.loads(body_bytes.decode("utf-8"))
                token = data.get("api_key")
        except Exception:
            pass

    if not token or token != key_required:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ----------------------------- Esquemas Pydantic -------------------------- #

class ChatRequest(BaseModel):
    # Campos alternativos de entrada
    prompt: Optional[str] = None
    input: Optional[str] = None
    text: Optional[str] = None
    query: Optional[str] = None
    message: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None  # [{role, content}, ...]

    # Sampling / límites (acepta varios nombres)
    max_new_tokens: Optional[int] = Field(None, alias="max_tokens")
    n_predict: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    # Otros
    api_key: Optional[str] = None  # por si la API la espera en body
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

    class Config:
        allow_population_by_field_name = True


class ChatResponse(BaseModel):
    text: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None  # respuesta cruda por si interesa depurar


# --------------------------- Utilidades de normalización ------------------ #

def _join_messages(messages: List[Dict[str, str]], template: str) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(template.format(role=role, content=content))
    return "\n".join(parts).strip()


def _canonicalize_request(req: ChatRequest) -> Tuple[str, int, float, float, List[str]]:
    """Devuelve: (prompt, max_tokens, temperature, top_p, stop)"""
    # prompt
    prompt = (
        req.prompt or req.input or req.text or req.query or req.message or ""
    ).strip()

    if not prompt and req.messages:
        prompt = _join_messages(req.messages, CFG.chat_join_template)

    if not prompt:
        raise HTTPException(status_code=422, detail="Missing 'prompt' or 'messages'.")

    # tokens
    max_tokens = (
        req.max_new_tokens if req.max_new_tokens is not None else
        req.n_predict if req.n_predict is not None else
        128
    )

    # sampling
    temperature = req.temperature if req.temperature is not None else CFG.temperature
    top_p = req.top_p if req.top_p is not None else CFG.top_p

    stop = req.stop or []

    return prompt, int(max_tokens), float(temperature), float(top_p), stop


def _generate_with_llama(prompt: str, max_tokens: int, temperature: float, top_p: float, stop: List[str]) -> Dict[str, Any]:
    """Llama.txt-like interface."""
    assert LLM is not None, "Modelo no cargado"
    # Llama.__call__ devuelve un dict con 'choices'[0]['text']
    out = LLM(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop if stop else None,
    )
    return out


# ---------------------------------- FastAPI -------------------------------- #

app = FastAPI(title="Llama GGUF Server", version="1.0.0")

# CORS abierto por defecto (opcionalmente restrínjelo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _ok=Depends(auth_dependency)):
    prompt, max_tokens, temperature, top_p, stop = _canonicalize_request(req)

    # Generación
    raw = _generate_with_llama(prompt, max_tokens, temperature, top_p, stop)

    # Normalización de salida
    text = ""
    try:
        if isinstance(raw, dict):
            # llama-cpp-python: {'choices': [{'text': '...'}], ...}
            choices = raw.get("choices")
            if isinstance(choices, list) and choices:
                text = choices[0].get("text") or ""
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
    except Exception:
        text = ""

    return ChatResponse(
        text=text.strip(),
        model=os.path.basename(CFG.model_path),
        usage=raw.get("usage") if isinstance(raw, dict) else None,
        raw=raw if isinstance(raw, dict) else None,
    )


# ------------------------------- CLI / main -------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> ServerConfig:
    ep = argparse.ArgumentParser(
        description="Servidor FastAPI para modelos GGUF con llama-cpp-python.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Ejemplos:
              python llama_server.py --model "C:\\ruta\\modelo.gguf"
              API_KEY=dev123 python llama_server.py --model ./models/openchat-3.5-1210.Q4_K_M.gguf --host 0.0.0.0 --port 8010
            """
        ),
    )
    ep.add_argument("--model", dest="model_path", default=os.getenv("MODEL_PATH", ""), help="Ruta al modelo .gguf")
    ep.add_argument("--ctx", type=int, default=int(os.getenv("CTX", "4096")))
    ep.add_argument("--threads", type=int, default=int(os.getenv("THREADS", "8")))
    ep.add_argument("--batch", type=int, default=int(os.getenv("BATCH", "256")))
    ep.add_argument("--gpu_layers", type=int, default=int(os.getenv("GPU_LAYERS", "0")))
    ep.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    ep.add_argument("--port", type=int, default=int(os.getenv("PORT", "8010")))
    ep.add_argument("--temperature", type=float, default=0.8)
    ep.add_argument("--top_p", type=float, default=0.9)
    args = ep.parse_args(argv)

    api_key = os.getenv("API_KEY") or os.getenv("DEMO_API_KEY")
    return ServerConfig(
        model_path=args.model_path,
        ctx=args.ctx,
        threads=args.threads,
        batch=args.batch,
        gpu_layers=args.gpu_layers,
        temperature=args.temperature,
        top_p=args.top_p,
        host=args.host,
        port=args.port,
        api_key=api_key,
    )


def main():
    global CFG
    CFG = parse_args()

    # Carga del modelo antes de arrancar Uvicorn
    _load_llama(CFG)

    print("INFO: Application startup complete.")
    print(f"INFO: Uvicorn running on http://{CFG.host}:{CFG.port}")

    uvicorn.run(
        app,
        host=CFG.host,
        port=CFG.port,
        log_level="info",
        # reload=False  # si quieres hot-reload, cámbialo a True (ojo con variables de entorno)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR al iniciar llama_server:", e)
        sys.exit(1)
