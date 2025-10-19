# -*- coding: utf-8 -*-
"""
API de tu AGI doméstica:
- Auth por API-Key (header X-API-Key)
- Q&A con RAG+Memoria
- Chat por sesiones (persistencia en JSONL)
- Indexador y reparación del store

Cómo usar:
  # PowerShell (local)
  $env:AGI_API_KEY = "mi_clave_agi"
  $env:PYTHONPATH = (Get-Location).Path
  uvicorn root.api_server:app --host 127.0.0.1 --port 8010 --reload
"""
from __future__ import annotations

import os
import sys
import json
import uuid
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Security
from pydantic import BaseModel
from fastapi.security.api_key import APIKeyHeader

# -------- settings robusto --------
try:
    from .settings import (
        RAG_SCORE_THRESHOLD, RAG_TOPK_DEFAULT,
        SESSIONS_DIR, API_KEY as CFG_API_KEY,
    )
except Exception:
    try:
        from settings import RAG_SCORE_THRESHOLD, RAG_TOPK_DEFAULT, SESSIONS_DIR, API_KEY as CFG_API_KEY  # type: ignore
    except Exception:
        from root.settings import RAG_SCORE_THRESHOLD, RAG_TOPK_DEFAULT, SESSIONS_DIR, API_KEY as CFG_API_KEY  # type: ignore

from rag.retriever import search as rag_search, citations as rag_citations, repair as rag_repair
from memory.memory import write as memory_write
from root.quipu_loop import QuipuLoop

# -----------------------------------------------------------------------------
# Seguridad: API-Key (header X-API-Key). Si no hay clave configurada, no exige.
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def require_api_key(api_key: Optional[str] = Security(api_key_header)):
    expected = os.getenv("AGI_API_KEY") or (CFG_API_KEY or "")
    if not expected:
        return None  # modo dev: sin clave
    if api_key == expected:
        return api_key
    raise HTTPException(status_code=401, detail=f"Missing or invalid {API_KEY_NAME}")

# -----------------------------------------------------------------------------
# Modelos
class AnswerIn(BaseModel):
    query: str
    project_id: str = "default"
    min_score: float = RAG_SCORE_THRESHOLD
    top_k: int = RAG_TOPK_DEFAULT

class SearchIn(BaseModel):
    query: str
    top_k: int = RAG_TOPK_DEFAULT
    min_score: float = RAG_SCORE_THRESHOLD

class MemoryWriteIn(BaseModel):
    text: str
    user: str = "api"
    project_id: str = "default"
    tags: List[str] = []
    importance: float = 0.4
    meta: Optional[Dict[str, Any]] = None

class IndexFolderIn(BaseModel):
    path: str
    ext: str = ".py,.md,.txt"
    mode: str = "append"   # "fresh" | "append"

# Chat
class ChatSessionCreateIn(BaseModel):
    session_id: Optional[str] = None
    project_id: str = "default"

class ChatSendIn(BaseModel):
    session_id: str
    message: str
    project_id: str = "default"
    min_score: float = RAG_SCORE_THRESHOLD
    top_k: int = RAG_TOPK_DEFAULT

# -----------------------------------------------------------------------------
app = FastAPI(title="AGI Doméstica API", version="1.1.0")

# -------------------- Helpers para sesiones --------------------
def _session_path(session_id: str) -> Path:
    return Path(SESSIONS_DIR) / f"{session_id}.jsonl"

def _append_session(session_id: str, item: Dict[str, Any]) -> None:
    p = _session_path(session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def _read_session(session_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    p = _session_path(session_id)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines[-limit:]]

# -----------------------------------------------------------------------------
# Endpoints

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/answer", dependencies=[Depends(require_api_key)])
def v1_answer(body: AnswerIn):
    loop = QuipuLoop(project_id=body.project_id, min_score=body.min_score, top_k=body.top_k)
    try:
        return loop.run(body.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"answer failed: {e}")

@app.post("/v1/search", dependencies=[Depends(require_api_key)])
def v1_search(body: SearchIn):
    try:
        hits = rag_search(body.query, top_k=body.top_k, min_score=body.min_score)
        cites = rag_citations(hits)
        return {"ok": True, "hits": hits, "citations": cites}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {e}")

@app.post("/v1/memory/write", dependencies=[Depends(require_api_key)])
def v1_memory_write(body: MemoryWriteIn):
    try:
        return memory_write(
            body.text, user=body.user, project_id=body.project_id,
            tags=body.tags, importance=body.importance, meta=body.meta or {}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"memory.write failed: {e}")

@app.post("/v1/index/folder", dependencies=[Depends(require_api_key)])
def v1_index_folder(body: IndexFolderIn):
    p = Path(body.path)
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail=f"path not found: {body.path}")
    cmd = [sys.executable, "-m", "scripts.index_folder", "--path", str(p), "--ext", body.ext, "--mode", body.mode]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"index_folder error: {proc.stderr.strip()}")
    return {"ok": True, "stdout": proc.stdout}

@app.post("/v1/rag/repair", dependencies=[Depends(require_api_key)])
def v1_rag_repair():
    try:
        return rag_repair()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"repair failed: {e}")

# -------------------- Chat: sesiones --------------------
@app.post("/v1/chat/session", dependencies=[Depends(require_api_key)])
def chat_session_create(body: ChatSessionCreateIn):
    sid = body.session_id or uuid.uuid4().hex
    # registra evento de creación
    _append_session(sid, {"role": "system", "event": "session_created", "project_id": body.project_id})
    return {"ok": True, "session_id": sid}

@app.get("/v1/chat/sessions", dependencies=[Depends(require_api_key)])
def chat_list_sessions():
    dirp = Path(SESSIONS_DIR)
    dirp.mkdir(parents=True, exist_ok=True)
    items = []
    for f in dirp.glob("*.jsonl"):
        try:
            st = f.stat()
            items.append({"session_id": f.stem, "size": st.st_size, "mtime": int(st.st_mtime)})
        except Exception:
            pass
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"ok": True, "sessions": items}

@app.get("/v1/chat/history/{session_id}", dependencies=[Depends(require_api_key)])
def chat_history(session_id: str, limit: int = 50):
    data = _read_session(session_id, limit=limit)
    return {"ok": True, "session_id": session_id, "messages": data}

@app.post("/v1/chat/send", dependencies=[Depends(require_api_key)])
def chat_send(body: ChatSendIn):
    if not body.session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    # guarda turno del usuario
    _append_session(body.session_id, {"role": "user", "content": body.message})

    # llama al loop
    loop = QuipuLoop(project_id=body.project_id, min_score=body.min_score, top_k=body.top_k)
    out = loop.run(body.message)

    # guarda turno del asistente
    _append_session(body.session_id, {"role": "assistant", "content": out.get("output", ""), "meta": out.get("stats", {})})
    return {"ok": True, "session_id": body.session_id, "response": out}
