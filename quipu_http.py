from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

from quipu_loop import QUIPU_CORE_VERSION, LoopConfig, QuipuLoop
from adapters.llama_cpp_adapter import LlamaCppAdapter

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / "web" / ".env.local"
DATA_DIR = ROOT_DIR / "data"
CONVERSATIONS_DB = DATA_DIR / "quipu_conversations.db"

DEFAULT_GREETING = "¡Hola! Soy tu AGI doméstica. ¿En qué te ayudo?"

def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_quipu_env() -> None:
    if not ENV_FILE.exists():
        return

    allowed = {
        "LLAMA_SERVER_URL",
        "BACKEND_URL",
        "LLAMA_MODEL_ID",
        "LLAMA_API_KEY",
        "AGI_API_KEY",
        "API_KEY",
    }

    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key in allowed and value:
            os.environ.setdefault(key, value)

def db_connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(CONVERSATIONS_DB),
        timeout=30,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn

def init_conversation_db() -> None:
    with db_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, id);

            CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at DESC);
            """
        )

def create_conversation(title: str = "Nuevo chat") -> Dict[str, Any]:
    conversation_id = uuid.uuid4().hex
    now = utc_iso()
    clean_title = str(title or "").strip() or "Nuevo chat"

    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO conversations(id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, clean_title, now, now),
        )
        conn.execute(
            """
            INSERT INTO messages(conversation_id, role, content, created_at)
            VALUES (?, 'assistant', ?, ?)
            """,
            (conversation_id, DEFAULT_GREETING, now),
        )

    return get_conversation(conversation_id)

def get_conversation(conversation_id: str) -> Dict[str, Any]:
    with db_connect() as conn:
        row = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()

        if row is None:
            raise KeyError(conversation_id)

        message_rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "messages": [
            {
                "role": item["role"],
                "content": item["content"],
                "created_at": item["created_at"],
            }
            for item in message_rows
        ],
    }

def list_conversations() -> List[Dict[str, Any]]:
    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                COUNT(m.id) AS message_count
            FROM conversations AS c
            LEFT JOIN messages AS m
                ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """
        ).fetchall()

    return [
        {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": int(row["message_count"] or 0),
        }
        for row in rows
    ]

def add_message(
    conversation_id: str,
    role: str,
    content: str,
) -> None:
    now = utc_iso()

    with db_connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()

        if exists is None:
            raise KeyError(conversation_id)

        conn.execute(
            """
            INSERT INTO messages(conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, str(content or ""), now),
        )
        conn.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
            """,
            (now, conversation_id),
        )

def maybe_set_automatic_title(
    conversation_id: str,
    user_text: str,
) -> None:
    with db_connect() as conn:
        row = conn.execute(
            """
            SELECT title,
                   (
                       SELECT COUNT(*)
                       FROM messages
                       WHERE conversation_id = ?
                         AND role = 'user'
                   ) AS user_count
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id, conversation_id),
        ).fetchone()

        if row is None:
            return

        if row["title"] != "Nuevo chat" or int(row["user_count"] or 0) != 1:
            return

        title = " ".join(str(user_text or "").split()).strip()
        if len(title) > 52:
            title = title[:49].rstrip() + "..."

        if not title:
            return

        conn.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (title, utc_iso(), conversation_id),
        )

load_quipu_env()
init_conversation_db()

cfg = LoopConfig(
    model_backend="llama-cpp",
    model_path=None,
    max_tokens=512,
    temperature=0.7,
    use_rag=True,
    use_planner=True,
    use_verifier=True,
    use_cognitive_control=True,
)

loop = QuipuLoop(cfg)

if not isinstance(loop.adapter, LlamaCppAdapter):
    raise RuntimeError(
        "QuipuLoop no está usando LlamaCppAdapter. "
        f"Adaptador actual: {type(loop.adapter).__name__}"
    )

app = FastAPI(
    title="Quipu Loop HTTP Gateway",
    version="1.0.0",
)

loop_lock = Lock()
db_lock = Lock()

@app.get("/health")
def health() -> Dict[str, Any]:
    status = loop.get_status()
    return {
        "ok": True,
        "service": "quipu-loop",
        "core_version": QUIPU_CORE_VERSION,
        "adapter": type(loop.adapter).__name__,
        "llama_server_url": os.getenv(
            "LLAMA_SERVER_URL",
            os.getenv("BACKEND_URL", "http://127.0.0.1:8123/v1"),
        ),
        "model": os.getenv("LLAMA_MODEL_ID", "qwen2.5-3b-instruct"),
        "conversation_persistence": True,
        "conversation_isolation": True,
        "conversation_db": str(CONVERSATIONS_DB),
        "core": status,
    }

@app.get("/conversations")
def conversations_list() -> Dict[str, Any]:
    with db_lock:
        return {"conversations": list_conversations()}

@app.post("/conversations")
def conversations_create(
    payload: Dict[str, Any] = Body(default={}),
) -> Dict[str, Any]:
    title = str(payload.get("title") or "Nuevo chat")

    with db_lock:
        conversation = create_conversation(title)

    return {"conversation": conversation}

@app.get("/conversations/{conversation_id}")
def conversations_get(conversation_id: str):
    try:
        with db_lock:
            conversation = get_conversation(conversation_id)

        return {"conversation": conversation}
    except KeyError:
        return JSONResponse(
            {
                "error": True,
                "detail": "La conversación no existe.",
            },
            status_code=404,
        )

@app.post("/chat")
def chat(payload: Dict[str, Any] = Body(...)):
    conversation_id = str(
        payload.get("conversation_id") or ""
    ).strip()

    messages: List[Dict[str, Any]] = payload.get("messages") or []
    direct_message = str(payload.get("message") or "").strip()

    if direct_message:
        last_user = direct_message
    else:
        last_user = next(
            (
                str(message.get("content", "")).strip()
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )

    if not last_user:
        return JSONResponse(
            {
                "error": True,
                "detail": "No se encontró un mensaje de usuario.",
            },
            status_code=400,
        )

    try:
        if not conversation_id:
            with db_lock:
                conversation = create_conversation()
            conversation_id = str(conversation["id"])
        else:
            with db_lock:
                conversation = get_conversation(conversation_id)

        # El contexto conversacional de corto plazo proviene exclusivamente
        # del chat seleccionado. La memoria consolidada de largo plazo sigue
        # siendo global dentro de Quipu.
        prior_context = [
            {
                "role": str(item.get("role") or ""),
                "content": str(item.get("content") or ""),
            }
            for item in (conversation.get("messages") or [])
            if str(item.get("role") or "") in ("user", "assistant")
            and str(item.get("content") or "").strip()
            and str(item.get("content") or "").strip() != DEFAULT_GREETING
        ][-8:]

        # Si un caller antiguo manda el historial explícito y el chat todavía
        # no tiene historia útil en SQLite, úsalo como compatibilidad.
        if not prior_context and messages:
            prior_context = [
                {
                    "role": str(item.get("role") or ""),
                    "content": str(item.get("content") or ""),
                }
                for item in messages[:-1]
                if str(item.get("role") or "") in ("user", "assistant")
                and str(item.get("content") or "").strip()
            ][-8:]

        with db_lock:
            add_message(conversation_id, "user", last_user)
            maybe_set_automatic_title(conversation_id, last_user)

        with loop_lock:
            reply = loop.run_once(
                last_user,
                session_id=conversation_id,
                recent_context_override=prior_context,
            )

        with db_lock:
            add_message(conversation_id, "assistant", reply)
            conversation = get_conversation(conversation_id)

        scientific = dict(
            getattr(loop, "_last_scientific_result", {}) or {}
        )

        return {
            "reply": reply,
            "engine": "QuipuLoop",
            "conversation_id": conversation_id,
            "conversation": conversation,
            "scientific_engine": bool(scientific.get("handled")),
            "scientific_intent": scientific.get("intent"),
        }

    except KeyError:
        return JSONResponse(
            {
                "error": True,
                "detail": "La conversación no existe.",
            },
            status_code=404,
        )
    except Exception as exc:
        return JSONResponse(
            {
                "error": True,
                "detail": f"{type(exc).__name__}: {exc}",
            },
            status_code=500,
        )

if __name__ == "__main__":
    uvicorn.run(
        "quipu_http:app",
        host="127.0.0.1",
        port=8130,
        reload=False,
    )
