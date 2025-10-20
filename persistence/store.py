# -*- coding: utf-8 -*-
"""
persistence.store
-----------------
Utilidades de persistencia ligeras para la AGI:
- Sesiones (historial de chat) en JSONL
- Checkpoints genéricos (cualquier payload) en JSON

Rutas por defecto: DATA_DIR/persistence/{sessions,checkpoints}
Si existe root.settings, se toma DATA_DIR desde ahí.

Todas las funciones devuelven dicts "amigables" y nunca lanzan
excepciones hacia arriba (atrapa y retorna {'ok': False, 'error': ...}).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import time
import uuid

# -------------------- Rutas base --------------------
def _default_data_dir() -> Path:
    # …/persistence/store.py -> …/persistence -> …/ (raíz repo)
    return Path(__file__).resolve().parents[2] / "data"

try:
    from root.settings import DATA_DIR as _SETTINGS_DATA_DIR  # type: ignore
    DATA_DIR = Path(_SETTINGS_DATA_DIR)
except Exception:
    DATA_DIR = _default_data_dir()

PERSIST_DIR = DATA_DIR / "persistence"
SESSIONS_DIR = PERSIST_DIR / "sessions"
CHECKPOINTS_DIR = PERSIST_DIR / "checkpoints"
for d in (PERSIST_DIR, SESSIONS_DIR, CHECKPOINTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -------------------- Util: IO atómica --------------------
def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _json_dumps(o: Any) -> str:
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))

def _json_loads(s: str) -> Any:
    return json.loads(s)

# -------------------- Modelos --------------------
@dataclass
class SessionMessage:
    session_id: str
    role: str
    content: str
    ts: float
    meta: Dict[str, Any]

# -------------------- Sesiones (JSONL) --------------------
def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.jsonl"

def create_session(session_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Crea una nueva sesión; si no pasas session_id, se genera un UUID corto.
    """
    try:
        sid = session_id or uuid.uuid4().hex[:12]
        p = _session_path(sid)
        if p.exists():
            return {"ok": True, "session_id": sid, "created": False, "path": str(p)}
        # Crear archivo vacío (cabecera opcional)
        header = {"type": "session_header", "session_id": sid, "ts": time.time(), "meta": meta or {}}
        _atomic_write_text(p, _json_dumps(header) + "\n")
        return {"ok": True, "session_id": sid, "created": True, "path": str(p)}
    except Exception as e:
        return {"ok": False, "error": f"create_session: {e}"}

def append_message(session_id: str, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Añade un mensaje a la sesión en formato JSONL (una línea por evento).
    """
    try:
        p = _session_path(session_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        msg = SessionMessage(session_id=session_id, role=role, content=content, ts=time.time(), meta=meta or {})
        # Append sin truncar (abrimos y añadimos una línea)
        with p.open("a", encoding="utf-8") as f:
            f.write(_json_dumps({"type": "message", **asdict(msg)}) + "\n")
        return {"ok": True, "session_id": session_id}
    except Exception as e:
        return {"ok": False, "error": f"append_message: {e}"}

def get_session(session_id: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Devuelve lista de eventos (header + messages). Si limit>0, devuelve
    sólo los últimos N mensajes (mantiene header).
    """
    try:
        p = _session_path(session_id)
        if not p.exists():
            return {"ok": False, "error": "not_found", "events": []}
        lines = p.read_text(encoding="utf-8").splitlines()
        events: List[Dict[str, Any]] = []
        for i, ln in enumerate(lines):
            if not ln.strip():
                continue
            ev = _json_loads(ln)
            events.append(ev)
        if limit and limit > 0:
            header = [ev for ev in events if ev.get("type") == "session_header"][:1]
            msgs = [ev for ev in events if ev.get("type") == "message"][-limit:]
            events = header + msgs
        return {"ok": True, "events": events}
    except Exception as e:
        return {"ok": False, "error": f"get_session: {e}", "events": []}

def list_sessions() -> Dict[str, Any]:
    """
    Lista archivos de sesión y metadatos mínimos.
    """
    try:
        out: List[Dict[str, Any]] = []
        for f in sorted(SESSIONS_DIR.glob("*.jsonl")):
            sid = f.stem
            size = f.stat().st_size
            out.append({"session_id": sid, "path": str(f), "bytes": size})
        return {"ok": True, "sessions": out}
    except Exception as e:
        return {"ok": False, "error": f"list_sessions: {e}", "sessions": []}

def delete_session(session_id: str) -> Dict[str, Any]:
    try:
        p = _session_path(session_id)
        if p.exists():
            p.unlink()
            return {"ok": True, "deleted": 1}
        return {"ok": True, "deleted": 0}
    except Exception as e:
        return {"ok": False, "error": f"delete_session: {e}"}

# -------------------- Checkpoints (JSON) --------------------
def _ckpt_path(name: str) -> Path:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", ".", "+"))
    return CHECKPOINTS_DIR / f"{safe}.json"

def save_checkpoint(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Guarda un checkpoint (JSON). Sobrescribe si ya existe.
    """
    try:
        p = _ckpt_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"name": name, "ts": time.time(), "payload": payload}
        _atomic_write_text(p, _json_dumps(rec))
        return {"ok": True, "path": str(p)}
    except Exception as e:
        return {"ok": False, "error": f"save_checkpoint: {e}"}

def load_checkpoint(name: str) -> Dict[str, Any]:
    try:
        p = _ckpt_path(name)
        if not p.exists():
            return {"ok": False, "error": "not_found"}
        data = _json_loads(_read_text(p))
        return {"ok": True, "checkpoint": data}
    except Exception as e:
        return {"ok": False, "error": f"load_checkpoint: {e}"}

def list_checkpoints(prefix: Optional[str] = None) -> Dict[str, Any]:
    try:
        out: List[Dict[str, Any]] = []
        for f in sorted(CHECKPOINTS_DIR.glob("*.json")):
            if prefix and not f.stem.startswith(prefix):
                continue
            try:
                meta = _json_loads(_read_text(f))
                ts = meta.get("ts")
            except Exception:
                ts = None
            out.append({"name": f.stem, "path": str(f), "ts": ts})
        return {"ok": True, "checkpoints": out}
    except Exception as e:
        return {"ok": False, "error": f"list_checkpoints: {e}", "checkpoints": []}

def prune_old_checkpoints(keep: int = 10, prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    Mantiene los más recientes (por ts) y elimina el resto.
    """
    try:
        lst = list_checkpoints(prefix).get("checkpoints", [])
        lst = [x for x in lst if x.get("ts") is not None]
        lst.sort(key=lambda x: x["ts"], reverse=True)
        to_del = lst[keep:]
        n = 0
        for it in to_del:
            try:
                Path(it["path"]).unlink()
                n += 1
            except Exception:
                pass
        return {"ok": True, "deleted": n, "kept": len(lst) - n}
    except Exception as e:
        return {"ok": False, "error": f"prune_old_checkpoints: {e}"}

# -------------------- CLI simple --------------------
def _print(obj: Any) -> None:
    print(_json_dumps(obj))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("persistence.store")
    sub = ap.add_subparsers(dest="cmd")

    s_new = sub.add_parser("session-new")
    s_new.add_argument("--id", dest="sid", default=None)

    s_add = sub.add_parser("session-add")
    s_add.add_argument("sid")
    s_add.add_argument("role")
    s_add.add_argument("text")

    s_get = sub.add_parser("session-get")
    s_get.add_argument("sid")
    s_get.add_argument("--limit", type=int, default=None)

    s_ls = sub.add_parser("session-ls")
    s_rm = sub.add_parser("session-rm")
    s_rm.add_argument("sid")

    c_save = sub.add_parser("ckpt-save")
    c_save.add_argument("name")
    c_save.add_argument("json_payload", help='ej: {"k":"v"}')

    c_load = sub.add_parser("ckpt-load")
    c_load.add_argument("name")

    c_ls = sub.add_parser("ckpt-ls")
    c_ls.add_argument("--prefix", default=None)

    c_prune = sub.add_parser("ckpt-prune")
    c_prune.add_argument("--keep", type=int, default=10)
    c_prune.add_argument("--prefix", default=None)

    args = ap.parse_args()
    if args.cmd == "session-new":
        _print(create_session(args.sid))
    elif args.cmd == "session-add":
        _print(append_message(args.sid, args.role, args.text))
    elif args.cmd == "session-get":
        _print(get_session(args.sid, limit=args.limit))
    elif args.cmd == "session-ls":
        _print(list_sessions())
    elif args.cmd == "session-rm":
        _print(delete_session(args.sid))
    elif args.cmd == "ckpt-save":
        try:
            payload = json.loads(args.json_payload)
        except Exception as e:
            _print({"ok": False, "error": f"json_payload parse: {e}"})
        else:
            _print(save_checkpoint(args.name, payload))
    elif args.cmd == "ckpt-load":
        _print(load_checkpoint(args.name))
    elif args.cmd == "ckpt-ls":
        _print(list_checkpoints(prefix=args.prefix))
    elif args.cmd == "ckpt-prune":
        _print(prune_old_checkpoints(keep=args.keep, prefix=args.prefix))
    else:
        ap.print_help()
