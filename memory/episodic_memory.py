# memory/episodic_memory.py
from __future__ import annotations
import json, time
from pathlib import Path
from typing import List, Dict, Any, Optional

class EpisodicMemory:
    def __init__(self, persist_dir: str = "chroma_db", fname: str = "episodes.json", max_items: int = 500):
        self.persist_path = Path(persist_dir) / fname
        self.max_items = max_items
        self._items: List[Dict[str, Any]] = []
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        if self.persist_path.exists():
            try:
                self._items = json.loads(self.persist_path.read_text(encoding="utf-8"))
            except Exception:
                self._items = []

    def persist(self):
        data = self._items[-self.max_items:]
        self.persist_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, intent: str, text: str, reply: str, topic: Optional[str] = None, tags: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None):
        item = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "intent": (intent or "unknown"),
            "topic": (topic or self._infer_topic(text)),
            "text": text,
            "reply": reply,
            "tags": tags or [],
            "meta": meta or {}
        }
        self._items.append(item)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items:]
        self.persist()
        return item

    def recent(self, k: int = 10) -> List[Dict[str, Any]]:
        return list(reversed(self._items[-k:]))

    def search(self, q: str, k: int = 20) -> List[Dict[str, Any]]:
        ql = (q or "").lower()
        out = [it for it in reversed(self._items) if ql in (it.get("text","")+it.get("reply","")+it.get("topic","")).lower()]
        return out[:k]

    def purge(self, keep_last: int = 200):
        self._items = self._items[-keep_last:]
        self.persist()

    @staticmethod
    def _infer_topic(text: str) -> str:
        t = (text or "").lower()
        if "me llamo" in t: return "perfil:nombre"
        if "estoy en" in t or "vivo en" in t: return "perfil:ubicacion"
        if "me gusta" in t: return "perfil:preferencias"
        if "calcula" in t or any(ch in t for ch in "+-*/^"): return "herramienta:calculadora"
        if "hora" in t or "fecha" in t or "mañana" in t or "ayer" in t: return "herramienta:tiempo"
        return "dialogo"
