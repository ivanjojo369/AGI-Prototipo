# memory/context_memory.py
# Memoria de contexto ligera con "slots" + historial reciente.
# Persistencia JSON (escritura atómica), bloqueo por hilo y helpers compatibles con tests/agents.

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# --------------------------------------------------------------------------------------
# Rutas y sincronización
# --------------------------------------------------------------------------------------

ROOT: Path = Path(__file__).resolve().parents[1]
NEW_PATH: Path = ROOT / "temp_memory" / "context.json"      # almacenamiento por defecto
OLD_PATH: Path = ROOT / "chroma_db" / "slot_memory.json"    # origen para migración legacy
_LOCK = threading.Lock()


# --------------------------------------------------------------------------------------
# Modelos de datos
# --------------------------------------------------------------------------------------

@dataclass
class SlotState:
    name: Optional[str] = None
    location: Optional[str] = None
    preferences: List[str] = field(default_factory=list)
    facts: Dict[str, str] = field(default_factory=dict)


@dataclass
class MessageRecord:
    role: str
    content: str
    ts: float
    meta: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Utilidades internas
# --------------------------------------------------------------------------------------

def _read_state(path: Path) -> Dict[str, Any]:
    """
    Lee el archivo JSON de memoria. Admite formas antiguas; si falla, retorna estructura vacía.
    """
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"slots": {}, "history": []}


def _write_state(path: Path, state: Dict[str, Any]) -> None:
    """
    Escritura atómica: escribe a .tmp y reemplaza para evitar corrupción.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _migrate_if_needed(src: Path, dst: Path) -> None:
    """
    Si existe el archivo legacy y no existe el nuevo, migra de forma best-effort.
    """
    if dst.exists() or not src.exists():
        return
    try:
        old = _read_state(src)
        _write_state(dst, {
            "slots": old.get("slots", {}),
            "history": old.get("history", []),
        })
    except Exception:
        # Si la migración falla, continuamos con estado vacío.
        pass


# --------------------------------------------------------------------------------------
# Núcleo de memoria
# --------------------------------------------------------------------------------------

class ContextMemory:
    """
    Memoria liviana: slots + historial reciente, persistida a disco.
    - max_history: tope suave de interacciones guardadas
    """

    def __init__(
        self,
        persist_path: Path | str = NEW_PATH,
        save_to_disk: bool = True,
        max_history: int = 40,
    ):
        self.persist_path: Path = Path(persist_path)
        self.save_to_disk: bool = bool(save_to_disk)
        self.max_history: int = int(max_history)

        self.slots: SlotState = SlotState()
        self.history: List[MessageRecord] = []

        _migrate_if_needed(OLD_PATH, self.persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_if_exists()

    # -------- Persistencia --------

    def _load_if_exists(self) -> None:
        if not self.persist_path.exists():
            return
        data = _read_state(self.persist_path)
        s = data.get("slots", {})
        self.slots = SlotState(
            name=s.get("name"),
            location=s.get("location"),
            preferences=list(s.get("preferences", [])),
            facts=dict(s.get("facts", {})),
        )
        hist: List[MessageRecord] = []
        for m in data.get("history", []):
            hist.append(
                MessageRecord(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    ts=float(m.get("ts", time.time())),
                    meta=dict(m.get("meta", {})),
                )
            )
        self.history = hist[-self.max_history:]

    def persist(self) -> None:
        if not self.save_to_disk:
            return
        state = {
            "slots": asdict(self.slots),
            "history": [asdict(h) for h in self.history[-self.max_history:]],
        }
        _write_state(self.persist_path, state)

    # -------- Slots API --------

    def update_slot(self, slot: str, value: str) -> None:
        slot = (slot or "").strip().lower()
        value = (value or "").strip()
        if not slot:
            return
        if slot == "name":
            self.slots.name = value
        elif slot == "location":
            self.slots.location = value
        else:
            self.slots.facts[slot] = value
        self.persist()

    def get_slot(self, slot: str) -> Optional[str]:
        slot = (slot or "").strip().lower()
        if slot == "name":
            return self.slots.name
        if slot == "location":
            return self.slots.location
        return self.slots.facts.get(slot)

    def add_preference(self, pref: str) -> None:
        pref = (pref or "").strip()
        if pref and pref not in self.slots.preferences:
            self.slots.preferences.append(pref)
            self.persist()

    def get_preferences(self) -> List[str]:
        return list(self.slots.preferences)

    def forget_slot(self, slot: str) -> None:
        slot = (slot or "").strip().lower()
        if slot == "name":
            self.slots.name = None
        elif slot == "location":
            self.slots.location = None
        else:
            self.slots.facts.pop(slot, None)
        self.persist()

    # -------- Historial --------

    def add_history(self, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.history.append(
            MessageRecord(role=str(role), content=str(content), ts=time.time(), meta=dict(meta or {}))
        )
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        self.persist()

    def recall_recent(self, k: int = 6) -> List[MessageRecord]:
        return self.history[-max(0, int(k)):] if k > 0 else []

    # -------- Utilidades --------

    def summary_text(self) -> str:
        """Resumen legible del estado para enriquecer reflexión/planificación."""
        parts: List[str] = []
        if self.slots.name:
            parts.append(f"Nombre del usuario: {self.slots.name}.")
        if self.slots.location:
            parts.append(f"Ubicación del usuario: {self.slots.location}.")
        if self.slots.preferences:
            prefs = ", ".join(self.slots.preferences[:8])
            parts.append(f"Preferencias: {prefs}.")
        if self.slots.facts:
            facts = "; ".join([f"{k}: {v}" for k, v in list(self.slots.facts.items())[:8]])
            parts.append(f"Hechos: {facts}.")
        return " ".join(parts) if parts else "Sin datos de slots aún."


# --------------------------------------------------------------------------------------
# API de módulo (helpers p/ tests)
# --------------------------------------------------------------------------------------

__all__ = [
    "ContextMemory",
    "save_interaction_to_memory",
    "load_context_window",
    "get_recent_context",
    "reset_context_memory",
    "retrieve_relevant_memories",
]

# Singleton perezoso para que los helpers usen el mismo store
_STORE: Optional[ContextMemory] = None

def _get_store() -> ContextMemory:
    global _STORE
    if _STORE is None:
        _STORE = ContextMemory(persist_path=NEW_PATH)
    return _STORE


def save_interaction_to_memory(
    user_message: Optional[str] = None,
    assistant_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Guarda una o dos interacciones en temp_memory/context.json.

    Compatibilidad:
      - save_interaction_to_memory(user_text, assistant_text, **meta)
      - save_interaction_to_memory(user_message="...", assistant_message="...", session_id="...", metadata={...})
    """
    md = dict(metadata or {})
    if session_id:
        md.setdefault("session_id", session_id)

    # Compat con firmas antiguas
    user_msg = kwargs.pop("user_text", None) or user_message
    assistant_msg = kwargs.pop("assistant_text", None) or assistant_message

    added = 0
    with _LOCK:
        store = _get_store()
        if user_msg:
            store.add_history("user", str(user_msg), meta=md)
            added += 1
        if assistant_msg:
            store.add_history("assistant", str(assistant_msg), meta=md)
            added += 1
    return {"ok": 1 if added else 0, "added": added}


def load_context_window(limit: int = 10, session_id: Optional[str] = None, **kwargs: Any) -> List[Dict[str, Any]]:
    """
    Devuelve las últimas 'limit' interacciones; si hay session_id, filtra.
    """
    with _LOCK:
        store = _get_store()
        items = store.recall_recent(max(1, int(limit or 10)))
    if session_id:
        items = [x for x in items if (x.meta or {}).get("session_id") == session_id]
    return [asdict(x) for x in items]


def get_recent_context(n: int = 10, **kwargs: Any) -> List[Dict[str, Any]]:
    """Alias usado en algunos tests."""
    return load_context_window(limit=n, **kwargs)


def reset_context_memory() -> Dict[str, Any]:
    """
    Limpia todo el contexto (útil para tests). Mantiene los slots, borra historial.
    """
    with _LOCK:
        store = _get_store()
        store.history = []
        store.persist()
    return {"ok": 1}


# --------------------------------------------------------------------------------------
# Búsqueda de recuerdos relevantes
# --------------------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    t = (text or "").lower()
    # separar por caracteres no alfanuméricos (incluye acentos comunes)
    return [tok for tok in re.split(r"[^\wáéíóúñü]+", t) if tok]

def _score_text(query_tokens: Set[str], text: str, ts: Optional[float]) -> float:
    if not text:
        return 0.0
    toks = set(_tokenize(text))
    base = float(len(query_tokens & toks)) if (toks and query_tokens) else 0.0
    # Bonus por recencia (decay por días)
    if ts:
        age_days = max(0.0, (time.time() - float(ts)) / 86_400.0)
        recency = 1.0 + math.exp(-age_days) * 0.5  # ~1.0–1.5
    else:
        recency = 1.0
    return base * recency

def retrieve_relevant_memories(
    query: str,
    top_k: int = 5,
    session_id: Optional[str] = None,
    use_slots: bool = True,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Busca recuerdos relevantes (historial y, opcionalmente, slots).
    Devuelve: [{"text": str, "score": float, "source": "history|slot", ...}, ...]
    Acepta alias: n / k -> top_k.
    """
    # alias típicos que algunos tests usan
    if "n" in kwargs and not kwargs.get("top_k"):
        top_k = int(kwargs["n"])
    if "k" in kwargs and not kwargs.get("top_k"):
        top_k = int(kwargs["k"])

    q_tokens: Set[str] = set(_tokenize(query or ""))
    store = _get_store()

    candidates: List[Dict[str, Any]] = []

    # 1) Historial (filtrado opcional por sesión)
    for m in store.history:
        if session_id and (m.meta or {}).get("session_id") != session_id:
            continue
        text = f"{m.role}: {m.content}"
        score = _score_text(q_tokens, text, m.ts)
        if score > 0:
            candidates.append({
                "text": text,
                "score": score,
                "source": "history",
                "role": m.role,
                "ts": m.ts,
                "meta": dict(m.meta or {}),
            })

    # 2) Slots (si se pide)
    if use_slots:
        s = store.slots
        slot_items: List[Tuple[str, str]] = []
        if s.name:
            slot_items.append(("name", s.name))
        if s.location:
            slot_items.append(("location", s.location))
        for k, v in (s.facts or {}).items():
            slot_items.append((k, v))
        if s.preferences:
            slot_items.append(("preferences", ", ".join(s.preferences)))

        for key, val in slot_items:
            text = f"{key}: {val}"
            score = _score_text(q_tokens, text, None)
            if score > 0:
                candidates.append({
                    "text": text,
                    "score": score,
                    "source": "slot",
                    "slot": key,
                })

    # Orden y recorte
    candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return candidates[: max(1, int(top_k or 5))]
