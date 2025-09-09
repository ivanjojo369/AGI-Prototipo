# memory/semantic_store.py
from __future__ import annotations
import json, time
from pathlib import Path
from typing import List, Dict, Any

MEM_PATH = Path("data/memory/semantic_store.json")
MEM_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load() -> List[Dict[str, Any]]:
    if MEM_PATH.exists():
        try: return json.loads(MEM_PATH.read_text(encoding="utf-8"))
        except: return []
    return []

def _save(items: List[Dict[str, Any]]):
    MEM_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def upsert_fact(text: str, tags: List[str] = None, meta: Dict[str, Any] = None, ttl_days: int = 0):
    now = int(time.time())
    items = _load()
    norm = _norm(text)
    # dedupe por texto normalizado
    items = [it for it in items if _norm(it.get("text","")) != norm]
    entry = {"text": text, "tags": tags or [], "meta": meta or {}, "ts": now}
    if ttl_days > 0: entry["ttl"] = ttl_days*86400
    items.insert(0, entry)  # más reciente al frente
    _save(items)

def search(q: str, k: int = 3) -> List[Dict[str, Any]]:
    qn = _norm(q)
    now = int(time.time())
    items = _load()
    # purge por TTL
    keep = []
    for it in items:
        ttl = it.get("ttl")
        if ttl and (now - int(it.get("ts", now)) > ttl): 
            continue
        keep.append(it)
    if len(keep) != len(items): _save(keep)
    # scoring simple: coincidencia + recencia
    scored = []
    for it in keep:
        t = it.get("text","")
        score = (t.lower().count(qn) if qn else 1) + 0.000001*max(0, it.get("ts",0))
        scored.append((score, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:k]]
