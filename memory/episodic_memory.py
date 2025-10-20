# -*- coding: utf-8 -*-
"""
Memoria de largo plazo (episódica + índice semántico) autónoma.
- API funcional estable:
    memory_write / memory_search / memory_reindex / memory_prune
- Clases de compatibilidad para loops antiguos:
    EpisodicMemory (append/recent) y SemanticMemory (add_fact/search)

Usa un embedder simple de n-gramas (sin dependencias externas). Cuando conectes
embeddings reales, sustituye la clase _Vectorizer manteniendo la interfaz.
"""
from __future__ import annotations
import json, uuid, time, math, hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from root.settings import MEMORY_CONF, EPISODES_JSONL, SEMANTIC_MEMORY_JSON, EMBED_DIM

# -------------------- Embedder n-gram --------------------
class _Vectorizer:
    def __init__(self, dim: int = EMBED_DIM, ngram: int = 3):
        self.dim = int(dim); self.ngram = int(ngram)
    def _ngrams(self, t: str):
        t = (t or "").lower()
        if len(t) < self.ngram:
            if t: yield t; return
        for i in range(len(t) - self.ngram + 1):
            yield t[i:i+self.ngram]
    def embed_text(self, text: str) -> List[float]:
        if not text: return [0.0] * self.dim
        vec = [0.0] * self.dim
        for ng in self._ngrams(text):
            h = int(hashlib.md5(ng.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[h] += 1.0
        n = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x / n for x in vec]

_V = _Vectorizer(dim=EMBED_DIM, ngram=3)
def embed_text(text: str) -> List[float]: return _V.embed_text(text)
def _cos(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b): return 0.0
    return sum(x*y for x, y in zip(a, b))

# -------------------- Paths/IO --------------------
EPISODES_FILE = Path(EPISODES_JSONL)
MEM_INDEX_FILE = Path(SEMANTIC_MEMORY_JSON)

def _ensure_files():
    EPISODES_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not EPISODES_FILE.exists(): EPISODES_FILE.write_text("", encoding="utf-8")
    MEM_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not MEM_INDEX_FILE.exists():
        MEM_INDEX_FILE.write_text(json.dumps({"dim": EMBED_DIM, "items": []}, ensure_ascii=False, indent=2), encoding="utf-8")

def _now() -> int: return int(time.time())
def _load_json(p: Path, d: Any):
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return d
def _save_json(p: Path, o: Any): p.write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding="utf-8")
def _cutoff(days: int) -> int: return 0 if not days or days <= 0 else _now() - days * 86400

# -------------------- API funcional --------------------
def memory_write(text: str, user: str = "default", project_id: str = "default",
                 tags: Optional[List[str]] = None, importance: float = 0.5,
                 meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _ensure_files()
    eid = f"mem-{uuid.uuid4().hex[:12]}"; ts = _now()
    ep = {"id": eid, "text": text, "ts": ts, "user": user, "project_id": project_id,
          "tags": list(tags or []), "importance": float(importance), "meta": dict(meta or {})}
    # 1) Episodios
    with EPISODES_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    # 2) Índice semántico
    vec = embed_text(text)
    idx = _load_json(MEM_INDEX_FILE, {"dim": EMBED_DIM, "items": []})
    idx.setdefault("items", []).append({
        "id": eid, "project_id": project_id, "user": user, "tags": ep["tags"],
        "importance": ep["importance"], "ts": ts, "vec": vec,
        "preview": text[:200], "text": text, "meta": ep["meta"],
    })
    idx["dim"] = EMBED_DIM; _save_json(MEM_INDEX_FILE, idx)
    memory_prune()
    return {"ok": True, "id": eid, "ts": ts}

def memory_search(query: str, topk: int = 5, project_id: Optional[str] = None,
                  tags: Optional[List[str]] = None, user: Optional[str] = None) -> List[Dict[str, Any]]:
    _ensure_files()
    idx = _load_json(MEM_INDEX_FILE, {"dim": EMBED_DIM, "items": []})
    items = idx.get("items", [])
    if not items: return []
    qv = embed_text(query)

    def _ok(it):
        if project_id and it.get("project_id") != project_id: return False
        if user and it.get("user") != user: return False
        if tags and not set(tags).issubset(set(it.get("tags") or [])): return False
        return True

    scored = []
    for it in items:
        if not _ok(it): continue
        scored.append({
            "id": it["id"], "score": round(float(_cos(qv, it.get("vec") or [])), 6),
            "project_id": it.get("project_id"), "user": it.get("user"),
            "tags": it.get("tags") or [], "importance": it.get("importance", 0.5),
            "ts": it.get("ts"), "preview": it.get("preview", ""), "text": it.get("text", "")
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[: max(1, int(topk))]

def memory_reindex(project_id: Optional[str] = None, user: Optional[str] = None) -> Dict[str, Any]:
    _ensure_files()
    items, total = [], 0
    for line in EPISODES_FILE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s: continue
        total += 1
        try: ep = json.loads(s)
        except Exception: continue
        if project_id and ep.get("project_id") != project_id: continue
        if user and ep.get("user") != user: continue
        txt = ep.get("text", "")
        items.append({
            "id": ep.get("id"), "project_id": ep.get("project_id", "default"),
            "user": ep.get("user", "default"), "tags": ep.get("tags") or [],
            "importance": float(ep.get("importance", 0.5)), "ts": ep.get("ts"),
            "vec": embed_text(txt), "preview": txt[:200], "text": txt, "meta": ep.get("meta", {})
        })
    _save_json(MEM_INDEX_FILE, {"dim": EMBED_DIM, "items": items})
    memory_prune()
    return {"ok": True, "reindexed": len(items), "from": total,
            "project": project_id or "all", "user": user or "all"}

# -------------------- Retención / Rotación --------------------
def _prune_episodes(lines: List[str]) -> List[str]:
    cutoff = _cutoff(int(MEMORY_CONF.get("episodic_ttl_days", 0)))
    max_items = int(MEMORY_CONF.get("episodic_max_items", 0))
    kept: List[Tuple[int, str]] = []
    for s in lines:
        try:
            ep = json.loads(s); ts = int(ep.get("ts", 0))
            if cutoff and ts < cutoff: continue
            kept.append((ts, s))
        except Exception:
            continue
    kept.sort(key=lambda x: x[0], reverse=True)
    if max_items and len(kept) > max_items: kept = kept[:max_items]
    return [s for _, s in kept]

def _prune_semantic(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cutoff = _cutoff(int(MEMORY_CONF.get("semantic_ttl_days", 0)))
    max_items = int(MEMORY_CONF.get("semantic_max_items", 0))
    filtered = [it for it in items if not (cutoff and int(it.get("ts", 0)) < cutoff)]
    filtered.sort(key=lambda it: int(it.get("ts", 0)), reverse=True)
    if max_items and len(filtered) > max_items: filtered = filtered[:max_items]
    return filtered

def memory_prune() -> Dict[str, Any]:
    _ensure_files()
    lines = [ln.strip() for ln in EPISODES_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    new_lines = _prune_episodes(lines)
    if len(new_lines) != len(lines):
        EPISODES_FILE.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
    idx = _load_json(MEM_INDEX_FILE, {"dim": EMBED_DIM, "items": []})
    old_items = idx.get("items", [])
    new_items = _prune_semantic(old_items)
    if len(new_items) != len(old_items):
        idx["items"] = new_items; _save_json(MEM_INDEX_FILE, idx)
    return {"ok": True,
            "episodes_before": len(lines), "episodes_after": len(new_lines),
            "semantic_before": len(old_items), "semantic_after": len(new_items)}

# -------------------- Clases de compatibilidad --------------------
class EpisodicMemory:
    """
    Compat shim:
      - recent(k) -> lista de episodios dict
      - append(episode: dict) -> escribe en episodes.jsonl y no toca semántica
    """
    def __init__(self, path: str | Path = EPISODES_FILE):
        self.path = Path(path); _ensure_files()

    def recent(self, k: int = MEMORY_CONF.get("recent_k", 6)) -> List[Dict[str, Any]]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()[-int(k):]
        except Exception:
            return []
        out = []
        for ln in lines:
            try: out.append(json.loads(ln))
            except Exception: continue
        return out

    def append(self, episode: Dict[str, Any]) -> None:
        """Añade un episodio arbitrario (compatibilidad con loops antiguos)."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")

class SemanticMemory:
    """
    Compat shim:
      - add_fact(text, meta=None) -> id
      - search(query, top_k) -> lista con {id, text, meta, score}
    Comparte el mismo archivo SEMANTIC_MEMORY_JSON.
    """
    def __init__(self, path: str | Path = MEM_INDEX_FILE):
        self.path = Path(path); _ensure_files()

    def _load(self): return _load_json(self.path, {"dim": EMBED_DIM, "items": []})
    def _save(self, obj): _save_json(self.path, obj)

    def add_fact(self, text: str, meta: Optional[Dict[str, Any]] = None) -> str:
        vec = embed_text(text); fid = f"sm-{uuid.uuid4().hex[:10]}"; ts = _now()
        idx = self._load()
        idx.setdefault("items", []).append({"id": fid, "text": text, "meta": dict(meta or {}),
                                            "vec": vec, "ts": ts, "preview": text[:200]})
        idx["dim"] = EMBED_DIM; self._save(idx)
        # NOTA: no aplicamos prune aquí para no interferir con tu política de “hechos”.
        return fid

    def search(self, query: str, top_k: int = MEMORY_CONF.get("semantic_top_k", 6)) -> List[Dict[str, Any]]:
        idx = self._load(); items = idx.get("items", []); 
        if not items: return []
        qv = embed_text(query)
        scored = []
        for it in items:
            txt = it.get("text") or it.get("preview","")
            sc = _cos(qv, it.get("vec") or [])
            scored.append({"id": it.get("id"), "text": txt, "meta": it.get("meta", {}),
                           "score": float(round(sc, 6))})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: max(1, int(top_k))]

__all__ = [
    # API funcional
    "memory_write","memory_search","memory_reindex","memory_prune","embed_text",
    # Clases shim
    "EpisodicMemory","SemanticMemory",
]
