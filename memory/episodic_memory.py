# memory/episodic_memory.py
# Memory 2.2.3 — memoria episódica estructurada + recuperación híbrida.
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .embeddings_local import embed as semantic_embed
except Exception:
    semantic_embed = None

try:
    from root import settings as S  # opcional
except Exception:
    S = None

PROJ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(getattr(S, "DATA_DIR", PROJ_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

EPISODES_JSONL = str(
    getattr(S, "EPISODES_JSONL", DATA_DIR / "episodes.jsonl")
)
SEMANTIC_MEMORY_JSON = str(
    getattr(S, "SEMANTIC_MEMORY_JSON", DATA_DIR / "semantic_memory.json")
)

EPISODES_FILE = Path(EPISODES_JSONL)
MEM_INDEX_FILE = Path(SEMANTIC_MEMORY_JSON)

_DEFAULT_MC = {
    "episodic_ttl_days": 0,
    "episodic_max_items": 0,
    "semantic_ttl_days": 0,
    "semantic_max_items": 0,
    "recent_k": 6,
    "semantic_top_k": 6,
}
MEMORY_CONF: Dict[str, Any] = dict(_DEFAULT_MC)

if getattr(S, "MEMORY_CONF", None):
    MEMORY_CONF.update(getattr(S, "MEMORY_CONF"))

# Palabras equivalentes que ayudan a recuperar un episodio aunque la
# consulta use una formulación distinta.
_FIELD_ALIASES: Dict[str, str] = {
    "proyecto": "proyecto iniciativa plan trabajo",
    "codigo": "código codigo clave alfanumérica alfanumerica identificador",
    "color": "color tono tonalidad",
    "numero": "número numero cifra cantidad",
    "lugar": "lugar sitio ubicación ubicacion espacio",
    "dia": "día dia fecha jornada",
    "objeto": "objeto elemento artículo articulo",
    "clave": "clave código codigo identificador contraseña contrasena",
    "estacion": "estación estacion parada terminal",
    "llave": "llave clave código codigo",
    "animal": "animal especie",
    "cantidad": "cantidad cifra número numero",
    "nombre": "nombre título titulo etiqueta",
}

_MEMORY_INTENT = (
    "recuerda",
    "recordar",
    "guarda",
    "guardar",
    "conserva",
    "conservar",
    "memoriza",
    "memorizar",
    "para más adelante",
    "para mas adelante",
    "para una prueba de memoria",
)

def _now() -> int:
    return int(time.time())

def _norm_key(value: str) -> str:
    value = (value or "").strip().lower()
    value = (
        value.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ü", "u")
        .replace("ñ", "n")
    )
    return re.sub(r"[^a-z0-9_]+", "_", value).strip("_")

_CONTROL_FIELD_PREFIXES = (
    "responde",
    "respondeme",
    "respuesta",
    "contesta",
    "contestame",
    "reply",
    "answer",
)

_CONTROL_FIELD_EXACT = {
    "respondeme_unicamente",
    "responde_unicamente",
    "responde_solo",
    "respondeme_solo",
    "respuesta_esperada",
    "formato_de_respuesta",
    "instruccion",
    "instrucciones",
}

def _is_control_field(key: str) -> bool:
    normalized = _norm_key(key)
    if normalized in _CONTROL_FIELD_EXACT:
        return True
    return any(normalized.startswith(prefix) for prefix in _CONTROL_FIELD_PREFIXES)

def _sanitize_fields(fields: Dict[str, Any]) -> Dict[str, str]:
    """Elimina pseudo-campos de instrucciones, incluso de episodios antiguos."""
    clean: Dict[str, str] = {}
    for raw_key, raw_value in dict(fields or {}).items():
        key = _norm_key(str(raw_key))
        value = str(raw_value).strip()
        if not key or not value or _is_control_field(key):
            continue
        clean[key] = value
    return clean


def _tokens(text: str) -> List[str]:
    return [
        t
        for t in re.split(
            r"[^\wáéíóúñü]+",
            (text or "").lower(),
        )
        if t
    ]

def extract_fields(text: str) -> Dict[str, str]:
    """Extrae campos factuales; descarta instrucciones de control."""
    fields: Dict[str, str] = {}
    for raw_line in (text or "").splitlines():
        line = raw_line.strip().lstrip("-*•").strip()
        if not line:
            continue
        match = re.match(
            r"^([A-Za-zÁÉÍÓÚáéíóúÑñÜü][\wÁÉÍÓÚáéíóúÑñÜü -]{0,40}?)"
            r"\s*(?:=|:)\s*(.+?)\s*$",
            line,
        )
        if not match:
            continue
        key = _norm_key(match.group(1))
        value = match.group(2).strip().rstrip(".,;")
        if not key or not value or _is_control_field(key):
            continue
        fields[key] = value
    return fields


def looks_like_memory_request(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _MEMORY_INTENT)

def looks_like_recall_request(text: str) -> bool:
    lowered = (text or "").lower()
    markers = (
        "recupera",
        "recuperar",
        "¿recuerdas",
        "recuerdas",
        "qué datos",
        "que datos",
        "dime los datos",
        "episodio",
        "memoria anterior",
        "memorias anteriores",
        "recuerdo anterior",
        "recuerdos anteriores",
        "sin que vuelva a darte",
        "sin que vuelva a proporcionarte",
        "sin que vuelva a dar",
    )
    return any(marker in lowered for marker in markers)

def factual_payload(text: str) -> str:
    """Devuelve solo pares factuales clave = valor."""
    fields = extract_fields(text)
    return "\n".join(f"{key} = {value}" for key, value in fields.items())


def derive_episode_key(
    fields: Dict[str, str],
    text: str,
) -> str:
    priority = (
        "proyecto",
        "lugar",
        "estacion",
        "nombre",
        "codigo",
        "clave",
        "llave",
    )

    for key in priority:
        value = fields.get(key)
        if value:
            return value

    # Si no hay campos, intenta tomar una etiqueta después de "sobre".
    match = re.search(
        r"\bsobre\s+([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚáéíóúÑñÜü -]{1,50})",
        text or "",
    )
    if match:
        return match.group(1).strip(" .,:;")

    return "episodio"

def enrich_episode_text(
    text: str,
    fields: Optional[Dict[str, str]] = None,
    episode_key: Optional[str] = None,
) -> str:
    fields = dict(fields or extract_fields(text))
    pieces = [text.strip()]

    if episode_key:
        pieces.append(f"tema episodio nombre etiqueta: {episode_key}")

    for key, value in fields.items():
        aliases = _FIELD_ALIASES.get(key, key.replace("_", " "))
        pieces.append(f"{aliases}: {value}")

    return "\n".join(p for p in pieces if p)

# -------------------------------------------------------------------
# Embeddings: SentenceTransformer si está disponible; fallback n-gram
# -------------------------------------------------------------------

class _FallbackVectorizer:
    def __init__(self, dim: int = 768, ngram: int = 3) -> None:
        self.dim = int(dim)
        self.ngram = int(ngram)

    def _ngrams(self, text: str):
        text = (text or "").lower()
        if len(text) < self.ngram:
            if text:
                yield text
            return
        for i in range(len(text) - self.ngram + 1):
            yield text[i : i + self.ngram]

    def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim

        for gram in self._ngrams(text):
            pos = (
                int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                % self.dim
            )
            vec[pos] += 1.0

        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

_FALLBACK = _FallbackVectorizer()

def _semantic_vectors(texts: List[str]) -> Optional[List[List[float]]]:
    if semantic_embed is None:
        return None

    try:
        vectors = semantic_embed(texts)
        if vectors and len(vectors) == len(texts):
            return vectors
    except Exception:
        pass

    return None

def embed_text(text: str) -> List[float]:
    vectors = _semantic_vectors([text])

    if vectors:
        return list(vectors[0])

    return _FALLBACK.embed_text(text)

def _cos(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))

# -------------------------------------------------------------------
# IO
# -------------------------------------------------------------------

def _ensure_files() -> None:
    EPISODES_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEM_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not EPISODES_FILE.exists():
        EPISODES_FILE.write_text("", encoding="utf-8")

    if not MEM_INDEX_FILE.exists():
        MEM_INDEX_FILE.write_text(
            json.dumps(
                {"version": "2.2.3", "items": []},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def _save_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def _load_store() -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    _ensure_files()
    obj = _load_json(
        MEM_INDEX_FILE,
        {"version": "2.2.3", "items": []},
    )

    if isinstance(obj, list):
        return "list", obj, {}

    return "dict", list(obj.get("items", [])), dict(obj)

def _save_store(
    shape: str,
    items: List[Dict[str, Any]],
    base: Dict[str, Any],
) -> None:
    if shape == "list":
        _save_json(MEM_INDEX_FILE, items)
        return

    base = dict(base or {})
    base["version"] = "2.2.3"
    base["items"] = items

    dims = {
        len(it.get("vec") or [])
        for it in items
        if it.get("vec")
    }
    base["dims"] = sorted(dims)

    _save_json(MEM_INDEX_FILE, base)

def _fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

# -------------------------------------------------------------------
# API
# -------------------------------------------------------------------

def memory_write(
    text: str,
    user: str = "default",
    project_id: str = "default",
    tags: Optional[List[str]] = None,
    importance: float = 0.8,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _ensure_files()

    text = str(text or "").strip()

    if not text:
        return {"ok": False, "reason": "empty"}

    fields = extract_fields(text)
    md = dict(meta or {})

    payload = factual_payload(text)
    if payload:
        text_for_storage = payload
    else:
        text_for_storage = text
    md_fields = md.get("fields")

    if isinstance(md_fields, dict):
        fields.update(_sanitize_fields(md_fields))

    fields = _sanitize_fields(fields)

    episode_key = str(
        md.get("episode_key")
        or derive_episode_key(fields, text)
    )

    semantic_text = enrich_episode_text(
        text_for_storage,
        fields=fields,
        episode_key=episode_key,
    )

    fp = _fingerprint(text_for_storage)
    shape, items, base = _load_store()

    # Evita duplicar exactamente el mismo episodio.
    for item in items:
        if item.get("fingerprint") == fp:
            return {
                "ok": True,
                "id": item.get("id"),
                "ts": item.get("ts"),
                "duplicate": True,
                "episode_key": item.get("episode_key"),
            }

    eid = f"mem-{uuid.uuid4().hex[:12]}"
    ts = _now()
    vec = embed_text(semantic_text)

    episode = {
        "id": eid,
        "text": text_for_storage,
        "raw_text": text,
        "semantic_text": semantic_text,
        "fields": fields,
        "episode_key": episode_key,
        "ts": ts,
        "user": user,
        "project_id": project_id,
        "tags": list(tags or []),
        "importance": float(importance),
        "meta": md,
        "fingerprint": fp,
    }

    with EPISODES_FILE.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(episode, ensure_ascii=False) + "\n"
        )

    items.append(
        {
            **episode,
            "vec": vec,
            "preview": text[:300],
        }
    )
    _save_store(shape, items, base)
    memory_prune()

    return {
        "ok": True,
        "id": eid,
        "ts": ts,
        "episode_key": episode_key,
        "fields": fields,
    }

def memory_search(
    query: str,
    topk: int = 5,
    project_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    user: Optional[str] = None,
) -> List[Dict[str, Any]]:
    _ensure_files()
    _, items, _ = _load_store()

    if not items:
        return []

    filtered: List[Dict[str, Any]] = []

    for item in items:
        if project_id and item.get("project_id") != project_id:
            continue
        if user and item.get("user") != user:
            continue
        if tags and not set(tags).issubset(set(item.get("tags") or [])):
            continue
        filtered.append(item)

    if not filtered:
        return []

    enriched_query = enrich_episode_text(query, fields={})

    sanitized_items: List[Dict[str, Any]] = []
    for item in filtered:
        clean_item = dict(item)
        clean_fields = _sanitize_fields(
            dict(item.get("fields") or extract_fields(str(item.get("text") or "")))
        )
        clean_item["fields"] = clean_fields
        clean_text = factual_payload(str(item.get("text") or ""))
        if not clean_text and clean_fields:
            clean_text = "\n".join(
                f"{key} = {value}" for key, value in clean_fields.items()
            )
        clean_item["text"] = clean_text or str(item.get("text") or "")
        clean_item["semantic_text"] = enrich_episode_text(
            clean_item["text"],
            fields=clean_fields,
            episode_key=str(item.get("episode_key") or "episodio"),
        )
        sanitized_items.append(clean_item)

    filtered = sanitized_items
    batch = [enriched_query] + [
        str(it.get("semantic_text") or it.get("text") or "")
        for it in filtered
    ]
    vectors = _semantic_vectors(batch)

    if vectors:
        qv = vectors[0]
        item_vectors = vectors[1:]
    else:
        qv = _FALLBACK.embed_text(enriched_query)
        item_vectors = [
            _FALLBACK.embed_text(
                str(it.get("semantic_text") or it.get("text") or "")
            )
            for it in filtered
        ]

    q_tokens = set(_tokens(enriched_query))
    now = max(1, _now())
    scored: List[Dict[str, Any]] = []

    for item, vec in zip(filtered, item_vectors):
        semantic_score = _cos(qv, vec)
        item_text = str(
            item.get("semantic_text")
            or item.get("text")
            or ""
        )
        overlap = len(q_tokens & set(_tokens(item_text)))

        episode_key = str(item.get("episode_key") or "")
        key_bonus = 0.0

        if episode_key and episode_key.lower() in query.lower():
            key_bonus = 0.35

        importance = float(item.get("importance", 0.8))
        age_days = max(
            0.0,
            (now - int(item.get("ts") or now)) / 86400.0,
        )
        recency = math.exp(-age_days / 90.0) * 0.03

        score = (
            semantic_score * 0.72
            + min(overlap, 8) / 8.0 * 0.18
            + key_bonus
            + min(max(importance, 0.0), 1.0) * 0.07
            + recency
        )

        scored.append(
            {
                "id": item.get("id"),
                "score": float(round(score, 6)),
                "project_id": item.get("project_id"),
                "user": item.get("user"),
                "tags": item.get("tags") or [],
                "importance": importance,
                "ts": item.get("ts"),
                "preview": item.get("preview", ""),
                "text": item.get("text", ""),
                "fields": _sanitize_fields(dict(item.get("fields") or {})),
                "episode_key": item.get("episode_key") or "episodio",
                "meta": dict(item.get("meta") or {}),
                "source": "episode",
            }
        )

    scored.sort(
        key=lambda x: (
            x.get("score", 0.0),
            x.get("ts", 0),
        ),
        reverse=True,
    )

    return scored[: max(1, int(topk or 5))]

def memory_reindex(
    project_id: Optional[str] = None,
    user: Optional[str] = None,
) -> Dict[str, Any]:
    _ensure_files()

    lines = [
        line.strip()
        for line in EPISODES_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    items: List[Dict[str, Any]] = []
    total = 0

    for raw in lines:
        total += 1

        try:
            ep = json.loads(raw)
        except Exception:
            continue

        if project_id and ep.get("project_id") != project_id:
            continue
        if user and ep.get("user") != user:
            continue

        text = str(ep.get("text") or "")
        fields = _sanitize_fields(
            dict(ep.get("fields") or extract_fields(text))
        )
        clean_text = factual_payload(text)
        if not clean_text and fields:
            clean_text = "\n".join(
                f"{key} = {value}" for key, value in fields.items()
            )
        if clean_text:
            text = clean_text

        episode_key = str(
            ep.get("episode_key")
            or derive_episode_key(fields, text)
        )
        semantic_text = enrich_episode_text(
            text,
            fields=fields,
            episode_key=episode_key,
        )

        ep["fields"] = fields
        ep["episode_key"] = episode_key
        ep["semantic_text"] = semantic_text
        ep["fingerprint"] = ep.get("fingerprint") or _fingerprint(text)
        ep["vec"] = embed_text(semantic_text)
        ep["preview"] = text[:300]
        items.append(ep)

    shape, _, base = _load_store()
    _save_store(shape, items, base)
    memory_prune()

    return {
        "ok": True,
        "reindexed": len(items),
        "from": total,
        "project": project_id or "all",
        "user": user or "all",
    }

# -------------------------------------------------------------------
# Retención
# -------------------------------------------------------------------

def _cutoff(days: int) -> int:
    return 0 if not days or days <= 0 else _now() - days * 86400

def memory_prune() -> Dict[str, Any]:
    _ensure_files()

    cutoff_ep = _cutoff(
        int(MEMORY_CONF.get("episodic_ttl_days", 0))
    )
    max_ep = int(MEMORY_CONF.get("episodic_max_items", 0))

    lines = [
        line.strip()
        for line in EPISODES_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    episodes: List[Tuple[int, str]] = []

    for raw in lines:
        try:
            ep = json.loads(raw)
            ts = int(ep.get("ts", 0))
            if cutoff_ep and ts < cutoff_ep:
                continue
            episodes.append((ts, raw))
        except Exception:
            continue

    episodes.sort(key=lambda pair: pair[0], reverse=True)

    if max_ep and len(episodes) > max_ep:
        episodes = episodes[:max_ep]

    new_lines = [raw for _, raw in episodes]

    if len(new_lines) != len(lines):
        EPISODES_FILE.write_text(
            "\n".join(new_lines)
            + ("\n" if new_lines else ""),
            encoding="utf-8",
        )

    shape, items, base = _load_store()
    cutoff_sem = _cutoff(
        int(MEMORY_CONF.get("semantic_ttl_days", 0))
    )
    max_sem = int(MEMORY_CONF.get("semantic_max_items", 0))

    sem_items = [
        it
        for it in items
        if not (
            cutoff_sem
            and int(it.get("ts", 0)) < cutoff_sem
        )
    ]
    sem_items.sort(
        key=lambda it: int(it.get("ts", 0)),
        reverse=True,
    )

    if max_sem and len(sem_items) > max_sem:
        sem_items = sem_items[:max_sem]

    if len(sem_items) != len(items):
        _save_store(shape, sem_items, base)

    return {
        "ok": True,
        "episodes_before": len(lines),
        "episodes_after": len(new_lines),
        "semantic_before": len(items),
        "semantic_after": len(sem_items),
    }

# -------------------------------------------------------------------
# Compatibilidad
# -------------------------------------------------------------------

class EpisodicMemory:
    def __init__(
        self,
        path: str | Path = EPISODES_FILE,
    ) -> None:
        self.path = Path(path)
        _ensure_files()

    def recent(
        self,
        k: int = int(MEMORY_CONF.get("recent_k", 6)),
    ) -> List[Dict[str, Any]]:
        try:
            lines = self.path.read_text(
                encoding="utf-8"
            ).splitlines()[-int(k):]
        except Exception:
            return []

        out: List[Dict[str, Any]] = []

        for line in lines:
            try:
                out.append(json.loads(line))
            except Exception:
                continue

        return out

    def append(self, episode: Dict[str, Any]) -> None:
        text = str(episode.get("text") or "")
        memory_write(
            text,
            user=str(episode.get("user", "default")),
            project_id=str(
                episode.get("project_id", "default")
            ),
            tags=list(episode.get("tags") or []),
            importance=float(
                episode.get("importance", 0.8)
            ),
            meta=dict(episode.get("meta") or {}),
        )

class SemanticMemory:
    def __init__(
        self,
        path: str | Path = MEM_INDEX_FILE,
    ) -> None:
        self.path = Path(path)
        _ensure_files()

    def add_fact(
        self,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        result = memory_write(
            text,
            tags=["semantic_fact"],
            meta=meta,
        )
        return str(result.get("id") or "")

    def search(
        self,
        query: str,
        top_k: int = int(
            MEMORY_CONF.get("semantic_top_k", 6)
        ),
    ) -> List[Dict[str, Any]]:
        return memory_search(query, topk=top_k)

__all__ = [
    "embed_text",
    "extract_fields",
    "_is_control_field",
    "_sanitize_fields",
    "looks_like_memory_request",
    "looks_like_recall_request",
    "factual_payload",
    "derive_episode_key",
    "enrich_episode_text",
    "memory_write",
    "memory_search",
    "memory_reindex",
    "memory_prune",
    "EpisodicMemory",
    "SemanticMemory",
]
