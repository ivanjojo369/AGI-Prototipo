from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from .vector_store import SimpleVectorStore

try:
    from .embeddings_local import embed as semantic_embed
except Exception:
    semantic_embed = None


_WORD = re.compile(
    r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9_-]+",
    re.UNICODE,
)

_CONTROL_PREFIXES = (
    "responde",
    "respondeme",
    "respuesta",
    "contesta",
    "contestame",
    "reply",
    "answer",
)


def _now() -> int:
    return int(time.time())


def _tokens(text: str) -> List[str]:
    return [
        token.lower()
        for token in _WORD.findall(text or "")
        if token
    ]


def _norm_text(text: str) -> str:
    value = (text or "").strip().lower()
    value = (
        value.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ü", "u")
        .replace("ñ", "n")
    )
    return re.sub(r"\s+", " ", value)


def _norm_key(text: str) -> str:
    value = _norm_text(text)
    return re.sub(
        r"[^a-z0-9_]+",
        "_",
        value,
    ).strip("_")


def _is_control_field(key: str) -> bool:
    normalized = _norm_key(key)
    return any(
        normalized.startswith(prefix)
        for prefix in _CONTROL_PREFIXES
    )


def _safe_fields(
    fields: Dict[str, Any],
) -> Dict[str, str]:
    out: Dict[str, str] = {}

    for raw_key, raw_value in dict(fields or {}).items():
        key = _norm_key(str(raw_key))
        value = str(raw_value).strip()

        if not key or not value:
            continue

        if _is_control_field(key):
            continue

        out[key] = value

    return out


def _cosine(
    a: List[float],
    b: List[float],
) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    return float(
        sum(x * y for x, y in zip(a, b))
    )


class SemanticMemory:
    """
    Memory 2.3 — Consolidación Semántica.

    Convierte recuerdos episódicos estructurados en conocimiento:
    - un slot estable por sujeto + predicado;
    - refuerzo mediante fuentes independientes;
    - confidence basada en support_count;
    - procedencia por source_id;
    - historial cuando un valor cambia;
    - recuperación con embeddings locales si existen;
    - fallback TF-IDF si no existen;
    - sin FAISS obligatorio.

    Persistencia:
        data/semantic_store.json

    SimpleVectorStore conserva compatibilidad con documentos legacy.
    """

    VERSION = "2.3"
    FACT_SCHEMA = "semantic_fact_v2.3"

    def __init__(
        self,
        base_dir: str = "data",
        *,
        initial_confidence: float = 0.65,
        reinforcement_step: float = 0.10,
        max_confidence: float = 0.98,
    ) -> None:
        self.base_dir = str(base_dir)
        self.store_path = os.path.join(
            self.base_dir,
            "semantic_store.json",
        )
        os.makedirs(
            self.base_dir,
            exist_ok=True,
        )

        self.store = SimpleVectorStore(
            self.store_path
        )

        self.initial_confidence = float(
            initial_confidence
        )
        self.reinforcement_step = float(
            reinforcement_step
        )
        self.max_confidence = float(
            max_confidence
        )

        self._doc_tf: Dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._idf: Dict[str, float] = {}
        self._doc_vec: Dict[
            str,
            Dict[str, float],
        ] = {}

        self._rebuild_lexical_index()

    # ----------------------------------------------------------
    # Identidad y utilidades
    # ----------------------------------------------------------

    @staticmethod
    def _fact_id(
        subject: str,
        predicate: str,
    ) -> str:
        raw = (
            f"{_norm_text(subject)}|"
            f"{_norm_key(predicate)}"
        ).encode("utf-8")

        digest = hashlib.sha256(
            raw
        ).hexdigest()[:20]

        return f"fact:{digest}"

    @staticmethod
    def _fact_text(
        subject: str,
        predicate: str,
        value: str,
    ) -> str:
        return (
            f"{subject} — "
            f"{predicate} = {value}"
        )

    def _confidence_for_support(
        self,
        support_count: int,
    ) -> float:
        support = max(
            1,
            int(support_count),
        )

        confidence = (
            self.initial_confidence
            + self.reinforcement_step
            * (support - 1)
        )

        return float(
            round(
                min(
                    self.max_confidence,
                    confidence,
                ),
                4,
            )
        )

    @classmethod
    def _is_structured_fact(
        cls,
        doc: Dict[str, Any],
    ) -> bool:
        meta = dict(
            doc.get("meta") or {}
        )

        return (
            meta.get("schema")
            == cls.FACT_SCHEMA
        )

    @staticmethod
    def _source_seen(
        meta: Dict[str, Any],
        source: str,
    ) -> bool:
        if source in list(
            meta.get("sources") or []
        ):
            return True

        for entry in list(
            meta.get("history") or []
        ):
            if source in list(
                entry.get("sources") or []
            ):
                return True

        return False

    # ----------------------------------------------------------
    # TF-IDF fallback
    # ----------------------------------------------------------

    def _rebuild_lexical_index(
        self,
    ) -> None:
        self._doc_tf.clear()
        self._df.clear()
        self._idf.clear()
        self._doc_vec.clear()

        docs = self.store.items()

        for doc in docs:
            doc_id = str(
                doc.get("id") or ""
            )
            tf = Counter(
                _tokens(
                    str(
                        doc.get("text")
                        or ""
                    )
                )
            )
            self._doc_tf[doc_id] = tf

            for token in tf:
                self._df[token] += 1

        n_docs = max(
            1,
            len(docs),
        )

        for token, df in self._df.items():
            self._idf[token] = (
                math.log(
                    (1 + n_docs)
                    / (1 + df)
                )
                + 1.0
            )

        for doc_id, tf in (
            self._doc_tf.items()
        ):
            self._doc_vec[
                doc_id
            ] = self._tfidf_norm(tf)

    def _tfidf_norm(
        self,
        tf: Counter[str],
    ) -> Dict[str, float]:
        vec: Dict[str, float] = {}

        for token, freq in tf.items():
            vec[token] = (
                float(freq)
                * float(
                    self._idf.get(
                        token,
                        0.0,
                    )
                )
            )

        norm = math.sqrt(
            sum(
                value * value
                for value in vec.values()
            )
        ) or 1.0

        return {
            token: value / norm
            for token, value in vec.items()
        }

    @staticmethod
    def _sparse_cos(
        a: Dict[str, float],
        b: Dict[str, float],
    ) -> float:
        if len(a) > len(b):
            a, b = b, a

        return float(
            sum(
                a.get(token, 0.0)
                * b.get(token, 0.0)
                for token in a
            )
        )

    # ----------------------------------------------------------
    # Consolidación
    # ----------------------------------------------------------

    def consolidate_fact(
        self,
        subject: str,
        predicate: str,
        value: str,
        *,
        source_id: Optional[str] = None,
        source_ts: Optional[int] = None,
        source_kind: str = "episode",
    ) -> Dict[str, Any]:
        subject = str(
            subject or ""
        ).strip()
        predicate = _norm_key(
            str(predicate or "")
        )
        value = str(
            value or ""
        ).strip()

        if (
            not subject
            or not predicate
            or not value
        ):
            return {
                "ok": False,
                "reason": "missing_fact_parts",
            }

        if _is_control_field(predicate):
            return {
                "ok": False,
                "reason": "control_field",
            }

        fact_id = self._fact_id(
            subject,
            predicate,
        )
        now = int(
            source_ts or _now()
        )

        source = str(
            source_id
            or (
                "derived:"
                + hashlib.sha1(
                    (
                        subject
                        + predicate
                        + value
                    ).encode("utf-8")
                ).hexdigest()[:12]
            )
        )

        existing = self.store.get(
            fact_id
        )

        if (
            not existing
            or not self._is_structured_fact(
                existing
            )
        ):
            meta = {
                "schema": self.FACT_SCHEMA,
                "version": self.VERSION,
                "subject": subject,
                "subject_norm": _norm_text(
                    subject
                ),
                "predicate": predicate,
                "value": value,
                "value_norm": _norm_text(
                    value
                ),
                "confidence":
                    self._confidence_for_support(
                        1
                    ),
                "support_count": 1,
                "sources": [source],
                "source_kind": source_kind,
                "created_at": now,
                "updated_at": now,
                "revision_count": 0,
                "status": "active",
                "history": [],
            }

            self.store.upsert(
                {
                    "id": fact_id,
                    "text": self._fact_text(
                        subject,
                        predicate,
                        value,
                    ),
                    "tags": [
                        "semantic_fact",
                        "active",
                        self.VERSION,
                    ],
                    "meta": meta,
                }
            )
            self._rebuild_lexical_index()

            return {
                "ok": True,
                "action": "created",
                "id": fact_id,
                **meta,
            }

        doc = dict(existing)
        meta = dict(
            doc.get("meta") or {}
        )

        old_value = str(
            meta.get("value") or ""
        )
        old_value_norm = _norm_text(
            old_value
        )
        new_value_norm = _norm_text(
            value
        )
        sources = list(
            meta.get("sources") or []
        )
        support_count = int(
            meta.get("support_count")
            or 1
        )
        updated_at = int(
            meta.get("updated_at")
            or 0
        )
        history = list(
            meta.get("history") or []
        )

        # Mismo valor: una nueva fuente refuerza.
        if old_value_norm == new_value_norm:
            changed = False

            if source not in sources:
                sources.append(source)
                support_count += 1
                changed = True

            meta["sources"] = sources
            meta["support_count"] = (
                support_count
            )
            meta["confidence"] = (
                self._confidence_for_support(
                    support_count
                )
            )
            meta["updated_at"] = max(
                updated_at,
                now,
            )

            if (
                changed
                or now > updated_at
            ):
                doc["meta"] = meta
                self.store.upsert(doc)
                self._rebuild_lexical_index()

            return {
                "ok": True,
                "action": (
                    "reinforced"
                    if changed
                    else "duplicate_source"
                ),
                "id": fact_id,
                **meta,
            }

        # La source ya existe en el historial:
        # migración totalmente idempotente.
        if self._source_seen(
            meta,
            source,
        ):
            return {
                "ok": True,
                "action": "duplicate_source",
                "id": fact_id,
                **meta,
            }

        # Observación antigua: se registra pero no sustituye lo actual.
        if (
            updated_at
            and now < updated_at
        ):
            history.append(
                {
                    "value": value,
                    "value_norm":
                        new_value_norm,
                    "confidence":
                        self._confidence_for_support(
                            1
                        ),
                    "support_count": 1,
                    "sources": [source],
                    "observed_at": now,
                    "status":
                        "historical_observation",
                }
            )

            meta["history"] = (
                history[-50:]
            )
            doc["meta"] = meta
            self.store.upsert(doc)
            self._rebuild_lexical_index()

            return {
                "ok": True,
                "action": "historical_only",
                "id": fact_id,
                **meta,
            }

        # Valor nuevo y más reciente:
        # el antiguo se conserva en history.
        history.append(
            {
                "value": old_value,
                "value_norm":
                    old_value_norm,
                "confidence": float(
                    meta.get("confidence")
                    or self.initial_confidence
                ),
                "support_count":
                    support_count,
                "sources": sources,
                "valid_from": int(
                    meta.get("created_at")
                    or updated_at
                    or now
                ),
                "valid_to": now,
                "status": "superseded",
            }
        )

        meta.update(
            {
                "value": value,
                "value_norm":
                    new_value_norm,
                "confidence":
                    self._confidence_for_support(
                        1
                    ),
                "support_count": 1,
                "sources": [source],
                "updated_at": now,
                "revision_count": (
                    int(
                        meta.get(
                            "revision_count"
                        )
                        or 0
                    )
                    + 1
                ),
                "status": "active",
                "history": history[-50:],
            }
        )

        doc["text"] = self._fact_text(
            subject,
            predicate,
            value,
        )
        doc["tags"] = [
            "semantic_fact",
            "active",
            self.VERSION,
        ]
        doc["meta"] = meta

        self.store.upsert(doc)
        self._rebuild_lexical_index()

        return {
            "ok": True,
            "action": "updated",
            "id": fact_id,
            **meta,
        }

    def consolidate_episode(
        self,
        episode: Dict[str, Any],
    ) -> Dict[str, Any]:
        episode = dict(
            episode or {}
        )
        fields = _safe_fields(
            dict(
                episode.get("fields")
                or {}
            )
        )

        if not fields:
            return {
                "ok": False,
                "reason":
                    "episode_without_fields",
                "facts": [],
            }

        subject = str(
            episode.get("episode_key")
            or ""
        ).strip()

        if (
            not subject
            or _norm_text(subject)
            == "episodio"
        ):
            for key in (
                "proyecto",
                "lugar",
                "estacion",
                "nombre",
            ):
                if fields.get(key):
                    subject = fields[key]
                    break

        if (
            not subject
            or _norm_text(subject)
            == "episodio"
        ):
            return {
                "ok": False,
                "reason":
                    "episode_without_subject",
                "facts": [],
            }

        source_id = str(
            episode.get("id")
            or (
                "episode:"
                + hashlib.sha1(
                    json.dumps(
                        fields,
                        sort_keys=True,
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest()[:12]
            )
        )

        source_ts = int(
            episode.get("ts")
            or _now()
        )

        results: List[
            Dict[str, Any]
        ] = []

        for (
            predicate,
            value,
        ) in fields.items():
            results.append(
                self.consolidate_fact(
                    subject,
                    predicate,
                    value,
                    source_id=source_id,
                    source_ts=source_ts,
                    source_kind="episode",
                )
            )

        return {
            "ok": True,
            "episode_key": subject,
            "source_id": source_id,
            "facts": results,
        }

    def consolidate_from_jsonl(
        self,
        path: str | Path,
    ) -> Dict[str, Any]:
        episode_path = Path(path)

        if not episode_path.exists():
            return {
                "ok": True,
                "episodes": 0,
                "facts_created": 0,
                "facts_reinforced": 0,
                "facts_updated": 0,
                "historical_only": 0,
                "duplicates": 0,
            }

        episodes: List[
            Dict[str, Any]
        ] = []

        for raw in (
            episode_path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        ):
            raw = raw.strip()

            if not raw:
                continue

            try:
                item = json.loads(raw)
            except Exception:
                continue

            if isinstance(item, dict):
                episodes.append(item)

        # Orden temporal ascendente.
        episodes.sort(
            key=lambda item: int(
                item.get("ts") or 0
            )
        )

        stats = {
            "ok": True,
            "episodes": 0,
            "facts_created": 0,
            "facts_reinforced": 0,
            "facts_updated": 0,
            "historical_only": 0,
            "duplicates": 0,
        }

        for episode in episodes:
            result = (
                self.consolidate_episode(
                    episode
                )
            )

            if not result.get("ok"):
                continue

            stats["episodes"] += 1

            for fact in (
                result.get("facts")
                or []
            ):
                action = fact.get(
                    "action"
                )

                if action == "created":
                    stats[
                        "facts_created"
                    ] += 1
                elif action == "reinforced":
                    stats[
                        "facts_reinforced"
                    ] += 1
                elif action == "updated":
                    stats[
                        "facts_updated"
                    ] += 1
                elif action == "historical_only":
                    stats[
                        "historical_only"
                    ] += 1
                elif action == "duplicate_source":
                    stats[
                        "duplicates"
                    ] += 1

        return stats

    # ----------------------------------------------------------
    # Consulta estructurada
    # ----------------------------------------------------------

    def facts_for_subject(
        self,
        subject: str,
        *,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        target = _norm_text(
            subject
        )
        out: List[
            Dict[str, Any]
        ] = []

        for doc in self.store.items():
            if not self._is_structured_fact(
                doc
            ):
                continue

            meta = dict(
                doc.get("meta")
                or {}
            )

            if (
                _norm_text(
                    str(
                        meta.get(
                            "subject"
                        )
                        or ""
                    )
                )
                != target
            ):
                continue

            if (
                active_only
                and meta.get("status")
                != "active"
            ):
                continue

            out.append(
                {
                    "doc_id":
                        doc.get("id"),
                    "text":
                        doc.get(
                            "text",
                            "",
                        ),
                    "subject":
                        meta.get(
                            "subject"
                        ),
                    "predicate":
                        meta.get(
                            "predicate"
                        ),
                    "value":
                        meta.get(
                            "value"
                        ),
                    "confidence":
                        float(
                            meta.get(
                                "confidence"
                            )
                            or 0.0
                        ),
                    "support_count":
                        int(
                            meta.get(
                                "support_count"
                            )
                            or 1
                        ),
                    "sources":
                        list(
                            meta.get(
                                "sources"
                            )
                            or []
                        ),
                    "status":
                        meta.get(
                            "status"
                        ),
                    "history":
                        list(
                            meta.get(
                                "history"
                            )
                            or []
                        ),
                }
            )

        out.sort(
            key=lambda item: str(
                item.get("predicate")
                or ""
            )
        )

        return out

    def get_fact(
        self,
        subject: str,
        predicate: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        fact_id = self._fact_id(
            subject,
            predicate,
        )
        doc = self.store.get(
            fact_id
        )

        if (
            not doc
            or not self._is_structured_fact(
                doc
            )
        ):
            return None

        meta = dict(
            doc.get("meta") or {}
        )

        return {
            "doc_id": fact_id,
            "text": doc.get(
                "text",
                "",
            ),
            **meta,
        }

    # ----------------------------------------------------------
    # Recuperación
    # ----------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        include_embeddings: bool = False,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        query = str(
            query or ""
        ).strip()

        if not query:
            return []

        docs = self.store.items()

        if active_only:
            filtered_docs = []

            for doc in docs:
                if not self._is_structured_fact(
                    doc
                ):
                    # Legacy sigue siendo consultable.
                    filtered_docs.append(
                        doc
                    )
                    continue

                meta = dict(
                    doc.get("meta")
                    or {}
                )

                if (
                    meta.get("status")
                    == "active"
                ):
                    filtered_docs.append(
                        doc
                    )

            docs = filtered_docs

        if not docs:
            return []

        texts = [
            str(
                doc.get("text")
                or ""
            )
            for doc in docs
        ]

        semantic_scores = [
            0.0
        ] * len(docs)

        semantic_vectors = None

        if semantic_embed is not None:
            try:
                vectors = semantic_embed(
                    [query] + texts
                )

                if (
                    vectors
                    and len(vectors)
                    == len(texts) + 1
                ):
                    qv = list(
                        vectors[0]
                    )
                    semantic_vectors = [
                        list(vector)
                        for vector
                        in vectors[1:]
                    ]
                    semantic_scores = [
                        _cosine(
                            qv,
                            vector,
                        )
                        for vector
                        in semantic_vectors
                    ]
            except Exception:
                semantic_vectors = None

        q_tokens = set(
            _tokens(query)
        )
        q_sparse = self._tfidf_norm(
            Counter(
                _tokens(query)
            )
        )

        hits: List[
            Dict[str, Any]
        ] = []

        for idx, doc in enumerate(
            docs
        ):
            doc_id = str(
                doc.get("id")
                or ""
            )
            text = str(
                doc.get("text")
                or ""
            )
            meta = dict(
                doc.get("meta")
                or {}
            )
            doc_tokens = set(
                _tokens(text)
            )
            overlap = len(
                q_tokens & doc_tokens
            )

            sparse = self._sparse_cos(
                q_sparse,
                self._doc_vec.get(
                    doc_id,
                    {},
                ),
            )
            semantic = (
                semantic_scores[idx]
            )

            structured = (
                self._is_structured_fact(
                    doc
                )
            )
            subject = str(
                meta.get("subject")
                or ""
            )
            confidence = float(
                meta.get("confidence")
                if structured
                else 0.50
            )

            subject_bonus = 0.0

            if subject:
                subject_norm = (
                    _norm_text(subject)
                )
                query_norm = (
                    _norm_text(query)
                )

                if (
                    subject_norm
                    and subject_norm
                    in query_norm
                ):
                    subject_bonus = 0.30

            lexical_bonus = (
                min(overlap, 6)
                / 6.0
            )

            if (
                semantic_vectors
                is not None
            ):
                score = (
                    semantic * 0.68
                    + lexical_bonus * 0.12
                    + subject_bonus
                    + confidence * 0.05
                )
            else:
                score = (
                    sparse * 0.70
                    + lexical_bonus * 0.20
                    + subject_bonus
                    + confidence * 0.05
                )

            if (
                score <= 0.05
                or (
                    overlap == 0
                    and subject_bonus == 0.0
                    and semantic < 0.18
                    and sparse <= 0.0
                )
            ):
                continue

            hit: Dict[
                str,
                Any,
            ] = {
                "doc_id": doc_id,
                "score": float(
                    round(
                        score,
                        6,
                    )
                ),
                "text": text,
                "kind": (
                    "semantic_fact"
                    if structured
                    else "legacy"
                ),
            }

            if structured:
                hit.update(
                    {
                        "subject":
                            subject,
                        "predicate":
                            meta.get(
                                "predicate"
                            ),
                        "value":
                            meta.get(
                                "value"
                            ),
                        "confidence":
                            confidence,
                        "support_count":
                            int(
                                meta.get(
                                    "support_count"
                                )
                                or 1
                            ),
                        "sources":
                            list(
                                meta.get(
                                    "sources"
                                )
                                or []
                            ),
                        "status":
                            meta.get(
                                "status"
                            ),
                        "history":
                            list(
                                meta.get(
                                    "history"
                                )
                                or []
                            ),
                    }
                )

            if (
                include_embeddings
                and semantic_vectors
                is not None
            ):
                hit["embedding"] = (
                    semantic_vectors[idx]
                )

            hits.append(hit)

        hits.sort(
            key=lambda item: (
                float(
                    item.get("score")
                    or 0.0
                ),
                float(
                    item.get(
                        "confidence"
                    )
                    or 0.0
                ),
                int(
                    item.get(
                        "support_count"
                    )
                    or 0
                ),
            ),
            reverse=True,
        )

        return hits[
            : max(
                1,
                int(top_k or 5),
            )
        ]

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        return self.search(
            query,
            top_k=k,
        )

    def query(
        self,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        return self.search(
            query,
            top_k=k,
        )

    # ----------------------------------------------------------
    # Compatibilidad
    # ----------------------------------------------------------

    def upsert(
        self,
        payload: Dict[str, Any],
    ) -> str:
        payload = dict(
            payload or {}
        )
        meta = dict(
            payload.get("meta")
            or {}
        )

        if all(
            meta.get(key)
            not in (None, "")
            for key in (
                "subject",
                "predicate",
                "value",
            )
        ):
            result = (
                self.consolidate_fact(
                    str(
                        meta[
                            "subject"
                        ]
                    ),
                    str(
                        meta[
                            "predicate"
                        ]
                    ),
                    str(
                        meta[
                            "value"
                        ]
                    ),
                    source_id=(
                        str(
                            meta.get(
                                "source_id"
                            )
                            or payload.get(
                                "id"
                            )
                            or ""
                        )
                        or None
                    ),
                    source_ts=int(
                        meta.get(
                            "source_ts"
                        )
                        or _now()
                    ),
                    source_kind=str(
                        meta.get(
                            "source_kind"
                        )
                        or "legacy_api"
                    ),
                )
            )

            return str(
                result.get("id")
                or ""
            )

        doc_id = self.store.upsert(
            payload
        )
        self._rebuild_lexical_index()
        return doc_id

    def write(
        self,
        text: str,
        meta: Optional[
            Dict[str, Any]
        ] = None,
    ) -> str:
        meta = dict(
            meta or {}
        )

        if all(
            meta.get(key)
            not in (None, "")
            for key in (
                "subject",
                "predicate",
                "value",
            )
        ):
            result = (
                self.consolidate_fact(
                    str(
                        meta[
                            "subject"
                        ]
                    ),
                    str(
                        meta[
                            "predicate"
                        ]
                    ),
                    str(
                        meta[
                            "value"
                        ]
                    ),
                    source_id=(
                        str(
                            meta.get(
                                "source_id"
                            )
                            or ""
                        )
                        or None
                    ),
                    source_kind=str(
                        meta.get(
                            "source_kind"
                        )
                        or "legacy_api"
                    ),
                )
            )

            return str(
                result.get("id")
                or ""
            )

        legacy_id = (
            "legacy:"
            + hashlib.sha1(
                (
                    str(text)
                    + json.dumps(
                        meta,
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                ).encode("utf-8")
            ).hexdigest()[:16]
        )

        self.store.upsert(
            {
                "id": legacy_id,
                "text": str(
                    text or ""
                ),
                "tags": [
                    "legacy_semantic"
                ],
                "meta": meta,
            }
        )
        self._rebuild_lexical_index()

        return legacy_id

    def count(self) -> int:
        return self.store.count

    def get_status(
        self,
    ) -> Dict[str, Any]:
        structured = 0
        legacy = 0
        subjects = set()

        for doc in self.store.items():
            if self._is_structured_fact(
                doc
            ):
                structured += 1
                meta = dict(
                    doc.get("meta")
                    or {}
                )

                if meta.get(
                    "subject"
                ):
                    subjects.add(
                        _norm_text(
                            str(
                                meta[
                                    "subject"
                                ]
                            )
                        )
                    )
            else:
                legacy += 1

        return {
            "ok": True,
            "version": self.VERSION,
            "store_path":
                self.store_path,
            "total_docs":
                self.store.count,
            "structured_facts":
                structured,
            "legacy_docs":
                legacy,
            "subjects":
                len(subjects),
        }

    @property
    def paths(
        self,
    ) -> Dict[str, str]:
        return {
            "store_path":
                self.store_path,
            "base_dir":
                self.base_dir,
        }
