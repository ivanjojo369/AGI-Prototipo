from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .embeddings_local import embed as semantic_embed
except Exception:
    semantic_embed = None


def _norm(text: str) -> str:
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


def _key(text: str) -> str:
    return re.sub(
        r"[^a-z0-9_]+",
        "_",
        _norm(text),
    ).strip("_")


def _tokens(text: str) -> List[str]:
    return [
        token
        for token in re.split(
            r"[^a-z0-9áéíóúñü_-]+",
            (text or "").lower(),
        )
        if token
    ]


def _cosine(
    a: List[float],
    b: List[float],
) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    return float(
        sum(x * y for x, y in zip(a, b))
    )


def _jaccard(
    a: str,
    b: str,
) -> float:
    left = set(_tokens(a))
    right = set(_tokens(b))

    if not left or not right:
        return 0.0

    return len(left & right) / float(
        len(left | right)
    )


class AssociativeMemory:
    """
    Memory 2.5.1 — memoria asociativa/contextual.

    Construye perfiles de conocimiento por sujeto a partir de SemanticMemory
    y asociaciones blandas entre sujetos.

    Importante:
    - una asociación NO equivale a una relación factual;
    - no crea nuevas aristas de verdad;
    - sirve para recuperar grupos de conocimientos conceptualmente próximos;
    - controla el presupuesto de contexto para reducir mezclas irrelevantes.

    Persistencia materializada:
        data/associative_store.json

    Fuente de verdad:
        SemanticMemory 2.3.x
        + RelationalMemory 2.4.x como señal auxiliar opcional.
    """

    VERSION = "2.5.1"
    STORE_SCHEMA = "associative_memory_v2.5.1"

    IDENTITY_PREDICATES = {
        "proyecto",
        "lugar",
        "estacion",
        "nombre",
        "sujeto",
    }

    # Un único valor compartido en estos campos NO basta para formar
    # una asociación conceptual fuerte. Ej.: lenguaje=Python no debe
    # unir automáticamente Calypso con Aster.
    WEAK_MATCH_PREDICATES = {
        "lenguaje",
        "estado",
        "ciudad",
        "dia",
        "fecha",
        "color",
        "numero",
        "codigo",
        "clave",
        "llave",
        "cantidad",
        "version",
    }

    ASSOCIATIVE_MARKERS = (
        "conocimientos asociados",
        "conocimiento asociado",
        "conocimientos relacionados",
        "temas asociados",
        "temas relacionados",
        "conceptos asociados",
        "conceptos relacionados",
        "ideas asociadas",
        "ideas relacionadas",
        "qué otros",
        "que otros",
        "qué más está asociado",
        "que mas esta asociado",
        "qué más se parece",
        "que mas se parece",
        "similares a",
        "similar a",
        "se parecen a",
        "comparten características",
        "comparten caracteristicas",
        "asociado con",
        "asociados con",
        "asociada con",
        "asociadas con",
    )

    def __init__(
        self,
        base_dir: str = "data",
        *,
        min_score: float = 0.48,
        max_related: int = 4,
    ) -> None:
        self.base_dir = str(base_dir)
        self.path = (
            Path(self.base_dir)
            / "associative_store.json"
        )
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.min_score = float(min_score)
        self.max_related = max(
            1,
            int(max_related),
        )

        self.profiles: Dict[
            str,
            Dict[str, Any],
        ] = {}
        self.associations: List[
            Dict[str, Any]
        ] = []

        self._load()

    # --------------------------------------------------------
    # Persistencia
    # --------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return

        try:
            payload = json.loads(
                self.path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return

        if not isinstance(payload, dict):
            return

        profiles = payload.get(
            "profiles"
        )
        associations = payload.get(
            "associations"
        )

        if isinstance(profiles, dict):
            self.profiles = profiles

        if isinstance(associations, list):
            self.associations = [
                item
                for item in associations
                if isinstance(item, dict)
            ]

    def _save(self) -> None:
        tmp = self.path.with_suffix(
            self.path.suffix + ".tmp"
        )

        tmp.write_text(
            json.dumps(
                {
                    "schema":
                        self.STORE_SCHEMA,
                    "version":
                        self.VERSION,
                    "updated_at":
                        int(time.time()),
                    "profiles":
                        self.profiles,
                    "associations":
                        self.associations,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        tmp.replace(
            self.path
        )

    # --------------------------------------------------------
    # Construcción de perfiles
    # --------------------------------------------------------

    @staticmethod
    def _profile_text(
        subject: str,
        fields: Dict[str, str],
    ) -> str:
        lines = [
            subject,
        ]

        for key, value in sorted(
            fields.items()
        ):
            lines.append(
                f"{key} {value}"
            )

        return " | ".join(lines)

    def _semantic_profiles(
        self,
        semantic_memory: Any,
    ) -> Dict[str, Dict[str, Any]]:
        if semantic_memory is None:
            return {}

        try:
            docs = list(
                semantic_memory.store.items()
            )
        except Exception:
            return {}

        schema = getattr(
            semantic_memory,
            "FACT_SCHEMA",
            "semantic_fact_v2.3",
        )

        grouped: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for doc in docs:
            meta = dict(
                doc.get("meta") or {}
            )

            if (
                meta.get("schema")
                != schema
                or meta.get("status")
                != "active"
            ):
                continue

            subject = str(
                meta.get("subject")
                or ""
            ).strip()
            predicate = _key(
                str(
                    meta.get(
                        "predicate"
                    )
                    or ""
                )
            )
            value = str(
                meta.get("value")
                or ""
            ).strip()

            if (
                not subject
                or not predicate
                or not value
            ):
                continue

            bucket = grouped.setdefault(
                _norm(subject),
                {
                    "subject": subject,
                    "fields": {},
                    "facts": [],
                },
            )

            bucket[
                "fields"
            ][predicate] = value

            bucket["facts"].append(
                {
                    "predicate":
                        predicate,
                    "value":
                        value,
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
                }
            )

        for normalized, profile in grouped.items():
            subject = str(
                profile["subject"]
            )
            fields = dict(
                profile["fields"]
            )

            profile["profile_text"] = (
                self._profile_text(
                    subject,
                    fields,
                )
            )
            profile["subject_norm"] = (
                normalized
            )

        return grouped

    def _direct_relational_pairs(
        self,
        relational_memory: Any,
    ) -> set:
        pairs = set()

        if relational_memory is None:
            return pairs

        try:
            edges = list(
                relational_memory.edges.values()
            )
        except Exception:
            return pairs

        for edge in edges:
            if (
                edge.get("status")
                != "active"
                or edge.get(
                    "object_kind"
                )
                != "entity"
            ):
                continue

            left = _norm(
                str(
                    edge.get(
                        "subject"
                    )
                    or ""
                )
            )
            right = _norm(
                str(
                    edge.get(
                        "object"
                    )
                    or ""
                )
            )

            if (
                left
                and right
                and left != right
            ):
                pairs.add(
                    tuple(
                        sorted(
                            (left, right)
                        )
                    )
                )

        return pairs

    def sync_from_semantic(
        self,
        semantic_memory: Any,
        relational_memory: Any = None,
    ) -> Dict[str, Any]:
        profiles = (
            self._semantic_profiles(
                semantic_memory
            )
        )

        if not profiles:
            self.profiles = {}
            self.associations = []
            self._save()

            return {
                "ok": True,
                "version":
                    self.VERSION,
                "profiles": 0,
                "associations": 0,
            }

        names = list(
            profiles.keys()
        )
        texts = [
            str(
                profiles[name][
                    "profile_text"
                ]
            )
            for name in names
        ]

        vectors = None

        if (
            semantic_embed is not None
            and len(texts) >= 2
        ):
            try:
                embedded = semantic_embed(
                    texts
                )
                if (
                    embedded
                    and len(embedded)
                    == len(texts)
                ):
                    vectors = [
                        list(vector)
                        for vector
                        in embedded
                    ]
            except Exception:
                vectors = None

        relation_pairs = (
            self._direct_relational_pairs(
                relational_memory
            )
        )

        associations: List[
            Dict[str, Any]
        ] = []

        for i, left_name in enumerate(
            names
        ):
            left = profiles[
                left_name
            ]
            left_fields = dict(
                left.get("fields")
                or {}
            )

            for j in range(
                i + 1,
                len(names),
            ):
                right_name = names[j]
                right = profiles[
                    right_name
                ]
                right_fields = dict(
                    right.get("fields")
                    or {}
                )

                shared_predicates = sorted(
                    set(
                        left_fields.keys()
                    )
                    & set(
                        right_fields.keys()
                    )
                    - self.IDENTITY_PREDICATES
                )

                shared_field_values = []

                for predicate in shared_predicates:
                    left_value = str(
                        left_fields.get(
                            predicate
                        )
                        or ""
                    ).strip()
                    right_value = str(
                        right_fields.get(
                            predicate
                        )
                        or ""
                    ).strip()

                    if (
                        left_value
                        and right_value
                        and _norm(left_value)
                        == _norm(right_value)
                    ):
                        shared_field_values.append(
                            {
                                "predicate":
                                    predicate,
                                "value":
                                    left_value,
                                "value_norm":
                                    _norm(
                                        left_value
                                    ),
                            }
                        )

                strong_shared = [
                    item
                    for item
                    in shared_field_values
                    if item[
                        "predicate"
                    ]
                    not in self.WEAK_MATCH_PREDICATES
                ]

                shared_values = sorted(
                    {
                        item[
                            "value_norm"
                        ]
                        for item
                        in shared_field_values
                        if item.get(
                            "value_norm"
                        )
                    }
                )

                lexical = _jaccard(
                    str(
                        left.get(
                            "profile_text"
                        )
                        or ""
                    ),
                    str(
                        right.get(
                            "profile_text"
                        )
                        or ""
                    ),
                )

                semantic = 0.0

                if vectors is not None:
                    semantic = _cosine(
                        vectors[i],
                        vectors[j],
                    )

                pair_key = tuple(
                    sorted(
                        (
                            left_name,
                            right_name,
                        )
                    )
                )
                direct_neighbor = (
                    pair_key
                    in relation_pairs
                )

                # Memory 2.5.1:
                # una asociación debe tener al menos una señal fuerte.
                # Compartir sólo "lenguaje=Python", "estado=activo", etc.
                # ya no es suficiente.
                qualifies = bool(
                    direct_neighbor
                    or strong_shared
                    or len(
                        shared_field_values
                    ) >= 2
                    or (
                        semantic >= 0.72
                        and lexical >= 0.18
                    )
                )

                if not qualifies:
                    continue

                score = (
                    semantic * 0.56
                    + lexical * 0.10
                    + min(
                        len(
                            strong_shared
                        ),
                        2,
                    ) * 0.22
                    + min(
                        len(
                            shared_field_values
                        ),
                        3,
                    ) * 0.08
                    + (
                        0.18
                        if direct_neighbor
                        else 0.0
                    )
                )

                # Un valor temático fuerte como dominio=oceanografía
                # puede sostener una asociación. Dos coincidencias fuertes
                # la vuelven especialmente sólida.
                if strong_shared:
                    score = max(
                        score,
                        0.62
                        + min(
                            len(
                                strong_shared
                            )
                            - 1,
                            1,
                        ) * 0.16,
                    )

                if len(
                    shared_field_values
                ) >= 2:
                    score = max(
                        score,
                        0.72,
                    )

                if direct_neighbor:
                    score = max(
                        score,
                        0.70,
                    )

                if (
                    semantic >= 0.72
                    and lexical >= 0.18
                ):
                    score = max(
                        score,
                        0.58,
                    )

                score = min(
                    1.0,
                    float(score),
                )

                if (
                    score
                    < self.min_score
                ):
                    continue

                reasons: List[str] = []

                if strong_shared:
                    reasons.append(
                        "strong_shared_fields"
                    )

                if len(
                    shared_field_values
                ) >= 2:
                    reasons.append(
                        "multiple_shared_fields"
                    )

                if (
                    semantic >= 0.72
                    and lexical >= 0.18
                ):
                    reasons.append(
                        "semantic_similarity"
                    )

                if direct_neighbor:
                    reasons.append(
                        "relational_neighbor"
                    )

                associations.append(
                    {
                        "left":
                            left["subject"],
                        "right":
                            right["subject"],
                        "left_norm":
                            left_name,
                        "right_norm":
                            right_name,
                        "score":
                            round(
                                score,
                                6,
                            ),
                        "reasons":
                            reasons,
                        "shared_values":
                            shared_values,
                        "shared_fields":
                            [
                                {
                                    "predicate":
                                        item[
                                            "predicate"
                                        ],
                                    "value":
                                        item[
                                            "value"
                                        ],
                                }
                                for item
                                in shared_field_values
                            ],
                        "shared_predicates":
                            shared_predicates,
                    }
                )

        associations.sort(
            key=lambda item:
                float(
                    item.get(
                        "score"
                    )
                    or 0.0
                ),
            reverse=True,
        )

        self.profiles = profiles
        self.associations = associations
        self._save()

        return {
            "ok": True,
            "version":
                self.VERSION,
            "profiles":
                len(
                    self.profiles
                ),
            "associations":
                len(
                    self.associations
                ),
        }

    # --------------------------------------------------------
    # Query routing
    # --------------------------------------------------------

    def subject_names(
        self,
    ) -> List[str]:
        return sorted(
            [
                str(
                    profile.get(
                        "subject"
                    )
                    or ""
                )
                for profile
                in self.profiles.values()
                if str(
                    profile.get(
                        "subject"
                    )
                    or ""
                ).strip()
            ],
            key=lambda value:
                (-len(value), value.lower()),
        )

    def find_subjects(
        self,
        query: str,
    ) -> List[str]:
        q = _norm(
            query
        )
        matches = []

        for subject in (
            self.subject_names()
        ):
            normalized = _norm(
                subject
            )
            start = q.find(
                normalized
            )

            if start >= 0:
                matches.append(
                    (
                        start,
                        -len(normalized),
                        subject,
                    )
                )

        matches.sort(
            key=lambda item:
                (item[0], item[1])
        )

        found: List[str] = []

        for _, _, subject in matches:
            if (
                subject
                not in found
            ):
                found.append(
                    subject
                )

        return found

    def is_associative_query(
        self,
        query: str,
    ) -> bool:
        text = _norm(
            query
        )

        if not self.find_subjects(
            query
        ):
            return False

        return any(
            _norm(marker)
            in text
            for marker
            in self.ASSOCIATIVE_MARKERS
        )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    def _profile_for_subject(
        self,
        subject: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        return self.profiles.get(
            _norm(subject)
        )

    def related_to(
        self,
        subject: str,
        *,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        target = _norm(
            subject
        )

        if not target:
            return []

        out = []

        for item in (
            self.associations
        ):
            left = str(
                item.get(
                    "left_norm"
                )
                or ""
            )
            right = str(
                item.get(
                    "right_norm"
                )
                or ""
            )

            if target == left:
                related = str(
                    item.get(
                        "right"
                    )
                    or ""
                )
            elif target == right:
                related = str(
                    item.get(
                        "left"
                    )
                    or ""
                )
            else:
                continue

            profile = (
                self._profile_for_subject(
                    related
                )
            )

            if profile is None:
                continue

            out.append(
                {
                    **item,
                    "related_subject":
                        related,
                    "related_profile":
                        profile,
                }
            )

        out.sort(
            key=lambda item:
                float(
                    item.get(
                        "score"
                    )
                    or 0.0
                ),
            reverse=True,
        )

        limit = (
            self.max_related
            if top_k is None
            else max(
                1,
                int(top_k),
            )
        )

        return out[:limit]

    @staticmethod
    def _fact_lines(
        profile: Dict[str, Any],
    ) -> List[str]:
        fields = dict(
            profile.get(
                "fields"
            )
            or {}
        )

        return [
            f"{key} = {value}"
            for key, value
            in sorted(
                fields.items()
            )
        ]

    def recall(
        self,
        query: str,
        *,
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        if not self.is_associative_query(
            query
        ):
            return []

        subjects = self.find_subjects(
            query
        )

        if not subjects:
            return []

        anchor = subjects[0]
        anchor_profile = (
            self._profile_for_subject(
                anchor
            )
        )

        if anchor_profile is None:
            return []

        related = self.related_to(
            anchor,
            top_k=max(
                1,
                min(
                    self.max_related,
                    int(top_k or 4),
                ),
            ),
        )

        if not related:
            return []

        selected_profiles = [
            anchor_profile,
        ]

        seen = {
            _norm(anchor)
        }

        association_public = []

        for item in related:
            related_subject = str(
                item.get(
                    "related_subject"
                )
                or ""
            )
            normalized = _norm(
                related_subject
            )

            if (
                not normalized
                or normalized
                in seen
            ):
                continue

            profile = item.get(
                "related_profile"
            )

            if not isinstance(
                profile,
                dict,
            ):
                continue

            seen.add(
                normalized
            )
            selected_profiles.append(
                profile
            )

            association_public.append(
                {
                    "subject":
                        related_subject,
                    "reasons":
                        list(
                            item.get(
                                "reasons"
                            )
                            or []
                        ),
                    "shared_values":
                        list(
                            item.get(
                                "shared_values"
                            )
                            or []
                        ),
                }
            )

        if len(selected_profiles) <= 1:
            return []

        blocks = []

        for profile in (
            selected_profiles
        ):
            subject = str(
                profile.get(
                    "subject"
                )
                or ""
            )

            lines = self._fact_lines(
                profile
            )

            if lines:
                blocks.append(
                    subject
                    + ":\n"
                    + "\n".join(lines)
                )

        if not blocks:
            return []

        return [
            {
                "role": "memory",
                "source": "association",
                "anchor_subject": anchor,
                "subjects": [
                    str(
                        profile.get(
                            "subject"
                        )
                        or ""
                    )
                    for profile
                    in selected_profiles
                ],
                "associations":
                    association_public,
                "profiles": [
                    {
                        "subject": str(
                            profile.get(
                                "subject"
                            )
                            or ""
                        ),
                        "fields": dict(
                            profile.get(
                                "fields"
                            )
                            or {}
                        ),
                    }
                    for profile
                    in selected_profiles
                ],
                "content":
                    "\n\n".join(
                        blocks
                    ),
            }
        ]

    def get_status(
        self,
    ) -> Dict[str, Any]:
        return {
            "ok": True,
            "version":
                self.VERSION,
            "path":
                str(self.path),
            "profiles":
                len(
                    self.profiles
                ),
            "associations":
                len(
                    self.associations
                ),
            "min_score":
                self.min_score,
        }
