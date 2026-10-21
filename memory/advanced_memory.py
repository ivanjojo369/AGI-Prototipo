from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> int:
    return int(time.time())


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


class AdvancedMemory:
    """
    Memory 2.9.1 — capa avanzada consolidada.

    Condensa seis capacidades que antes estaban planeadas como niveles separados:

    1) Relevancia / saliencia / olvido adaptativo.
       - Puntúa recuerdos por importancia, confianza, accesos y recencia.
       - NO borra conocimiento canónico automáticamente.
       - Sólo marca candidatos de archivo para evitar pérdida accidental.

    2) Conflictos / incertidumbre / procedencia.
       - Materializa el valor actual, valores históricos, confianza y fuentes.
       - No inventa una resolución: conserva la decisión canónica de SemanticMemory.

    3) Memoria procedimental.
       - Guarda procedimientos explícitos y secuencias de pasos reutilizables.

    4) Memoria prospectiva.
       - Guarda objetivos, condiciones, estado y siguiente paso.

    5) Abstracción jerárquica / esquemas.
       - Construye un esquema por sujeto con hechos, relaciones y asociaciones.

    6) Metamemoria.
       - Resume qué se conoce, cuánta evidencia existe, conflictos y cobertura.

    Persistencia:
        data/advanced_memory_store.json

    Memory 3.0 queda reservado para la orquestación final del sistema.
    """

    VERSION = "2.9.1"
    SCHEMA = "advanced_memory_v2.9.1"

    CONTROL_PREFIXES = (
        "responde",
        "respondeme",
        "respuesta",
        "contesta",
        "contestame",
        "reply",
        "answer",
    )

    PROCEDURAL_MARKERS = (
        "procedimiento",
        "procedimiento para",
        "pasos para",
        "pasos de",
        "rutina",
        "workflow",
        "flujo de trabajo",
        "cómo hacer",
        "como hacer",
        "cómo hago",
        "como hago",
    )

    PROSPECTIVE_MARKERS = (
        "objetivo",
        "objetivos",
        "pendiente",
        "pendientes",
        "siguiente paso",
        "siguientes pasos",
        "plan pendiente",
        "planes pendientes",
        "intención",
        "intenciones",
        "meta pendiente",
    )

    SCHEMA_MARKERS = (
        "esquema",
        "estructura de conocimiento",
        "resumen estructurado",
        "organiza lo que sabes",
        "organiza el conocimiento",
        "mapa de conocimiento",
    )

    CONFLICT_MARKERS = (
        "conflicto",
        "conflictos",
        "contradicción",
        "contradicciones",
        "contradiccion",
        "contradicciones",
        "incertidumbre",
        "procedencia",
        "fuentes",
        "fuente",
        "confianza",
        "historial del dato",
        "versiones del dato",
    )

    SALIENCE_MARKERS = (
        "recuerdos prioritarios",
        "memorias prioritarias",
        "más importante",
        "mas importante",
        "importancia de la memoria",
        "saliencia",
        "qué conservar",
        "que conservar",
        "qué archivar",
        "que archivar",
        "olvido adaptativo",
    )

    META_MARKERS = (
        "metamemoria",
        "estado de tu memoria",
        "estado de la memoria",
        # "qué recuerdas sobre" es deliberadamente ambiguo:
        # puede ser una consulta factual normal. La metamemoria debe
        # activarse por señales explícitas de cobertura/confianza.
        "qué tan seguro",
        "que tan seguro",
        "de dónde sabes",
        "de donde sabes",
        "qué sabes que sabes",
        "que sabes que sabes",
        "qué te falta saber",
        "que te falta saber",
        "cobertura de memoria",
    )

    def __init__(
        self,
        base_dir: str = "data",
        *,
        archive_threshold: float = 0.18,
    ) -> None:
        self.base_dir = str(base_dir)
        self.path = (
            Path(self.base_dir)
            / "advanced_memory_store.json"
        )
        self.legacy_path = (
            Path(self.base_dir)
            / "memory29_store.json"
        )
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.archive_threshold = float(
            archive_threshold
        )
        self.state = self._empty_state()
        self._load()

    def _empty_state(
        self,
    ) -> Dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "version": self.VERSION,
            "updated_at": 0,
            "salience": {},
            "conflicts": {},
            "procedures": {},
            "goals": {},
            "schemas": {},
            "meta": {},
        }

    def _load(
        self,
    ) -> None:
        # Memory 2.9.1 usa advanced_memory_store.json. Si sólo existe el
        # almacén anterior, lo migra de forma no destructiva.
        load_path = self.path

        if (
            not load_path.exists()
            and self.legacy_path.exists()
        ):
            load_path = self.legacy_path

        if not load_path.exists():
            return

        try:
            payload = json.loads(
                load_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return

        if not isinstance(
            payload,
            dict,
        ):
            return

        state = self._empty_state()

        for key in state:
            if key in payload:
                state[key] = payload[key]

        state["version"] = self.VERSION
        state["schema"] = self.SCHEMA
        self.state = state

        # Si cargamos el almacén legacy, materializamos inmediatamente
        # la copia canónica nueva. El archivo viejo se conserva.
        if load_path == self.legacy_path:
            self._save()

    def _save(
        self,
    ) -> None:
        self.state["schema"] = self.SCHEMA
        self.state["version"] = self.VERSION
        self.state["updated_at"] = _now()

        tmp = self.path.with_suffix(
            self.path.suffix + ".tmp"
        )

        tmp.write_text(
            json.dumps(
                self.state,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    # --------------------------------------------------------
    # Parsing
    # --------------------------------------------------------

    @classmethod
    def _is_control_field(
        cls,
        key: str,
    ) -> bool:
        normalized = _key(key)
        return any(
            normalized.startswith(prefix)
            for prefix in cls.CONTROL_PREFIXES
        )

    @classmethod
    def extract_fields(
        cls,
        text: str,
    ) -> Dict[str, str]:
        fields: Dict[str, str] = {}

        for raw_line in (
            text or ""
        ).splitlines():
            line = raw_line.strip()

            if not line:
                continue

            match = re.match(
                r"^([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9 _-]{1,80})\s*[:=]\s*(.+?)\s*$",
                line,
            )

            if not match:
                continue

            key = _key(
                match.group(1)
            )
            value = match.group(2).strip()

            if (
                not key
                or not value
                or cls._is_control_field(
                    key
                )
            ):
                continue

            fields[key] = value

        return fields

    @staticmethod
    def _numbered_steps(
        text: str,
    ) -> List[str]:
        steps: List[str] = []

        for raw_line in (
            text or ""
        ).splitlines():
            line = raw_line.strip()

            match = re.match(
                r"^(?:paso[_ ]?)?(\d{1,2})[\.\)\:= -]+\s*(.+?)\s*$",
                line,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            value = match.group(2).strip()

            if value:
                steps.append(value)

        return steps

    # --------------------------------------------------------
    # Salience
    # --------------------------------------------------------

    @staticmethod
    def _salience_score(
        entry: Dict[str, Any],
        now: Optional[int] = None,
    ) -> float:
        now = int(
            now or _now()
        )

        importance = float(
            entry.get(
                "importance"
            )
            or 0.5
        )
        confidence = float(
            entry.get(
                "confidence"
            )
            or 0.5
        )
        accesses = int(
            entry.get(
                "access_count"
            )
            or 0
        )
        last_access = int(
            entry.get(
                "last_access"
            )
            or entry.get(
                "created_at"
            )
            or now
        )

        age_days = max(
            0.0,
            (now - last_access)
            / 86400.0,
        )

        recency = math.exp(
            -age_days / 45.0
        )
        access_signal = min(
            1.0,
            math.log1p(
                max(
                    0,
                    accesses,
                )
            )
            / math.log(8.0),
        )

        score = (
            importance * 0.42
            + confidence * 0.28
            + access_signal * 0.18
            + recency * 0.12
        )

        return float(
            round(
                max(
                    0.0,
                    min(
                        1.0,
                        score,
                    ),
                ),
                6,
            )
        )

    def _ensure_salience(
        self,
        item_id: str,
        *,
        label: str,
        kind: str,
        importance: float,
        confidence: float,
    ) -> Dict[str, Any]:
        store = self.state[
            "salience"
        ]
        now = _now()

        entry = dict(
            store.get(
                item_id
            )
            or {}
        )

        entry.setdefault(
            "id",
            item_id,
        )
        entry["label"] = label
        entry["kind"] = kind
        entry["importance"] = max(
            float(
                entry.get(
                    "importance"
                )
                or 0.0
            ),
            float(importance),
        )
        entry["confidence"] = max(
            float(
                entry.get(
                    "confidence"
                )
                or 0.0
            ),
            float(confidence),
        )
        entry.setdefault(
            "access_count",
            0,
        )
        entry.setdefault(
            "created_at",
            now,
        )
        entry.setdefault(
            "last_access",
            now,
        )
        entry["score"] = (
            self._salience_score(
                entry,
                now,
            )
        )
        entry["archive_candidate"] = bool(
            entry["score"]
            < self.archive_threshold
            and int(
                entry.get(
                    "access_count"
                )
                or 0
            )
            == 0
        )

        store[
            item_id
        ] = entry

        return entry

    def note_recall(
        self,
        recalls: List[
            Dict[str, Any]
        ],
    ) -> None:
        if not recalls:
            return

        now = _now()
        changed = False

        for item in recalls:
            source = str(
                item.get(
                    "source"
                )
                or "memory"
            )

            labels: List[str] = []

            for key in (
                "semantic_subject",
                "episode_key",
                "anchor_subject",
            ):
                value = str(
                    item.get(key)
                    or ""
                ).strip()
                if value:
                    labels.append(value)

            for value in (
                item.get(
                    "subjects"
                )
                or []
            ):
                value = str(
                    value
                    or ""
                ).strip()
                if value:
                    labels.append(value)

            for label in labels:
                item_id = (
                    f"{source}:"
                    f"{_norm(label)}"
                )

                entry = self._ensure_salience(
                    item_id,
                    label=label,
                    kind=source,
                    importance=0.70,
                    confidence=0.70,
                )
                entry[
                    "access_count"
                ] = (
                    int(
                        entry.get(
                            "access_count"
                        )
                        or 0
                    )
                    + 1
                )
                entry[
                    "last_access"
                ] = now
                entry["score"] = (
                    self._salience_score(
                        entry,
                        now,
                    )
                )
                entry[
                    "archive_candidate"
                ] = False
                self.state[
                    "salience"
                ][item_id] = entry
                changed = True

        if changed:
            self._save()

    # --------------------------------------------------------
    # Sync canonical sources
    # --------------------------------------------------------

    def sync(
        self,
        semantic_memory: Any = None,
        relational_memory: Any = None,
        associative_memory: Any = None,
    ) -> Dict[str, Any]:
        facts_by_subject: Dict[
            str,
            Dict[str, Any],
        ] = {}

        conflicts: Dict[
            str,
            Dict[str, Any],
        ] = {}

        if semantic_memory is not None:
            try:
                docs = list(
                    semantic_memory.store.items()
                )
                fact_schema = getattr(
                    semantic_memory,
                    "FACT_SCHEMA",
                    "semantic_fact_v2.3",
                )
            except Exception:
                docs = []
                fact_schema = (
                    "semantic_fact_v2.3"
                )

            for doc in docs:
                meta = dict(
                    doc.get(
                        "meta"
                    )
                    or {}
                )

                if (
                    meta.get(
                        "schema"
                    )
                    != fact_schema
                    or meta.get(
                        "status"
                    )
                    != "active"
                ):
                    continue

                subject = str(
                    meta.get(
                        "subject"
                    )
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
                    meta.get(
                        "value"
                    )
                    or ""
                ).strip()

                if (
                    not subject
                    or not predicate
                    or not value
                ):
                    continue

                subject_key = _norm(
                    subject
                )

                bucket = (
                    facts_by_subject
                    .setdefault(
                        subject_key,
                        {
                            "subject":
                                subject,
                            "facts": {},
                        },
                    )
                )

                bucket[
                    "facts"
                ][predicate] = {
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
                    "sources":
                        list(
                            meta.get(
                                "sources"
                            )
                            or []
                        ),
                    "history":
                        list(
                            meta.get(
                                "history"
                            )
                            or []
                        ),
                }

                fact_id = (
                    f"fact:"
                    f"{subject_key}:"
                    f"{predicate}"
                )

                confidence = float(
                    meta.get(
                        "confidence"
                    )
                    or 0.0
                )

                support_count = int(
                    meta.get(
                        "support_count"
                    )
                    or 1
                )

                importance = min(
                    0.96,
                    0.52
                    + min(
                        support_count,
                        4,
                    ) * 0.08,
                )

                self._ensure_salience(
                    fact_id,
                    label=(
                        f"{subject}."
                        f"{predicate}"
                    ),
                    kind="semantic_fact",
                    importance=importance,
                    confidence=confidence,
                )

                historical = []

                for h in list(
                    meta.get(
                        "history"
                    )
                    or []
                ):
                    if not isinstance(
                        h,
                        dict,
                    ):
                        continue

                    old_value = str(
                        h.get(
                            "value"
                        )
                        or ""
                    ).strip()

                    if not old_value:
                        continue

                    historical.append(
                        {
                            "value":
                                old_value,
                            "status":
                                str(
                                    h.get(
                                        "status"
                                    )
                                    or "historical"
                                ),
                            "confidence":
                                float(
                                    h.get(
                                        "confidence"
                                    )
                                    or 0.0
                                ),
                            "sources":
                                list(
                                    h.get(
                                        "sources"
                                    )
                                    or []
                                ),
                            "valid_from":
                                h.get(
                                    "valid_from"
                                ),
                            "valid_to":
                                h.get(
                                    "valid_to"
                                ),
                        }
                    )

                if (
                    historical
                    or confidence < 0.60
                ):
                    conflict_id = (
                        f"{subject_key}:"
                        f"{predicate}"
                    )

                    conflicts[
                        conflict_id
                    ] = {
                        "subject":
                            subject,
                        "predicate":
                            predicate,
                        "current_value":
                            value,
                        "current_confidence":
                            confidence,
                        "current_sources":
                            list(
                                meta.get(
                                    "sources"
                                )
                                or []
                            ),
                        "alternatives":
                            historical,
                        "status": (
                            "current_with_history"
                            if historical
                            else "low_confidence"
                        ),
                    }

        relations_by_subject: Dict[
            str,
            List[
                Dict[str, Any]
            ],
        ] = {}

        if relational_memory is not None:
            try:
                edges = list(
                    relational_memory
                    .edges.values()
                )
            except Exception:
                edges = []

            for edge in edges:
                if (
                    edge.get(
                        "status"
                    )
                    != "active"
                    or edge.get(
                        "object_kind"
                    )
                    != "entity"
                ):
                    continue

                subject = str(
                    edge.get(
                        "subject"
                    )
                    or ""
                ).strip()

                if not subject:
                    continue

                relations_by_subject.setdefault(
                    _norm(subject),
                    [],
                ).append(
                    {
                        "predicate":
                            str(
                                edge.get(
                                    "predicate"
                                )
                                or ""
                            ),
                        "object":
                            str(
                                edge.get(
                                    "object"
                                )
                                or ""
                            ),
                        "object_kind":
                            str(
                                edge.get(
                                    "object_kind"
                                )
                                or ""
                            ),
                    }
                )

        associations_by_subject: Dict[
            str,
            List[str],
        ] = {}

        if associative_memory is not None:
            try:
                assoc_items = list(
                    associative_memory
                    .associations
                )
            except Exception:
                assoc_items = []

            for item in assoc_items:
                left = str(
                    item.get(
                        "left"
                    )
                    or ""
                ).strip()
                right = str(
                    item.get(
                        "right"
                    )
                    or ""
                ).strip()

                if left and right:
                    associations_by_subject.setdefault(
                        _norm(left),
                        [],
                    ).append(right)
                    associations_by_subject.setdefault(
                        _norm(right),
                        [],
                    ).append(left)

        schemas: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for subject_key, item in (
            facts_by_subject.items()
        ):
            subject = str(
                item.get(
                    "subject"
                )
                or ""
            )

            schemas[
                subject_key
            ] = {
                "subject":
                    subject,
                "facts":
                    dict(
                        item.get(
                            "facts"
                        )
                        or {}
                    ),
                "relations":
                    relations_by_subject
                    .get(
                        subject_key,
                        [],
                    ),
                "associations":
                    sorted(
                        set(
                            associations_by_subject
                            .get(
                                subject_key,
                                [],
                            )
                        )
                    ),
            }

        self.state[
            "conflicts"
        ] = conflicts
        self.state[
            "schemas"
        ] = schemas

        # Recalcular salience y candidatos de archivo sin borrar nada.
        now = _now()

        for item_id, entry in list(
            self.state[
                "salience"
            ].items()
        ):
            entry = dict(
                entry
                or {}
            )
            entry["score"] = (
                self._salience_score(
                    entry,
                    now,
                )
            )
            entry[
                "archive_candidate"
            ] = bool(
                entry[
                    "score"
                ]
                < self.archive_threshold
                and int(
                    entry.get(
                        "access_count"
                    )
                    or 0
                )
                == 0
            )
            self.state[
                "salience"
            ][item_id] = entry

        meta = {
            "subjects":
                len(schemas),
            "facts":
                sum(
                    len(
                        schema.get(
                            "facts"
                        )
                        or {}
                    )
                    for schema
                    in schemas.values()
                ),
            "conflict_records":
                len(conflicts),
            "procedures":
                len(
                    self.state[
                        "procedures"
                    ]
                ),
            "goals":
                len(
                    self.state[
                        "goals"
                    ]
                ),
            "pending_goals":
                sum(
                    1
                    for goal
                    in self.state[
                        "goals"
                    ].values()
                    if _norm(
                        str(
                            goal.get(
                                "status"
                            )
                            or "pendiente"
                        )
                    )
                    not in (
                        "completado",
                        "completa",
                        "completo",
                        "done",
                        "cancelado",
                        "cancelada",
                    )
                ),
            "archive_candidates":
                sum(
                    1
                    for entry
                    in self.state[
                        "salience"
                    ].values()
                    if entry.get(
                        "archive_candidate"
                    )
                ),
        }

        self.state[
            "meta"
        ] = meta
        self._save()

        return {
            "ok": True,
            "version":
                self.VERSION,
            **meta,
        }

    # --------------------------------------------------------
    # Explicit writes
    # --------------------------------------------------------

    def ingest_user(
        self,
        text: str,
        fields: Optional[
            Dict[str, str]
        ] = None,
    ) -> Dict[str, Any]:
        text = str(
            text or ""
        ).strip()
        fields = dict(
            fields
            or self.extract_fields(
                text
            )
        )

        actions: List[str] = []
        now = _now()

        # -------- procedural --------
        procedure_name = str(
            fields.get(
                "procedimiento"
            )
            or fields.get(
                "tarea"
            )
            or ""
        ).strip()

        steps = (
            self._numbered_steps(
                text
            )
        )

        if (
            procedure_name
            and steps
        ):
            proc_id = _norm(
                procedure_name
            )

            existing = dict(
                self.state[
                    "procedures"
                ].get(
                    proc_id
                )
                or {}
            )

            self.state[
                "procedures"
            ][proc_id] = {
                "name":
                    procedure_name,
                "steps":
                    steps,
                "status":
                    "active",
                "created_at":
                    int(
                        existing.get(
                            "created_at"
                        )
                        or now
                    ),
                "updated_at":
                    now,
                "success_count":
                    int(
                        existing.get(
                            "success_count"
                        )
                        or 0
                    ),
                "failure_count":
                    int(
                        existing.get(
                            "failure_count"
                        )
                        or 0
                    ),
            }

            self._ensure_salience(
                f"procedure:{proc_id}",
                label=procedure_name,
                kind="procedure",
                importance=0.88,
                confidence=0.85,
            )
            actions.append(
                "procedure_saved"
            )

        # -------- prospective --------
        objective = str(
            fields.get(
                "objetivo"
            )
            or fields.get(
                "meta"
            )
            or ""
        ).strip()

        if objective:
            goal_id = _norm(
                objective
            )

            existing = dict(
                self.state[
                    "goals"
                ].get(
                    goal_id
                )
                or {}
            )

            status = str(
                fields.get(
                    "estado"
                )
                or existing.get(
                    "status"
                )
                or "pendiente"
            ).strip()

            self.state[
                "goals"
            ][goal_id] = {
                "objective":
                    objective,
                "status":
                    status,
                "condition":
                    str(
                        fields.get(
                            "condicion"
                        )
                        or fields.get(
                            "condición"
                        )
                        or existing.get(
                            "condition"
                        )
                        or ""
                    ).strip(),
                "next_step":
                    str(
                        fields.get(
                            "siguiente_paso"
                        )
                        or fields.get(
                            "proximo_paso"
                        )
                        or existing.get(
                            "next_step"
                        )
                        or ""
                    ).strip(),
                "due":
                    str(
                        fields.get(
                            "fecha"
                        )
                        or fields.get(
                            "fecha_limite"
                        )
                        or existing.get(
                            "due"
                        )
                        or ""
                    ).strip(),
                "created_at":
                    int(
                        existing.get(
                            "created_at"
                        )
                        or now
                    ),
                "updated_at":
                    now,
            }

            self._ensure_salience(
                f"goal:{goal_id}",
                label=objective,
                kind="goal",
                importance=0.92,
                confidence=0.90,
            )
            actions.append(
                "goal_saved"
            )

        if actions:
            self._save()

        return {
            "ok": True,
            "actions":
                actions,
        }

    # --------------------------------------------------------
    # Query helpers
    # --------------------------------------------------------

    def _find_subject(
        self,
        query: str,
    ) -> Optional[str]:
        q = _norm(
            query
        )

        matches = []

        for key, schema in (
            self.state[
                "schemas"
            ].items()
        ):
            subject = str(
                schema.get(
                    "subject"
                )
                or ""
            ).strip()

            if not subject:
                continue

            pos = q.find(
                _norm(subject)
            )

            if pos >= 0:
                matches.append(
                    (
                        pos,
                        -len(subject),
                        key,
                    )
                )

        if not matches:
            return None

        matches.sort()
        return matches[0][2]

    @staticmethod
    def _contains_any(
        query: str,
        markers: tuple,
    ) -> bool:
        text = _norm(
            query
        )
        return any(
            _norm(marker)
            in text
            for marker
            in markers
        )

    def is_memory29_query(
        self,
        query: str,
    ) -> bool:
        groups = (
            self.PROCEDURAL_MARKERS,
            self.PROSPECTIVE_MARKERS,
            self.SCHEMA_MARKERS,
            self.CONFLICT_MARKERS,
            self.SALIENCE_MARKERS,
            self.META_MARKERS,
        )

        return any(
            self._contains_any(
                query,
                group,
            )
            for group in groups
        )

    def _touch_salience(
        self,
        item_id: str,
    ) -> None:
        entry = dict(
            self.state[
                "salience"
            ].get(
                item_id
            )
            or {}
        )

        if not entry:
            return

        now = _now()
        entry[
            "access_count"
        ] = (
            int(
                entry.get(
                    "access_count"
                )
                or 0
            )
            + 1
        )
        entry[
            "last_access"
        ] = now
        entry["score"] = (
            self._salience_score(
                entry,
                now,
            )
        )
        entry[
            "archive_candidate"
        ] = False

        self.state[
            "salience"
        ][item_id] = entry
        self._save()

    # --------------------------------------------------------
    # Rendering
    # --------------------------------------------------------

    def _recall_procedural(
        self,
        query: str,
    ) -> List[str]:
        procedures = self.state[
            "procedures"
        ]

        if not procedures:
            return [
                "No hay procedimientos guardados."
            ]

        q = _norm(
            query
        )
        candidates = []

        for proc_id, proc in (
            procedures.items()
        ):
            name = str(
                proc.get(
                    "name"
                )
                or proc_id
            )
            score = 0

            if _norm(name) in q:
                score += 100

            overlap = len(
                set(
                    _tokens(name)
                )
                & set(
                    _tokens(query)
                )
            )
            score += overlap

            candidates.append(
                (
                    score,
                    proc_id,
                    proc,
                )
            )

        candidates.sort(
            key=lambda item:
                item[0],
            reverse=True,
        )

        selected = (
            candidates[:1]
            if candidates
            and candidates[0][0] > 0
            else candidates[:3]
        )

        lines: List[str] = []

        for _, proc_id, proc in selected:
            name = str(
                proc.get(
                    "name"
                )
                or "procedimiento"
            )
            self._touch_salience(
                f"procedure:{proc_id}"
            )

            lines.append(
                f"{name}:"
            )

            for idx, step in enumerate(
                proc.get(
                    "steps"
                )
                or [],
                1,
            ):
                lines.append(
                    f"{idx}. {step}"
                )

        return lines

    def _recall_goals(
        self,
        query: str,
    ) -> List[str]:
        goals = self.state[
            "goals"
        ]

        if not goals:
            return [
                "No hay objetivos guardados."
            ]

        q = _norm(
            query
        )
        wants_pending = (
            "pendiente"
            in q
        )
        selected = []

        for goal_id, goal in goals.items():
            status = _norm(
                str(
                    goal.get(
                        "status"
                    )
                    or "pendiente"
                )
            )

            completed = (
                status
                in (
                    "completado",
                    "completa",
                    "completo",
                    "done",
                    "cancelado",
                    "cancelada",
                )
            )

            if (
                wants_pending
                and completed
            ):
                continue

            selected.append(
                (
                    goal_id,
                    goal,
                )
            )

        selected.sort(
            key=lambda item:
                int(
                    item[1].get(
                        "updated_at"
                    )
                    or 0
                ),
            reverse=True,
        )

        # Important: a valid advanced-memory query with zero pending goals
        # still returns authoritative evidence, preventing fallback to an
        # older episodic "estado = pendiente".
        if (
            wants_pending
            and not selected
        ):
            return [
                "No hay objetivos pendientes."
            ]

        lines: List[str] = []

        for goal_id, goal in selected[:5]:
            self._touch_salience(
                f"goal:{goal_id}"
            )

            lines.append(
                "objetivo = "
                + str(
                    goal.get(
                        "objective"
                    )
                    or ""
                )
            )
            lines.append(
                "estado = "
                + str(
                    goal.get(
                        "status"
                    )
                    or "pendiente"
                )
            )

            if goal.get(
                "condition"
            ):
                lines.append(
                    "condición = "
                    + str(
                        goal[
                            "condition"
                        ]
                    )
                )

            if goal.get(
                "next_step"
            ):
                lines.append(
                    "siguiente paso = "
                    + str(
                        goal[
                            "next_step"
                        ]
                    )
                )

            if goal.get(
                "due"
            ):
                lines.append(
                    "fecha = "
                    + str(
                        goal[
                            "due"
                        ]
                    )
                )

            lines.append("")

        while (
            lines
            and not lines[-1]
        ):
            lines.pop()

        return lines

    def _recall_schema(
        self,
        query: str,
    ) -> List[str]:
        subject_key = (
            self._find_subject(
                query
            )
        )

        if not subject_key:
            return [
                "No se identificó un sujeto conocido para construir el esquema."
            ]

        schema = self.state[
            "schemas"
        ].get(
            subject_key,
            {}
        )

        subject = str(
            schema.get(
                "subject"
            )
            or subject_key
        )

        lines = [
            f"{subject}:",
            "HECHOS:",
        ]

        facts = dict(
            schema.get(
                "facts"
            )
            or {}
        )

        for predicate, fact in sorted(
            facts.items()
        ):
            lines.append(
                f"- {predicate} = "
                f"{fact.get('value')}"
            )

        relations = list(
            schema.get(
                "relations"
            )
            or []
        )

        if relations:
            lines.append(
                "RELACIONES:"
            )

            for rel in relations:
                lines.append(
                    f"- {subject} "
                    f"-[{rel.get('predicate')}]-> "
                    f"{rel.get('object')}"
                )

        associations = list(
            schema.get(
                "associations"
            )
            or []
        )

        if associations:
            lines.append(
                "ASOCIACIONES:"
            )

            for value in associations:
                lines.append(
                    f"- {value}"
                )

        return lines

    def _recall_conflicts(
        self,
        query: str,
    ) -> List[str]:
        subject_key = (
            self._find_subject(
                query
            )
        )
        q = _norm(
            query
        )

        candidates = []

        for conflict in (
            self.state[
                "conflicts"
            ].values()
        ):
            if (
                subject_key
                and _norm(
                    str(
                        conflict.get(
                            "subject"
                        )
                        or ""
                    )
                )
                != subject_key
            ):
                continue

            candidates.append(
                conflict
            )

        # If the user names a specific predicate (e.g. "lenguaje"),
        # restrict the output to that predicate.
        requested_predicates = {
            str(
                item.get(
                    "predicate"
                )
                or ""
            )
            for item in candidates
            if str(
                item.get(
                    "predicate"
                )
                or ""
            ).strip()
            and (
                _norm(
                    str(
                        item.get(
                            "predicate"
                        )
                    ).replace(
                        "_",
                        " ",
                    )
                )
                in q
            )
        }

        if requested_predicates:
            items = [
                item
                for item in candidates
                if str(
                    item.get(
                        "predicate"
                    )
                    or ""
                )
                in requested_predicates
            ]
        else:
            items = candidates

        if not items:
            return [
                "No hay conflictos, versiones históricas o hechos de baja confianza para esa consulta."
            ]

        lines: List[str] = []

        for item in items[:8]:
            lines.append(
                f"{item.get('subject')}.{item.get('predicate')}:"
            )
            lines.append(
                "valor actual = "
                + str(
                    item.get(
                        "current_value"
                    )
                )
            )
            lines.append(
                "confianza actual = "
                + str(
                    round(
                        float(
                            item.get(
                                "current_confidence"
                            )
                            or 0.0
                        ),
                        3,
                    )
                )
            )

            sources = list(
                item.get(
                    "current_sources"
                )
                or []
            )

            if sources:
                lines.append(
                    "fuentes actuales = "
                    + ", ".join(
                        str(source)
                        for source
                        in sources[:6]
                    )
                )

            alternatives = list(
                item.get(
                    "alternatives"
                )
                or []
            )

            for alt in alternatives[:5]:
                lines.append(
                    "valor anterior = "
                    + str(
                        alt.get(
                            "value"
                        )
                    )
                    + " ["
                    + str(
                        alt.get(
                            "status"
                        )
                    )
                    + "]"
                )

            lines.append("")

        while (
            lines
            and not lines[-1]
        ):
            lines.pop()

        return lines

    def _recall_salience(
        self,
        query: str,
    ) -> List[str]:
        entries = [
            dict(entry)
            for entry
            in self.state[
                "salience"
            ].values()
        ]

        entries.sort(
            key=lambda entry:
                float(
                    entry.get(
                        "score"
                    )
                    or 0.0
                ),
            reverse=True,
        )

        wants_archive = any(
            marker in _norm(
                query
            )
            for marker
            in (
                "archivar",
                "olvid",
            )
        )

        if wants_archive:
            entries = [
                entry
                for entry in entries
                if entry.get(
                    "archive_candidate"
                )
            ]

        if not entries:
            return [
                "No hay candidatos de archivo."
                if wants_archive
                else "No hay registros de saliencia."
            ]

        lines = []

        for entry in entries[:10]:
            lines.append(
                f"{entry.get('label')} "
                f"| score={entry.get('score')} "
                f"| accesos={entry.get('access_count', 0)} "
                f"| archivar={'sí' if entry.get('archive_candidate') else 'no'}"
            )

        return lines

    def _recall_meta(
        self,
        query: str,
    ) -> List[str]:
        subject_key = (
            self._find_subject(
                query
            )
        )

        if subject_key:
            schema = self.state[
                "schemas"
            ].get(
                subject_key,
                {}
            )
            subject = str(
                schema.get(
                    "subject"
                )
                or subject_key
            )

            facts = dict(
                schema.get(
                    "facts"
                )
                or {}
            )

            confidences = [
                float(
                    fact.get(
                        "confidence"
                    )
                    or 0.0
                )
                for fact
                in facts.values()
            ]

            avg_conf = (
                sum(confidences)
                / len(confidences)
                if confidences
                else 0.0
            )

            conflicts = [
                item
                for item in (
                    self.state[
                        "conflicts"
                    ].values()
                )
                if _norm(
                    str(
                        item.get(
                            "subject"
                        )
                        or ""
                    )
                )
                == subject_key
            ]

            sources = set()

            for fact in facts.values():
                for source in (
                    fact.get(
                        "sources"
                    )
                    or []
                ):
                    sources.add(
                        str(source)
                    )

            lines = [
                f"sujeto = {subject}",
                f"hechos activos = {len(facts)}",
                f"confianza media = {round(avg_conf, 3)}",
                f"registros con historial/incertidumbre = {len(conflicts)}",
                f"fuentes conocidas = {len(sources)}",
                f"relaciones activas = {len(schema.get('relations') or [])}",
                f"asociaciones conceptuales = {len(schema.get('associations') or [])}",
            ]

            return lines

        meta = dict(
            self.state.get(
                "meta"
            )
            or {}
        )

        return [
            f"versión = {self.VERSION}",
            f"sujetos = {meta.get('subjects', 0)}",
            f"hechos = {meta.get('facts', 0)}",
            f"conflictos/incertidumbres = {meta.get('conflict_records', 0)}",
            f"procedimientos = {meta.get('procedures', 0)}",
            f"objetivos = {meta.get('goals', 0)}",
            f"objetivos pendientes = {meta.get('pending_goals', 0)}",
            f"candidatos de archivo = {meta.get('archive_candidates', 0)}",
        ]

    @classmethod
    def _is_global_meta_query(
        cls,
        query: str,
    ) -> bool:
        markers = (
            "metamemoria",
            "estado de tu memoria",
            "estado de la memoria",
            "cobertura de memoria",
            "qué sabes que sabes",
            "que sabes que sabes",
            "qué te falta saber",
            "que te falta saber",
        )
        return cls._contains_any(
            query,
            markers,
        )

    def recall(
        self,
        query: str,
        *,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not self.is_memory29_query(
            query
        ):
            return []

        kind = ""
        lines: List[str] = []

        # Global metamemory must beat incidental category words in a report
        # request such as "incluye procedimientos, objetivos...".
        if self._is_global_meta_query(
            query
        ):
            kind = "metamemory"
            lines = (
                self._recall_meta(
                    query
                )
            )
        elif self._contains_any(
            query,
            self.PROCEDURAL_MARKERS,
        ):
            kind = "procedural"
            lines = (
                self._recall_procedural(
                    query
                )
            )
        elif self._contains_any(
            query,
            self.PROSPECTIVE_MARKERS,
        ):
            kind = "prospective"
            lines = (
                self._recall_goals(
                    query
                )
            )
        elif self._contains_any(
            query,
            self.SCHEMA_MARKERS,
        ):
            kind = "schema"
            lines = (
                self._recall_schema(
                    query
                )
            )
        elif self._contains_any(
            query,
            self.CONFLICT_MARKERS,
        ):
            kind = "conflict"
            lines = (
                self._recall_conflicts(
                    query
                )
            )
        elif self._contains_any(
            query,
            self.SALIENCE_MARKERS,
        ):
            kind = "salience"
            lines = (
                self._recall_salience(
                    query
                )
            )
        elif self._contains_any(
            query,
            self.META_MARKERS,
        ):
            kind = "metamemory"
            lines = (
                self._recall_meta(
                    query
                )
            )

        if not lines:
            return []

        return [
            {
                "role":
                    "memory",
                "source":
                    "advanced_memory",
                "advanced_kind":
                    kind,
                "display_lines":
                    lines,
                "content":
                    "\n".join(
                        lines
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
                str(
                    self.path
                ),
            **dict(
                self.state.get(
                    "meta"
                )
                or {}
            ),
        }
