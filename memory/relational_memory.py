from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _norm(text: str) -> str:
    value = (text or "").strip().lower()
    value = (
        value.replace("á", "a").replace("é", "e")
        .replace("í", "i").replace("ó", "o")
        .replace("ú", "u").replace("ü", "u")
        .replace("ñ", "n")
    )
    return re.sub(r"\s+", " ", value)


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", _norm(text)).strip("_")


class RelationalMemory:
    """
    Memory 2.4.1 — memoria relacional y temporal.

    Materializa los hechos activos de SemanticMemory como:
        sujeto -[predicado]-> objeto

    No infiere relaciones nuevas. Sólo:
    - recupera aristas existentes;
    - encuentra caminos entre entidades;
    - expande vecindarios;
    - expone relaciones históricas preservadas por SemanticMemory.

    Fuente de verdad: SemanticMemory.
    Índice materializado: data/relational_store.json
    """

    VERSION = "2.4.1"
    SCHEMA = "relational_edge_v2.4"

    IDENTITY = {"proyecto", "lugar", "estacion", "nombre", "sujeto"}

    REL_MARKERS = (
        "relacion", "relación", "relacionado", "relacionada",
        "conecta", "conectado", "conexion", "conexión",
        "vinculo", "vínculo", "vincula", "cadena",
        "camino", "ruta", "enlace", "entre", "depende",
    )

    HISTORY_MARKERS = (
        "antes", "anterior", "anteriormente", "previo", "previa",
        "historial", "historico", "histórico", "pasado",
        "reemplazado", "sustituido",
    )

    def __init__(self, base_dir: str = "data") -> None:
        self.base_dir = str(base_dir)
        self.path = Path(self.base_dir) / "relational_store.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.edges: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            raw = payload.get("edges", {}) if isinstance(payload, dict) else {}
            self.edges = dict(raw) if isinstance(raw, dict) else {}
        except Exception:
            self.edges = {}

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "version": self.VERSION,
                    "updated_at": int(time.time()),
                    "edges": self.edges,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    @staticmethod
    def _edge_id(subject: str, predicate: str) -> str:
        raw = f"{_norm(subject)}|{_key(predicate)}".encode("utf-8")
        return "edge:" + hashlib.sha256(raw).hexdigest()[:20]

    def sync_from_semantic(self, semantic_memory: Any) -> Dict[str, Any]:
        if semantic_memory is None:
            return {"ok": False, "reason": "semantic_memory_unavailable"}

        try:
            docs = list(semantic_memory.store.items())
        except Exception:
            return {"ok": False, "reason": "semantic_store_unavailable"}

        schema = getattr(semantic_memory, "FACT_SCHEMA", "semantic_fact_v2.3")
        active_docs: List[Dict[str, Any]] = []
        subjects: Dict[str, str] = {}

        for doc in docs:
            meta = dict(doc.get("meta") or {})
            if meta.get("schema") != schema or meta.get("status") != "active":
                continue
            subject = str(meta.get("subject") or "").strip()
            if not subject:
                continue
            subjects[_norm(subject)] = subject
            active_docs.append(meta)

        new_edges: Dict[str, Dict[str, Any]] = {}

        for meta in active_docs:
            subject = str(meta.get("subject") or "").strip()
            predicate = _key(str(meta.get("predicate") or ""))
            obj = str(meta.get("value") or "").strip()

            if not subject or not predicate or not obj:
                continue

            if predicate in self.IDENTITY and _norm(subject) == _norm(obj):
                continue

            history = []
            for h in list(meta.get("history") or []):
                if not isinstance(h, dict):
                    continue
                old_value = str(h.get("value") or "").strip()
                if not old_value:
                    continue
                history.append(
                    {
                        "subject": subject,
                        "predicate": predicate,
                        "object": old_value,
                        "status": str(h.get("status") or "historical"),
                        "confidence": float(h.get("confidence") or 0.0),
                        "support_count": int(h.get("support_count") or 1),
                        "sources": list(h.get("sources") or []),
                        "valid_from": h.get("valid_from"),
                        "valid_to": h.get("valid_to"),
                        "observed_at": h.get("observed_at"),
                    }
                )

            edge_id = self._edge_id(subject, predicate)
            new_edges[edge_id] = {
                "schema": self.SCHEMA,
                "version": self.VERSION,
                "id": edge_id,
                "subject": subject,
                "subject_norm": _norm(subject),
                "predicate": predicate,
                "object": obj,
                "object_norm": _norm(obj),
                "object_kind": "entity" if _norm(obj) in subjects else "literal",
                "confidence": float(meta.get("confidence") or 0.0),
                "support_count": int(meta.get("support_count") or 1),
                "sources": list(meta.get("sources") or []),
                "updated_at": int(meta.get("updated_at") or 0),
                "status": "active",
                "history": history,
            }

        self.edges = new_edges
        self._save()

        return {
            "ok": True,
            "version": self.VERSION,
            "edges": len(self.edges),
            "entities": len(self.entity_names()),
            "historical_relations": sum(
                len(e.get("history") or []) for e in self.edges.values()
            ),
        }

    def entity_names(self) -> List[str]:
        names: Dict[str, str] = {}
        for edge in self.edges.values():
            subject = str(edge.get("subject") or "").strip()
            if subject:
                names[_norm(subject)] = subject
            if edge.get("object_kind") == "entity":
                obj = str(edge.get("object") or "").strip()
                if obj:
                    names[_norm(obj)] = obj
        return sorted(names.values(), key=lambda x: (-len(x), x.lower()))

    def find_entities(self, query: str) -> List[str]:
        """
        Devuelve entidades en el mismo orden en que aparecen en la consulta.

        Memory 2.4 ordenaba entity_names() por longitud del nombre; eso podía
        invertir una pregunta como "Aster e Instituto Boreal" y presentar el
        camino relacional al revés. Aquí usamos posición textual y preferimos
        el nombre más largo cuando dos entidades se solapan.
        """
        q = _norm(query)
        matches = []

        for name in self.entity_names():
            normalized_name = _norm(name)

            if not normalized_name:
                continue

            start = q.find(normalized_name)

            if start < 0:
                continue

            end = start + len(normalized_name)
            matches.append(
                (
                    start,
                    -len(normalized_name),
                    end,
                    name,
                )
            )

        matches.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        found: List[str] = []
        occupied = []

        for start, _, end, name in matches:
            # Si "Instituto" e "Instituto Boreal" empiezan en la misma
            # posición, conserva sólo la coincidencia más específica.
            if any(
                start < used_end
                and end > used_start
                for used_start, used_end
                in occupied
            ):
                continue

            found.append(name)
            occupied.append(
                (start, end)
            )

        return found

    def is_relational_query(self, query: str) -> bool:
        text = _norm(query)
        entities = self.find_entities(query)

        if len(entities) >= 2:
            return True
        if not entities:
            return False

        if any(_norm(m) in text for m in self.REL_MARKERS):
            return True
        if any(_norm(m) in text for m in self.HISTORY_MARKERS):
            return True
        return False

    def _outgoing(self, entity: str) -> List[Dict[str, Any]]:
        target = _norm(entity)
        return [
            dict(e)
            for e in self.edges.values()
            if e.get("status") == "active"
            and _norm(str(e.get("subject") or "")) == target
        ]

    def _incoming(self, entity: str) -> List[Dict[str, Any]]:
        target = _norm(entity)
        return [
            dict(e)
            for e in self.edges.values()
            if e.get("status") == "active"
            and e.get("object_kind") == "entity"
            and _norm(str(e.get("object") or "")) == target
        ]

    @staticmethod
    def _line(edge: Dict[str, Any], historical: bool = False) -> str:
        suffix = " (anterior)" if historical else ""
        return (
            f"{edge.get('subject')} "
            f"-[{edge.get('predicate')}]-> "
            f"{edge.get('object')}{suffix}"
        )

    def path_between(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
    ) -> List[Dict[str, Any]]:
        s = _norm(source)
        t = _norm(target)

        if not s or not t or s == t:
            return []

        adjacency: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}

        for edge in self.edges.values():
            if edge.get("status") != "active" or edge.get("object_kind") != "entity":
                continue

            a = _norm(str(edge.get("subject") or ""))
            b = _norm(str(edge.get("object") or ""))
            if not a or not b:
                continue

            adjacency.setdefault(a, []).append((b, dict(edge)))
            adjacency.setdefault(b, []).append((a, dict(edge)))

        queue = deque([(s, [], {s})])

        while queue:
            node, path, seen = queue.popleft()
            if len(path) >= max_hops:
                continue

            for neighbor, edge in adjacency.get(node, []):
                if neighbor in seen:
                    continue

                new_path = path + [edge]
                if neighbor == t:
                    return new_path

                queue.append((neighbor, new_path, seen | {neighbor}))

        return []

    def neighborhood(
        self,
        entity: str,
        depth: int = 2,
        max_edges: int = 12,
    ) -> List[Dict[str, Any]]:
        queue = deque([(entity, 0)])
        visited = {_norm(entity)}
        selected: List[Dict[str, Any]] = []
        seen_edges = set()

        while queue:
            current, level = queue.popleft()

            for edge in self._outgoing(current) + self._incoming(current):
                edge_id = str(edge.get("id") or "")
                if edge_id in seen_edges:
                    continue

                seen_edges.add(edge_id)
                selected.append(edge)

                if len(selected) >= max_edges:
                    return selected

                if level + 1 >= depth:
                    continue

                candidates = [str(edge.get("subject") or "")]
                if edge.get("object_kind") == "entity":
                    candidates.append(str(edge.get("object") or ""))

                for candidate in candidates:
                    n = _norm(candidate)
                    if n and n not in visited:
                        visited.add(n)
                        queue.append((candidate, level + 1))

        return selected

    def historical_relations(
        self,
        entity: str,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        target = _norm(entity)
        q = _norm(query or "")
        out: List[Dict[str, Any]] = []

        # Si el usuario menciona un predicado concreto, filtrar por él.
        matched_predicates = {
            str(edge.get("predicate") or "")
            for edge in self.edges.values()
            if _norm(str(edge.get("subject") or "")) == target
            and _norm(str(edge.get("predicate") or "")) in q
        }

        for edge in self.edges.values():
            if _norm(str(edge.get("subject") or "")) != target:
                continue
            predicate = str(edge.get("predicate") or "")
            if matched_predicates and predicate not in matched_predicates:
                continue

            for item in list(edge.get("history") or []):
                if isinstance(item, dict):
                    out.append(dict(item))

        return out

    def recall(
        self,
        query: str,
        max_hops: int = 3,
        max_edges: int = 12,
    ) -> List[Dict[str, Any]]:
        if not self.is_relational_query(query):
            return []

        entities = self.find_entities(query)
        text = _norm(query)
        wants_history = any(_norm(m) in text for m in self.HISTORY_MARKERS)

        if wants_history and entities:
            history = self.historical_relations(entities[0], query=query)
            if history:
                lines = [self._line(e, historical=True) for e in history[:max_edges]]
                return [{
                    "role": "memory",
                    "source": "relation",
                    "relation_type": "history",
                    "entities": entities[:1],
                    "edges": history[:max_edges],
                    "content": "\n".join(lines),
                    "display_lines": lines,
                }]

        if len(entities) >= 2:
            path = self.path_between(entities[0], entities[1], max_hops=max_hops)
            if path:
                lines = [self._line(e) for e in path]
                return [{
                    "role": "memory",
                    "source": "relation",
                    "relation_type": "path",
                    "entities": entities[:2],
                    "edges": path,
                    "content": "\n".join(lines),
                    "display_lines": lines,
                }]

        if entities:
            edges = self.neighborhood(
                entities[0],
                depth=2,
                max_edges=max_edges,
            )
            if edges:
                lines = [self._line(e) for e in edges]
                return [{
                    "role": "memory",
                    "source": "relation",
                    "relation_type": "neighborhood",
                    "entities": entities[:1],
                    "edges": edges,
                    "content": "\n".join(lines),
                    "display_lines": lines,
                }]

        return []

    def get_status(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "version": self.VERSION,
            "path": str(self.path),
            "edges": len(self.edges),
            "entities": len(self.entity_names()),
            "historical_relations": sum(
                len(e.get("history") or []) for e in self.edges.values()
            ),
        }
