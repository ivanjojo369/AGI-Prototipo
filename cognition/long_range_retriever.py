from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .cognitive_state import CognitiveState, MemoryCandidate
from .relevance_gate import MemoryRelevanceGate
from .working_world_state import WorkingWorldState


@dataclass
class LongRangeResult:
    """
    Resultado estructurado de una reconstrucción longitudinal.
    """
    query: str

    accepted_memories: List[MemoryCandidate] = field(default_factory=list)
    rejected_memories: List[MemoryCandidate] = field(default_factory=list)

    reconstructed_state: Dict[str, Any] = field(default_factory=dict)
    missing_keys: List[str] = field(default_factory=list)

    provenance: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    processed_events: int = 0
    controller_notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not self.missing_keys


class LongRangeRetriever:
    """
    Quipu Cognitive Control 1.0
    LongRangeRetriever v1.0.0

    Objetivo:
      - Reconstruir el estado relevante de una conversación extensa.
      - Combinar VARIAS memorias parciales en vez de exigir que una sola
        memoria responda toda la consulta.
      - Filtrar ruido mediante MemoryRelevanceGate.
      - Aplicar los recuerdos aceptados en orden temporal sobre
        WorkingWorldState.
      - Conservar provenance y reportar información faltante.

    IMPORTANTE:
      Esta versión NO usa un LLM para inventar/extraer hechos de texto libre.
      Los recuerdos pueden llevar hechos estructurados en metadata["world_facts"].
      Eso hace la reconstrucción determinista y testeable.

    Formato soportado de world_facts:

      SET:
      {
        "kind": "set",
        "key": "colony.population.initial",
        "value": 100,
        "source": "user"
      }

      UPDATE NUMÉRICO:
      {
        "kind": "numeric_update",
        "key": "colony.food_reserves_kg",
        "operation": "percent_loss",
        "amount": 25,
        "source": "user"
      }

      UNKNOWN:
      {
        "kind": "unknown",
        "key": "colony.population.year_100",
        "reason": "Falta tasa demográfica."
      }

      INFERRED:
      {
        "kind": "inferred",
        "key": "colony.food_reserves_after_loss_kg",
        "value": 30000,
        "source": "reasoner",
        "derived_from": [...]
      }
    """

    VERSION = "1.0.1"

    ORDER_FIELDS = (
        "turn_index",
        "block_index",
        "sequence",
        "order",
        "timestamp",
    )

    def __init__(
        self,
        relevance_gate: Optional[MemoryRelevanceGate] = None,
    ) -> None:
        self.relevance_gate = relevance_gate or MemoryRelevanceGate()

    def reconstruct(
        self,
        query: str,
        candidates: Sequence[MemoryCandidate],
        *,
        required_keys: Optional[Sequence[str]] = None,
        base_world_state: Optional[WorkingWorldState] = None,
    ) -> LongRangeResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query debe ser un string no vacío.")

        world = base_world_state or WorkingWorldState()

        state = CognitiveState(
            user_query=query.strip(),
            memory_candidates=list(candidates),
        )

        # 1) Filtrar relevancia contextual.
        self.relevance_gate.filter_state(state)

        # 1.1) En razonamiento de largo alcance, un recuerdo puede ser
        # imprescindible aunque su texto no repita literalmente la pregunta.
        # Si un recuerdo contiene un world_fact que toca una required_key,
        # se promociona como evidencia estructural. Esto evita perder piezas
        # como la población inicial cuando la consulta solo pregunta por
        # "si alcanzan las reservas".
        self._promote_required_key_candidates(
            state,
            required_keys=required_keys or [],
        )

        # 2) Procesar solo recuerdos aceptados en orden temporal.
        accepted = sorted(
            state.accepted_memories,
            key=self._candidate_order,
        )

        provenance: Dict[str, List[str]] = {}
        warnings: List[str] = []
        processed_events = 0

        for candidate in accepted:
            facts = self._extract_world_facts(candidate)

            if not facts:
                # Puede ser contexto útil sin ser un hecho estructurado.
                continue

            for fact in facts:
                try:
                    touched_key = self._apply_fact(world, candidate, fact)
                    processed_events += 1

                    if touched_key:
                        provenance.setdefault(touched_key, [])
                        source_label = self._source_label(candidate, fact)
                        if source_label not in provenance[touched_key]:
                            provenance[touched_key].append(source_label)

                except Exception as exc:
                    warnings.append(
                        f"No se pudo aplicar un world_fact desde "
                        f"{self._candidate_label(candidate)}: {exc}"
                    )

        reconstructed = world.export_current_state()

        missing_keys: List[str] = []
        for key in required_keys or []:
            if not self._has_known_current_value(world, key):
                missing_keys.append(key)

        result = LongRangeResult(
            query=query.strip(),
            accepted_memories=list(accepted),
            rejected_memories=list(state.rejected_memories),
            reconstructed_state=reconstructed,
            missing_keys=missing_keys,
            provenance=provenance,
            warnings=warnings,
            processed_events=processed_events,
            controller_notes={
                "version": self.VERSION,
                "accepted_count": len(accepted),
                "rejected_count": len(state.rejected_memories),
                "required_keys": list(required_keys or []),
            },
        )

        return result

    def reconstruct_into_cognitive_state(
        self,
        cognitive_state: CognitiveState,
        *,
        required_keys: Optional[Sequence[str]] = None,
        base_world_state: Optional[WorkingWorldState] = None,
    ) -> LongRangeResult:
        """
        Variante de integración directa con CognitiveState.

        Usa cognitive_state.memory_candidates, reconstruye el mundo y
        sincroniza:
          - accepted_memories
          - rejected_memories
          - world_state
          - missing_information
          - conflicts
        """
        if not isinstance(cognitive_state, CognitiveState):
            raise TypeError("cognitive_state debe ser CognitiveState.")

        world = base_world_state or WorkingWorldState()

        result = self.reconstruct(
            cognitive_state.user_query,
            cognitive_state.memory_candidates,
            required_keys=required_keys,
            base_world_state=world,
        )

        cognitive_state.accepted_memories = list(result.accepted_memories)
        cognitive_state.rejected_memories = list(result.rejected_memories)
        cognitive_state.world_state = dict(result.reconstructed_state)

        for key in result.missing_keys:
            message = f"Falta información requerida: {key}"
            if message not in cognitive_state.missing_information:
                cognitive_state.missing_information.append(message)

        # Exportar conflictos activos de WorkingWorldState.
        for conflict in world.get_conflicts():
            message = (
                f"Conflicto activo en {conflict.key}: "
                f"{conflict.value!r} source={conflict.source}"
            )
            if message not in cognitive_state.conflicts:
                cognitive_state.conflicts.append(message)

        cognitive_state.turn_metadata.setdefault("long_range_retriever", {})
        cognitive_state.turn_metadata["long_range_retriever"].update(
            {
                "version": self.VERSION,
                "accepted_count": len(result.accepted_memories),
                "rejected_count": len(result.rejected_memories),
                "processed_events": result.processed_events,
                "missing_keys": list(result.missing_keys),
            }
        )

        return result


    def _promote_required_key_candidates(
        self,
        state: CognitiveState,
        *,
        required_keys: Sequence[str],
    ) -> None:
        """
        Promociona recuerdos rechazados por similitud textual cuando sus
        world_facts alimentan directamente una clave requerida por la tarea.

        Esto NO usa retrieval_score como bypass. La promoción solo ocurre si
        existe una coincidencia estructural exacta entre world_fact.key y una
        required_key explícita del plan de reconstrucción.
        """
        required = {str(key).strip() for key in required_keys if str(key).strip()}
        if not required:
            return

        for candidate in list(state.rejected_memories):
            facts = self._extract_world_facts(candidate)
            touched = {
                str(fact.get("key", "")).strip()
                for fact in facts
                if str(fact.get("key", "")).strip()
            }
            matched = sorted(required & touched)
            if not matched:
                continue

            candidate.metadata.setdefault("long_range_retriever", {})
            candidate.metadata["long_range_retriever"].update(
                {
                    "version": self.VERSION,
                    "promoted_by_required_key": matched,
                    "promotion_reason": (
                        "Evidencia estructural requerida por la reconstrucción "
                        "de largo alcance."
                    ),
                }
            )

            state.accept_memory(candidate)

    # ------------------------------------------------------------------
    # World facts
    # ------------------------------------------------------------------

    def _extract_world_facts(
        self,
        candidate: MemoryCandidate,
    ) -> List[Dict[str, Any]]:
        metadata = candidate.metadata or {}

        raw = metadata.get("world_facts")
        if raw is None:
            raw = metadata.get("world_fact")

        if raw is None:
            return []

        if isinstance(raw, dict):
            raw = [raw]

        if not isinstance(raw, (list, tuple)):
            raise TypeError("world_facts debe ser dict o lista de dicts.")

        facts: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                facts.append(dict(item))
        return facts

    def _apply_fact(
        self,
        world: WorkingWorldState,
        candidate: MemoryCandidate,
        fact: Dict[str, Any],
    ) -> Optional[str]:
        kind = str(fact.get("kind", "set")).strip().lower()
        key = fact.get("key")

        if not isinstance(key, str) or not key.strip():
            raise ValueError("world_fact sin key válida.")

        source = str(
            fact.get(
                "source",
                candidate.metadata.get("source", candidate.source or "memory"),
            )
        ).strip().lower()

        confidence = self._bounded_float(
            fact.get("confidence", candidate.confidence or 0.8)
        )

        metadata = dict(fact.get("metadata") or {})
        metadata.setdefault(
            "memory_type",
            candidate.memory_type,
        )

        if kind == "set":
            if "value" not in fact:
                raise ValueError(f"world_fact set sin value para {key!r}.")

            world.set_fact(
                key,
                fact["value"],
                source=source,
                confidence=confidence,
                authoritative=fact.get("authoritative"),
                metadata=metadata,
            )
            return key

        if kind == "numeric_update":
            operation = fact.get("operation")
            amount = fact.get("amount")

            if not isinstance(operation, str):
                raise ValueError(
                    f"numeric_update sin operation para {key!r}."
                )
            if amount is None:
                raise ValueError(
                    f"numeric_update sin amount para {key!r}."
                )

            world.apply_numeric_update(
                key,
                operation,
                amount,
                source=source,
                confidence=confidence,
                metadata=metadata,
            )
            return key

        if kind == "unknown":
            world.mark_unknown(
                key,
                source=source,
                reason=fact.get("reason"),
            )
            return key

        if kind == "inferred":
            if "value" not in fact:
                raise ValueError(
                    f"world_fact inferred sin value para {key!r}."
                )

            world.set_inferred(
                key,
                fact["value"],
                source=source,
                confidence=confidence,
                derived_from=list(fact.get("derived_from") or []),
                metadata=metadata,
            )
            return key

        raise ValueError(f"kind de world_fact no soportado: {kind!r}")

    # ------------------------------------------------------------------
    # Orden temporal
    # ------------------------------------------------------------------

    def _candidate_order(self, candidate: MemoryCandidate) -> Tuple[int, float, str]:
        """
        Orden determinista.

        1. candidatos con orden explícito
        2. valor numérico del orden
        3. etiqueta estable para desempate
        """
        metadata = candidate.metadata or {}

        for field in self.ORDER_FIELDS:
            if field not in metadata:
                continue

            value = metadata[field]
            try:
                return (0, float(value), self._candidate_label(candidate))
            except (TypeError, ValueError):
                continue

        # Si no hay orden explícito, preservar orden estable al final.
        return (1, 0.0, self._candidate_label(candidate))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_known_current_value(
        world: WorkingWorldState,
        key: str,
    ) -> bool:
        current = world.get_current_fact(key)
        if current is None:
            return False

        if current.status != "current":
            return False

        return current.value is not None

    @staticmethod
    def _source_label(
        candidate: MemoryCandidate,
        fact: Dict[str, Any],
    ) -> str:
        metadata = candidate.metadata or {}

        for field in ("block_index", "turn_index", "sequence", "order"):
            if field in metadata:
                return f"{field}:{metadata[field]}"

        if fact.get("source"):
            return f"source:{fact['source']}"

        if candidate.source:
            return f"source:{candidate.source}"

        return "memory"

    @staticmethod
    def _candidate_label(candidate: MemoryCandidate) -> str:
        text = str(candidate.content).replace("\n", " ").strip()
        if len(text) > 80:
            text = text[:77] + "..."
        return text or "<memory>"

    @staticmethod
    def _bounded_float(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number))
