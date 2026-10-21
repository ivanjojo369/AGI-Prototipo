from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import math
import time


VALID_STATUSES = {
    "current",
    "historical",
    "inferred",
    "unknown",
    "conflict",
}


@dataclass
class WorldFact:
    """
    Una afirmación estructurada dentro del estado de trabajo de Quipu.

    status:
      - current: hecho vigente
      - historical: hecho que fue vigente pero ya fue reemplazado
      - inferred: conclusión derivada, no declarada directamente por el usuario
      - unknown: dato explícitamente desconocido
      - conflict: existe una contradicción no resuelta
    """

    key: str
    value: Any
    status: str = "current"

    source: str = "unknown"
    confidence: float = 1.0

    created_at: float = field(default_factory=time.time)
    superseded_at: Optional[float] = None

    derived_from: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"status inválido: {self.status!r}. "
                f"Valores válidos: {sorted(VALID_STATUSES)}"
            )

        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


class WorkingWorldState:
    """
    Quipu Cognitive Control 1.0
    WorkingWorldState v1.0.0

    Mantiene una representación estructurada y temporal de los hechos
    relevantes de la conversación.

    PRINCIPIOS:
      1. Un hecho nuevo no destruye el anterior: lo mueve a histórico.
      2. Una inferencia no puede sobrescribir silenciosamente un hecho
         autoritativo declarado por el usuario.
      3. Los datos desconocidos se representan como desconocidos.
      4. Las transformaciones numéricas conservan provenance/derived_from.
      5. Los conflictos permanecen visibles hasta que se resuelvan.
    """

    VERSION = "1.0.0"

    AUTHORITATIVE_SOURCES = {
        "user",
        "system",
        "tool",
        "verified",
    }

    def __init__(self) -> None:
        self._facts: Dict[str, List[WorldFact]] = {}
        self._conflicts: Dict[str, List[WorldFact]] = {}

    # ------------------------------------------------------------------
    # Escritura de hechos
    # ------------------------------------------------------------------

    def set_fact(
        self,
        key: str,
        value: Any,
        *,
        source: str = "unknown",
        confidence: float = 1.0,
        authoritative: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorldFact:
        """
        Establece un hecho vigente.

        Si ya existe un hecho current diferente:
          - si la nueva fuente es autoritativa, el anterior pasa a historical.
          - si NO es autoritativa, se registra un conflicto y NO se sustituye
            silenciosamente el hecho vigente.
        """
        self._validate_key(key)

        source_norm = str(source).strip().lower() or "unknown"
        if authoritative is None:
            authoritative = source_norm in self.AUTHORITATIVE_SOURCES

        current = self.get_current_fact(key)

        if current is None:
            fact = WorldFact(
                key=key,
                value=value,
                status="current",
                source=source_norm,
                confidence=confidence,
                metadata=dict(metadata or {}),
            )
            self._facts.setdefault(key, []).append(fact)
            return fact

        if self._same_value(current.value, value):
            # Refuerzo del mismo hecho; no crear duplicados innecesarios.
            current.confidence = max(current.confidence, float(confidence))
            current.metadata.update(metadata or {})
            return current

        if authoritative:
            current.status = "historical"
            current.superseded_at = time.time()

            fact = WorldFact(
                key=key,
                value=value,
                status="current",
                source=source_norm,
                confidence=confidence,
                metadata=dict(metadata or {}),
            )
            self._facts.setdefault(key, []).append(fact)
            return fact

        # Una inferencia / recuerdo no autoritativo que contradice el estado
        # actual queda registrado como conflicto, pero NO toma control.
        conflict = WorldFact(
            key=key,
            value=value,
            status="conflict",
            source=source_norm,
            confidence=confidence,
            metadata=dict(metadata or {}),
        )
        self._facts.setdefault(key, []).append(conflict)
        self._conflicts.setdefault(key, []).append(conflict)
        return conflict

    def set_inferred(
        self,
        key: str,
        value: Any,
        *,
        source: str = "reasoner",
        confidence: float = 0.7,
        derived_from: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorldFact:
        """
        Añade una inferencia sin convertirla automáticamente en verdad vigente.
        """
        self._validate_key(key)

        fact = WorldFact(
            key=key,
            value=value,
            status="inferred",
            source=source,
            confidence=confidence,
            derived_from=list(derived_from or []),
            metadata=dict(metadata or {}),
        )
        self._facts.setdefault(key, []).append(fact)
        return fact

    def mark_unknown(
        self,
        key: str,
        *,
        source: str = "metacognitive_gate",
        reason: Optional[str] = None,
    ) -> WorldFact:
        """
        Representa explícitamente que un dato no se conoce.

        No borra hechos históricos ni inferencias anteriores.
        """
        self._validate_key(key)

        current = self.get_current_fact(key)
        if current is not None:
            current.status = "historical"
            current.superseded_at = time.time()

        fact = WorldFact(
            key=key,
            value=None,
            status="unknown",
            source=source,
            confidence=1.0,
            metadata={"reason": reason} if reason else {},
        )
        self._facts.setdefault(key, []).append(fact)
        return fact

    # ------------------------------------------------------------------
    # Transformaciones numéricas
    # ------------------------------------------------------------------

    def apply_numeric_update(
        self,
        key: str,
        operation: str,
        amount: float,
        *,
        source: str = "reasoner",
        confidence: float = 0.95,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorldFact:
        """
        Aplica una transformación a un hecho numérico vigente.

        operations:
          set
          add
          subtract
          multiply
          percent_loss
          percent_gain
        """
        self._validate_key(key)

        current = self.get_current_fact(key)
        if current is None or current.status != "current":
            raise KeyError(
                f"No existe un hecho vigente para {key!r}; "
                "no se puede aplicar una transformación."
            )

        if not isinstance(current.value, (int, float)) or isinstance(current.value, bool):
            raise TypeError(
                f"El valor vigente de {key!r} no es numérico: {current.value!r}"
            )

        op = operation.strip().lower()
        old_value = float(current.value)
        amount = float(amount)

        if op == "set":
            new_value = amount
        elif op == "add":
            new_value = old_value + amount
        elif op == "subtract":
            new_value = old_value - amount
        elif op == "multiply":
            new_value = old_value * amount
        elif op == "percent_loss":
            self._validate_percent(amount)
            new_value = old_value * (1.0 - amount / 100.0)
        elif op == "percent_gain":
            self._validate_percent(amount)
            new_value = old_value * (1.0 + amount / 100.0)
        else:
            raise ValueError(
                "operation inválida. Usa: set, add, subtract, multiply, "
                "percent_loss o percent_gain."
            )

        # Preservar enteros limpios cuando corresponda.
        if math.isfinite(new_value) and float(new_value).is_integer():
            new_value = int(new_value)

        current.status = "historical"
        current.superseded_at = time.time()

        derived = [
            f"{key}={current.value}",
            f"{operation}({amount})",
        ]

        fact = WorldFact(
            key=key,
            value=new_value,
            status="current",
            source=source,
            confidence=confidence,
            derived_from=derived,
            metadata={
                "operation": op,
                "operand": amount,
                **dict(metadata or {}),
            },
        )

        self._facts.setdefault(key, []).append(fact)
        return fact

    # ------------------------------------------------------------------
    # Lectura
    # ------------------------------------------------------------------

    def get_current_fact(self, key: str) -> Optional[WorldFact]:
        facts = self._facts.get(key, [])
        for fact in reversed(facts):
            if fact.status == "current":
                return fact
        return None

    def get_current(self, key: str, default: Any = None) -> Any:
        fact = self.get_current_fact(key)
        return fact.value if fact is not None else default

    def get_history(self, key: str) -> List[WorldFact]:
        return [
            fact
            for fact in self._facts.get(key, [])
            if fact.status == "historical"
        ]

    def get_inferred(self, key: str) -> List[WorldFact]:
        return [
            fact
            for fact in self._facts.get(key, [])
            if fact.status == "inferred"
        ]

    def get_conflicts(self, key: Optional[str] = None) -> List[WorldFact]:
        if key is not None:
            return list(self._conflicts.get(key, []))

        result: List[WorldFact] = []
        for items in self._conflicts.values():
            result.extend(items)
        return result

    def has_conflict(self, key: str) -> bool:
        return bool(self._conflicts.get(key))

    def keys(self) -> List[str]:
        return sorted(self._facts.keys())

    # ------------------------------------------------------------------
    # Resolución explícita de conflictos
    # ------------------------------------------------------------------

    def resolve_conflict(
        self,
        key: str,
        value: Any,
        *,
        source: str = "user",
        confidence: float = 1.0,
    ) -> WorldFact:
        """
        Resuelve explícitamente un conflicto.

        El valor elegido se vuelve el hecho actual.
        Los conflictos previos permanecen en el historial como trazabilidad,
        pero se eliminan de la lista de conflictos activos.
        """
        self._validate_key(key)

        current = self.get_current_fact(key)
        if current is not None and not self._same_value(current.value, value):
            current.status = "historical"
            current.superseded_at = time.time()

        for conflict in self._conflicts.get(key, []):
            conflict.metadata["resolved"] = True
            conflict.metadata["resolved_value"] = value

        self._conflicts[key] = []

        fact = WorldFact(
            key=key,
            value=value,
            status="current",
            source=source,
            confidence=confidence,
            metadata={"resolved_conflict": True},
        )
        self._facts.setdefault(key, []).append(fact)
        return fact

    # ------------------------------------------------------------------
    # Integración con CognitiveState
    # ------------------------------------------------------------------

    def export_current_state(self) -> Dict[str, Any]:
        """
        Diccionario compacto con únicamente los hechos vigentes.

        Diseñado para inyectarse en CognitiveState.world_state.
        """
        result: Dict[str, Any] = {}

        for key in self.keys():
            fact = self.get_current_fact(key)
            if fact is not None:
                result[key] = fact.value

        return result

    def export_detailed_state(self) -> Dict[str, Any]:
        """
        Estado completo para logs, debugging y Verifier.
        """
        return {
            "version": self.VERSION,
            "current": self.export_current_state(),
            "facts": {
                key: [fact.snapshot() for fact in facts]
                for key, facts in self._facts.items()
            },
            "active_conflicts": {
                key: [fact.snapshot() for fact in facts]
                for key, facts in self._conflicts.items()
                if facts
            },
        }

    def sync_to_cognitive_state(self, cognitive_state: Any) -> Any:
        """
        Copia el estado vigente al atributo world_state de CognitiveState.

        Se evita importar CognitiveState aquí para no crear dependencias
        circulares innecesarias.
        """
        if not hasattr(cognitive_state, "world_state"):
            raise TypeError("El objeto no posee atributo world_state.")

        cognitive_state.world_state = self.export_current_state()

        if hasattr(cognitive_state, "conflicts"):
            for key in sorted(self._conflicts):
                if self._conflicts[key]:
                    message = f"Conflicto activo en world_state: {key}"
                    if message not in cognitive_state.conflicts:
                        cognitive_state.conflicts.append(message)

        return cognitive_state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key debe ser un string no vacío.")

    @staticmethod
    def _validate_percent(value: float) -> None:
        if value < 0 or value > 100:
            raise ValueError("El porcentaje debe estar entre 0 y 100.")

    @staticmethod
    def _same_value(a: Any, b: Any) -> bool:
        return a == b
