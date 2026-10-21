from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class MemoryCandidate:
    """Recuerdo candidato recuperado por cualquier subsistema de memoria."""

    content: Any
    memory_type: str = "unknown"
    source: Optional[str] = None
    relevance: float = 0.0
    entity_match: float = 0.0
    temporal_match: float = 0.0
    confidence: float = 0.0
    accepted: bool = False
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveState:
    """Estado cognitivo compartido de un turno de Quipu Cognitive Control 1.0."""

    user_query: str
    intent: Optional[str] = None
    secondary_intents: List[str] = field(default_factory=list)
    decision_confidence: float = 0.0

    recent_context: List[Dict[str, Any]] = field(default_factory=list)
    world_state: Dict[str, Any] = field(default_factory=dict)
    conversation_state: Dict[str, Any] = field(default_factory=dict)

    memory_candidates: List[MemoryCandidate] = field(default_factory=list)
    accepted_memories: List[MemoryCandidate] = field(default_factory=list)
    rejected_memories: List[MemoryCandidate] = field(default_factory=list)

    missing_information: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    requires_memory: bool = False
    requires_long_range: bool = False
    requires_calculation: bool = False
    requires_planning: bool = False
    requires_verification: bool = True
    requires_self_review: bool = False
    requires_tools: bool = False

    tools_needed: List[str] = field(default_factory=list)

    draft_answer: Optional[str] = None
    verification_result: Optional[Dict[str, Any]] = None
    correction_attempts: int = 0
    turn_metadata: Dict[str, Any] = field(default_factory=dict)

    def add_tool(self, tool_name: str) -> None:
        if tool_name not in self.tools_needed:
            self.tools_needed.append(tool_name)

    def add_missing(self, item: str) -> None:
        if item and item not in self.missing_information:
            self.missing_information.append(item)

    def add_conflict(self, item: str) -> None:
        if item and item not in self.conflicts:
            self.conflicts.append(item)

    def add_assumption(self, item: str) -> None:
        if item and item not in self.assumptions:
            self.assumptions.append(item)

    def accept_memory(self, candidate: MemoryCandidate) -> None:
        candidate.accepted = True
        candidate.rejection_reason = None
        if candidate not in self.accepted_memories:
            self.accepted_memories.append(candidate)
        if candidate in self.rejected_memories:
            self.rejected_memories.remove(candidate)

    def reject_memory(self, candidate: MemoryCandidate, reason: str) -> None:
        candidate.accepted = False
        candidate.rejection_reason = reason
        if candidate not in self.rejected_memories:
            self.rejected_memories.append(candidate)
        if candidate in self.accepted_memories:
            self.accepted_memories.remove(candidate)

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)
