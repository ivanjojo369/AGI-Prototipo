# -*- coding: utf-8 -*-
"""
Fachada de Memoria de largo plazo.
Reexporta la API implementada en episodic_memory.py y NO depende de RAG.
"""
from __future__ import annotations

from .episodic_memory import (
    memory_write,
    memory_search,
    memory_reindex,
    memory_prune,
    EpisodicMemory,
    SemanticMemory,
)

# Aliases cortos
write = memory_write
search = memory_search
reindex = memory_reindex
prune = memory_prune

__all__ = [
    "memory_write", "memory_search", "memory_reindex", "memory_prune",
    "write", "search", "reindex", "prune",
    "EpisodicMemory", "SemanticMemory",
]
