# -*- coding: utf-8 -*-
"""
skills/router.py
Enrutador ligero de "skills" (acciones) para la AGI doméstica.

- Registro dinámico de skills con decorador @router.register("nombre").
- Llamada segura con manejo de errores y salida normalizada.
- Skills incluidas (si los módulos existen):
    • rag.search(query, top_k=5, min_score=0.45)
    • rag.answer(query, top_k=5, min_score=0.45)
    • memory.write(text, user=None, project_id=None, tags=None, importance=0.5)
    • memory.search(query, top_k=5, user=None, project_id=None, tags=None)
    • memory.reindex(project_id=None, user='all')
    • memory.prune()            # compacción / mantenimiento
    • index.folder(path='.', ext='.py,.md,.txt', mode='append', chunk=1200, overlap=200)

Todas las skills devuelven dict con al menos: {"ok": bool, ...}
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Dict, Optional

# ------------------------------------------------------------
# Utilidad: importaciones tolerantes (para no romper en ausencia)
# ------------------------------------------------------------

def _try_import(func: Callable[[], Any]) -> Optional[Any]:
    try:
        return func()
    except Exception:
        return None


# Intentamos cargar RAG
_rag = _try_import(lambda: __import__("rag.retriever", fromlist=["*"]))
if _rag:
    search_rag = getattr(_rag, "search", None)
    citations_rag = getattr(_rag, "citations", None)
    rag_answer = getattr(_rag, "rag_answer", None)
    rag_repair = getattr(_rag, "repair", None)  # opcional
else:
    search_rag = citations_rag = rag_answer = rag_repair = None

# Intentamos cargar fachada de Memoria
_mem = _try_import(lambda: __import__("memory.memory", fromlist=["*"]))
if _mem:
    memory_write = getattr(_mem, "write", None)
    memory_search = getattr(_mem, "search", None)
    memory_reindex = getattr(_mem, "reindex", None)
    memory_prune = getattr(_mem, "prune", None)
else:
    memory_write = memory_search = memory_reindex = memory_prune = None

# Intentamos cargar script de indexado
_idx = _try_import(lambda: __import__("scripts.index_folder", fromlist=["*"]))
if _idx:
    idx_main = getattr(_idx, "main", None)
else:
    idx_main = None


# ------------------------------------------------------------
# Router de skills
# ------------------------------------------------------------

class SkillError(RuntimeError):
    pass


class SkillRouter:
    def __init__(self) -> None:
        self._skills: Dict[str, Callable[..., Dict[str, Any]]] = {}

    # Decorador para registrar
    def register(self, name: Optional[str] = None):
        def _decorator(fn: Callable[..., Dict[str, Any]]):
            key = name or fn.__name__
            if key in self._skills:
                raise SkillError(f"Skill '{key}' ya registrada.")
            self._skills[key] = fn
            return fn
        return _decorator

    def has(self, name: str) -> bool:
        return name in self._skills

    def names(self):
        return sorted(self._skills.keys())

    # Llamada segura con manejo de errores
    def call(self, name: str, **kwargs) -> Dict[str, Any]:
        fn = self._skills.get(name)
        if not fn:
            return {"ok": False, "error": f"Skill '{name}' no existe.", "available": self.names()}
        try:
            out = fn(**kwargs)
            if not isinstance(out, dict):
                out = {"ok": True, "result": out}
            if "ok" not in out:
                out["ok"] = True
            return out
        except Exception as e:
            return {
                "ok": False,
                "error": f"{e}",
                "trace": traceback.format_exc(limit=4),
            }


# Singleton del router
router = SkillRouter()


# ------------------------------------------------------------
# Implementaciones de skills (se registran solo si hay módulos)
# ------------------------------------------------------------

# -------- RAG --------
if search_rag is not None:

    @router.register("rag.search")
    def _rag_search(query: str, top_k: int = 5, min_score: float = 0.45, **_) -> Dict[str, Any]:
        hits = search_rag(query, top_k=top_k, min_score=min_score)
        cites = citations_rag(hits) if citations_rag else []
        return {"ok": True, "hits": hits, "citations": cites, "stats": {"top_k": top_k, "min_score": min_score}}

if rag_answer is not None:

    @router.register("rag.answer")
    def _rag_answer(query: str, top_k: int = 5, min_score: float = 0.45, **kwargs) -> Dict[str, Any]:
        text, ctx = rag_answer(query, top_k=top_k, min_score=min_score, **kwargs)
        return {
            "ok": True,
            "output": text,
            "context": ctx,
            "stats": {"rag_hits": len(ctx.get("rag_hits", [])), "mem_hits": len(ctx.get("mem_hits", [])),
                      "top_k": top_k, "min_score": min_score},
        }

if rag_repair is not None:

    @router.register("rag.repair")
    def _rag_repair(**kwargs) -> Dict[str, Any]:
        # Implementación opcional; passthrough de parámetros
        res = rag_repair(**kwargs)
        ok = isinstance(res, dict) and res.get("ok", True)
        return res if isinstance(res, dict) else {"ok": ok, "result": res}


# -------- Memoria --------
if memory_write is not None:

    @router.register("memory.write")
    def _mem_write(text: str, user: Optional[str] = None, project_id: Optional[str] = None,
                   tags: Optional[list[str]] = None, importance: float = 0.5, **_) -> Dict[str, Any]:
        res = memory_write(text=text, user=user, project_id=project_id, tags=tags or [], importance=importance)
        return res if isinstance(res, dict) else {"ok": True, "result": res}

if memory_search is not None:

    @router.register("memory.search")
    def _mem_search(query: str, top_k: int = 5, user: Optional[str] = None,
                    project_id: Optional[str] = None, tags: Optional[list[str]] = None, **_) -> Dict[str, Any]:
        hits = memory_search(query=query, top_k=top_k, user=user, project_id=project_id, tags=tags or [])
        return {"ok": True, "hits": hits, "stats": {"top_k": top_k}}

if memory_reindex is not None:

    @router.register("memory.reindex")
    def _mem_reindex(project_id: Optional[str] = None, user: str = "all", **_) -> Dict[str, Any]:
        res = memory_reindex(project_id=project_id, user=user)
        return res if isinstance(res, dict) else {"ok": True, "result": res}

if memory_prune is not None:

    @router.register("memory.prune")
    def _mem_prune(**_) -> Dict[str, Any]:
        res = memory_prune()
        return res if isinstance(res, dict) else {"ok": True, "result": res}


# -------- Indexador (carpeta -> índice semántico) --------
if idx_main is not None:

    @router.register("index.folder")
    def _index_folder(path: str = ".", ext: str = ".py,.md,.txt", mode: str = "append",
                      chunk: int = 1200, overlap: int = 200, **_) -> Dict[str, Any]:
        """
        Envuelve el script de indexado. Equivalente CLI:
        python -m scripts.index_folder --path <path> --ext <ext> --mode <mode> --chunk 1200 --overlap 200
        """
        argv = [
            "--path", path,
            "--ext", ext,
            "--mode", mode,
            "--chunk", str(chunk),
            "--overlap", str(overlap),
        ]
        # idx_main debe aceptar argv estilo sys.argv[1:]
        out = idx_main(argv)  # type: ignore
        if isinstance(out, dict):
            return out
        return {"ok": True, "result": out, "args": {"path": path, "ext": ext, "mode": mode,
                                                    "chunk": chunk, "overlap": overlap}}


# ------------------------------------------------------------
# API pública
# ------------------------------------------------------------

def call(skill: str, **kwargs) -> Dict[str, Any]:
    """Atajo global: skills.router.call(...)"""
    return router.call(skill, **kwargs)


def available() -> list[str]:
    """Devuelve la lista de skills disponibles."""
    return list(router.names())


__all__ = ["router", "call", "available", "SkillRouter", "SkillError"]
