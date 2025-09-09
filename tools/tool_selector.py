from __future__ import annotations

import json
import re
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def _project_root(start: Optional[Path] = None) -> Path:
    """
    Busca la raíz del proyecto donde está manifest.json, subiendo desde /tools/.
    """
    p = (start or Path(__file__)).resolve()
    for candidate in [p, *p.parents]:
        m = candidate / "manifest.json"
        if m.exists():
            return candidate
    # fallback: carpeta padre de tools/
    return Path(__file__).resolve().parent.parent

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _tokenize(s: str) -> List[str]:
    s = _norm(s)
    # quitar puntuación común
    s = re.sub(r"[^\wáéíóúñü]+", " ", s, flags=re.IGNORECASE)
    return [t for t in s.split() if t]

def _safe_get(d: Dict[str, Any], key: str, default: Any = None):
    return d[key] if isinstance(d, dict) and key in d else default


# ---------------------------------------------------------------------
# Manifest y ToolSpec
# ---------------------------------------------------------------------

@dataclass
class ToolSpec:
    name: str
    module: str
    entrypoint: str = "run"
    args_schema: Dict[str, str] = None
    cost: str = "low"
    risks: str = ""
    keywords: List[str] = None
    examples: List[str] = None
    available: bool = False
    reason: str = ""
    _callable: Any = None  # se setea al resolver dinámicamente

    def whitelist_args(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filtra kwargs según args_schema; ignora extras inesperados.
        Soporta tipos anotados como: "str", "int", "float", "bool" y con "?" opcional.
        """
        if not self.args_schema:
            return kwargs
        out: Dict[str, Any] = {}
        for k, ann in self.args_schema.items():
            opt = ann.endswith("?")
            t = ann[:-1] if opt else ann
            if k not in kwargs:
                continue
            v = kwargs[k]
            try:
                if t == "str":
                    out[k] = str(v)
                elif t == "int":
                    out[k] = int(v)
                elif t == "float":
                    out[k] = float(v)
                elif t == "bool":
                    out[k] = bool(v)
                else:
                    # tipo desconocido -> pasar tal cual
                    out[k] = v
            except Exception:
                # si no se puede castear, lo omitimos silenciosamente
                continue
        return out


class ToolRegistry:
    def __init__(self, manifest_path: Optional[Path] = None):
        self.root = _project_root()
        self.manifest_path = manifest_path or (self.root / "manifest.json")
        self.tools: Dict[str, ToolSpec] = {}
        self._load_manifest()
        self._resolve_callables()

    # ---------------------- carga y resolución ----------------------

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest.json no encontrado en {self.manifest_path}")
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if "tools" not in data or not isinstance(data["tools"], list):
            raise ValueError("manifest.json inválido: falta 'tools'")

        for t in data["tools"]:
            spec = ToolSpec(
                name=_safe_get(t, "name", ""),
                module=_safe_get(t, "module", ""),
                entrypoint=_safe_get(t, "entrypoint", "run"),
                args_schema=_safe_get(t, "args_schema", {}) or {},
                cost=_safe_get(t, "cost", "low"),
                risks=_safe_get(t, "risks", ""),
                keywords=[_norm(k) for k in (_safe_get(t, "keywords", []) or [])],
                examples=[_norm(e) for e in (_safe_get(t, "examples", []) or [])],
            )
            if spec.name:
                self.tools[spec.name] = spec

    def _resolve_callables(self) -> None:
        for spec in self.tools.values():
            try:
                mod = importlib.import_module(spec.module)
                fn = getattr(mod, spec.entrypoint, None)
                if callable(fn):
                    spec._callable = fn
                    spec.available = True
                else:
                    spec.available = False
                    spec.reason = f"entrypoint '{spec.entrypoint}' no encontrado en {spec.module}"
            except Exception as e:
                spec.available = False
                spec.reason = f"no se pudo importar {spec.module}: {e}"

    # ---------------------- consulta/ejecución ----------------------

    def list_tools(self, available_only: bool = True) -> List[ToolSpec]:
        items = list(self.tools.values())
        if available_only:
            items = [t for t in items if t.available]
        return items

    def best_match(self, query: str, top_k: int = 1) -> List[Tuple[ToolSpec, float]]:
        """
        Matching sencillo basado en tokens que pondera:
        - coincidencias con keywords (x2)
        - coincidencias con ejemplos (x1)
        - coincidencias de nombre (x3 si aparece de forma explícita)
        """
        q_tokens = set(_tokenize(query))
        results: List[Tuple[ToolSpec, float]] = []
        for spec in self.list_tools(available_only=True):
            score = 0.0
            # nombre explícito
            if spec.name in query or spec.module in query:
                score += 3.0
            # keywords
            for kw in spec.keywords or []:
                kw_tokens = set(_tokenize(kw))
                inter = q_tokens & kw_tokens
                if inter:
                    score += 2.0 * len(inter)
            # ejemplos
            for ex in spec.examples or []:
                ex_tokens = set(_tokenize(ex))
                inter = q_tokens & ex_tokens
                if inter:
                    score += 1.0 * len(inter)
            # pequeños bonos por palabras clave genéricas
            if any(t in q_tokens for t in ["leer", "archivo", "lista", "directorio"]):
                if spec.name == "files_io":
                    score += 1.0
            if any(t in q_tokens for t in ["http", "https", "web", "url"]):
                if spec.name == "http_fetch":
                    score += 1.0
            results.append((spec, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max(1, int(top_k))]

    def run_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        spec = self.tools.get(name)
        if not spec:
            return {"ok": 0, "error": f"tool '{name}' no registrada"}
        if not spec.available:
            return {"ok": 0, "error": f"tool '{name}' no disponible: {spec.reason}"}
        safe_kwargs = spec.whitelist_args(kwargs or {})
        try:
            out = spec._callable(**safe_kwargs)  # type: ignore[misc]
            if isinstance(out, dict) and "ok" not in out:
                out = {"ok": 1, **out}
            return out
        except Exception as e:
            return {"ok": 0, "error": f"error ejecutando {name}: {e}"}

    def decide_and_run(self, query: str, fallback: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Elige la mejor herramienta para la consulta y la ejecuta.
        Si no hay score suficiente, usa 'fallback' si se provee.
        """
        ranked = self.best_match(query, top_k=1)
        if not ranked:
            if fallback:
                return self.run_tool(fallback, **kwargs)
            return {"ok": 0, "error": "no hay herramientas disponibles"}
        spec, score = ranked[0]
        # umbral muy bajo para proyectos chicos; ajusta si agregas más tools
        if score <= 0 and fallback:
            return self.run_tool(fallback, **kwargs)
        return self.run_tool(spec.name, **kwargs)


# ---------------------------------------------------------------------
# API de módulo de alto nivel
# ---------------------------------------------------------------------

_registry: Optional[ToolRegistry] = None

def _get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry

def select(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Devuelve [(tool_name, score), ...] de las mejores coincidencias.
    """
    reg = _get_registry()
    ranked = reg.best_match(query, top_k=top_k)
    return [(spec.name, score) for spec, score in ranked]

def run(name: str, **kwargs) -> Dict[str, Any]:
    """
    Ejecuta una herramienta por nombre.
    """
    return _get_registry().run_tool(name, **kwargs)

def decide_and_run(query: str, **kwargs) -> Dict[str, Any]:
    """
    Decide y ejecuta una herramienta según la consulta en lenguaje natural.
    """
    return _get_registry().decide_and_run(query, **kwargs)

def list_tools(available_only: bool = True) -> List[Dict[str, Any]]:
    reg = _get_registry()
    return [
        {
            "name": t.name,
            "module": t.module,
            "entrypoint": t.entrypoint,
            "available": t.available,
            "reason": t.reason,
            "keywords": t.keywords,
            "examples": t.examples,
        }
        for t in reg.list_tools(available_only=available_only)
    ]


# ---------------------------------------------------------------------
# CLI básico: `python -m tools.tool_selector "consulta ..."`
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "lista directorio"
    print(">> Consulta:", q)
    print(">> Mejores coincidencias:", select(q))
    if select(q):
        name, _ = select(q, top_k=1)[0]
        print(f">> Ejecutando '{name}' (demo)...")
        # Ejemplos mínimos de kwargs para que no truene:
        kwargs = {}
        if name == "files_io":
            kwargs = {"path": ".", "action": "list"}
        if name == "http_fetch":
            kwargs = {"url": "https://example.com", "timeout": 5, "text_max_chars": 5000}
        print(run(name, **kwargs))
