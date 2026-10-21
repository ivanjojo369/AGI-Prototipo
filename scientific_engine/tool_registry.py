from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, Optional


@dataclass
class ToolSpec:
    name: str
    domain: str
    callable: Callable[..., Any]
    deterministic: bool = True
    description: str = ""
    available: bool = True
    dependency: Optional[str] = None

    def snapshot(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("callable", None)
        return data


class ToolRegistry:
    VERSION = "1.1.0"

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        *,
        domain: str,
        callable: Callable[..., Any],
        deterministic: bool = True,
        description: str = "",
        available: bool = True,
        dependency: Optional[str] = None,
    ) -> ToolSpec:
        spec = ToolSpec(
            name=str(name),
            domain=str(domain),
            callable=callable,
            deterministic=bool(deterministic),
            description=str(description),
            available=bool(available),
            dependency=dependency,
        )
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(str(name))

    def execute(self, name: str, /, **kwargs: Any) -> Any:
        spec = self.get(name)
        if spec is None:
            raise KeyError(f"Herramienta científica desconocida: {name}")
        if not spec.available:
            dep = f" ({spec.dependency})" if spec.dependency else ""
            raise RuntimeError(f"Herramienta no disponible{dep}: {name}")
        return spec.callable(**kwargs)

    def list_tools(self, *, domain: Optional[str] = None) -> list[Dict[str, Any]]:
        rows = []
        for spec in self._tools.values():
            if domain is not None and spec.domain != domain:
                continue
            rows.append(spec.snapshot())
        return sorted(rows, key=lambda row: row["name"])
