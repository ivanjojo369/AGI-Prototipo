from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from tools.tool_selector import decide_and_run, list_tools, select, run as run_tool


@dataclass
class ToolAgentConfig:
    fallback_tool: Optional[str] = None   # e.g., "files_io"
    min_score_required: float = 0.0       # mantenlo en 0.0 para entornos chicos


class ToolAgent:
    """
    Agente muy simple que decide y ejecuta herramientas usando tool_selector.
    Diseñado para integrarse en tu loop actual sin reescribirlo.
    """

    def __init__(self, cfg: Optional[ToolAgentConfig] = None):
        self.cfg = cfg or ToolAgentConfig()

    def think(self, query: str):
        """Devuelve candidatos (tool, score) sin ejecutar."""
        return select(query, top_k=3)

    def act(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Decide y ejecuta. kwargs se pasan a la herramienta seleccionada.
        """
        # Si quieres forzar una herramienta explícita, usa run_tool()
        res = decide_and_run(query, fallback=self.cfg.fallback_tool, **kwargs)
        return {
            "ok": int(bool(res.get("ok"))),
            "query": query,
            "tool_result": res,
        }

    @staticmethod
    def available_tools():
        return list_tools(available_only=True)


# CLI de prueba: python -m agents.tool_agent "lista el directorio actual"
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "lista el directorio actual"
    agent = ToolAgent()
    print("Herramientas disponibles:", agent.available_tools())
    print("Candidatos:", agent.think(q))
    # demo kwargs
    kwargs = {}
    if "directorio" in q or "lista" in q:
        kwargs = {"path": ".", "action": "list"}
    out = agent.act(q, **kwargs)
    print("Resultado:", out)
