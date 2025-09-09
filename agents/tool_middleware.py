from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from tools.tool_selector import decide_and_run, select, list_tools


@dataclass
class ToolMiddlewareConfig:
    # Umbral para decidir si vale la pena usar herramientas (score del mejor candidato).
    # El selector ya es conservador; 0.0 funciona bien en repos pequeños.
    min_score_required: float = 0.0
    # Herramienta de respaldo si el matching no es suficiente.
    fallback_tool: Optional[str] = None
    # Palabras clave que disparan intento de tool-use antes que el modelo.
    trigger_keywords: tuple[str, ...] = (
        "lista", "directorio", "archivo", "leer", "csv", "excel",
        "url", "http", "https", "descarga", "web",
        "fecha", "hora", "utc", "cdmx",
        "calcula", "resultado", "porcentaje", "raíz", "log", "ln",
    )


class ToolMiddleware:
    """
    Middleware minimalista para decidir y ejecutar herramientas antes del LLM.
    Úsalo desde tu loop: si handle() devuelve handled=True, ya respondió la herramienta.
    """

    def __init__(self, cfg: Optional[ToolMiddlewareConfig] = None):
        self.cfg = cfg or ToolMiddlewareConfig()

    @staticmethod
    def available_tools() -> list[dict[str, Any]]:
        return list_tools(available_only=True)

    def should_try_tools(self, user_text: str) -> bool:
        t = (user_text or "").lower()
        if any(k in t for k in self.cfg.trigger_keywords):
            return True
        # Heurísticas simples basadas en formato / comando
        if t.strip().startswith("/tool"):
            return True
        return False

    def parse_explicit_command(self, user_text: str) -> tuple[Optional[str], Dict[str, Any]]:
        """
        Soporta comandos del tipo:
          /tool http_fetch url=https://example.com timeout=5 text_max_chars=4000
          /tool files_io path=. action=list
        Devuelve (tool_name, kwargs) o (None, {}).
        """
        t = (user_text or "").strip()
        if not t.startswith("/tool"):
            return None, {}
        parts = t.split()
        if len(parts) < 2:
            return None, {}
        tool_name = parts[1].strip()
        kwargs: Dict[str, Any] = {}
        for p in parts[2:]:
            if "=" in p:
                k, v = p.split("=", 1)
                k, v = k.strip(), v.strip()
                # cast sencillos
                if v.isdigit():
                    kwargs[k] = int(v)
                else:
                    try:
                        kwargs[k] = float(v)
                    except Exception:
                        kwargs[k] = v
        return tool_name, kwargs

    def handle(self, user_text: str, **hint_kwargs) -> Dict[str, Any]:
        """
        Intenta resolver con herramientas. Devuelve:
          {
            "handled": bool,
            "tool": str | None,
            "result": dict | None,
            "reason": str
          }
        """
        if not user_text or not user_text.strip():
            return {"handled": False, "tool": None, "result": None, "reason": "empty"}

        # 1) Comando explícito
        tool_name, cmd_kwargs = self.parse_explicit_command(user_text)
        if tool_name:
            out = decide_and_run(f"usar {tool_name}", fallback=self.cfg.fallback_tool, **cmd_kwargs)
            return {
                "handled": bool(out.get("ok")),
                "tool": tool_name,
                "result": out,
                "reason": "explicit_command"
            }

        # 2) Heurística por palabras clave
        if not self.should_try_tools(user_text):
            return {"handled": False, "tool": None, "result": None, "reason": "not_triggered"}

        # 3) Consulta natural → selector
        # Puedes pasarle hints: p.ej., path=".", action="list" si tu UI lo sabe
        out = decide_and_run(user_text, fallback=self.cfg.fallback_tool, **(hint_kwargs or {}))
        tool_used = out.get("tool") if "tool" in out else None  # por si alguna tool devuelve ese campo
        return {
            "handled": bool(out.get("ok")),
            "tool": tool_used,
            "result": out,
            "reason": "selector"
        }
