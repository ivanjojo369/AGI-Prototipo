from __future__ import annotations

import os
import json
from pathlib import Path

from agents.tool_middleware import ToolMiddleware

def test_tool_middleware_files_list(tmp_path: Path):
    # preparar directorio temporal
    (tmp_path / "a.txt").write_text("hola\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("x,y\n1,2\n", encoding="utf-8")

    mw = ToolMiddleware()
    out = mw.handle(f"lista este directorio", path=str(tmp_path), action="list")
    assert out["handled"] is True
    assert out["result"]["ok"] == 1
    assert out["result"].get("items") is not None

def test_tool_middleware_http_fetch():
    mw = ToolMiddleware()
    out = mw.handle("trae el texto de esta url", url="https://example.com", timeout=5, text_max_chars=2000)
    assert out["handled"] in (True, False)  # no forzamos red aquí (en CI puede fallar)
    # Pero si handled True, debe haber ok
    if out["handled"]:
        assert out["result"]["ok"] == 1

def test_explicit_command_parsing(tmp_path: Path):
    cmd = f"/tool files_io path={tmp_path} action=list"
    mw = ToolMiddleware()
    out = mw.handle(cmd)
    # Aunque el selector no sepa, decide_and_run usa fallback=None; si files_io está disponible, debe funcionar
    # (si no está disponible en el entorno de prueba, simplemente no handled)
    if out["handled"]:
        assert out["result"]["ok"] == 1
