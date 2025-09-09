from __future__ import annotations
import subprocess, sys, textwrap, tempfile, os
from typing import Optional, Dict

# Ejecuta Python en proceso separado y "aislado" (-I) con timeout.
# *Sin* acceso a archivos/red por diseño del snippet; aún así no es una caja perfecta.

MAX_OUTPUT = 8_000  # caracteres


def run(code: str, timeout_s: float = 2.0, stdin: str = "") -> Dict[str, str | int]:
    if not code:
        return {"ok": 0, "stdout": "", "stderr": "código vacío"}
    snippet = textwrap.dedent(code)
    with tempfile.TemporaryDirectory() as td:
        # Evita escribir ficheros; sólo pasamos por stdin
        p = subprocess.run(
            [sys.executable, "-I", "-"],
            input=stdin.encode() if stdin else None,
            text=False,
            capture_output=True,
            timeout=timeout_s,
            cwd=td,
            env={"PYTHONIOENCODING": "utf-8"},
            check=False,
        )
    out = (p.stdout or b"").decode(errors="replace")[:MAX_OUTPUT]
    err = (p.stderr or b"").decode(errors="replace")[:MAX_OUTPUT]
    return {"ok": 1 if p.returncode == 0 else 0, "returncode": p.returncode, "stdout": out, "stderr": err}
