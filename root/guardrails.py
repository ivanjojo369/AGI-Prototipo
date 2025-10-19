# -*- coding: utf-8 -*-
"""
Guardrails ligeros para QuipuLoop:
- Pre-chequeo de consulta (longitud, comandos peligrosos)
- Post-chequeo de salida (redacción de secretos, truncado suave)
- Sin dependencias externas
"""
from __future__ import annotations
import re
from typing import Dict, Any, List, Optional

# Imports robustos de settings
try:
    from .settings import STEP_SOFT_TIMEOUT_SECS, ALLOWLIST_TOOLS
except Exception:  # pragma: no cover
    try:
        from settings import STEP_SOFT_TIMEOUT_SECS, ALLOWLIST_TOOLS  # type: ignore
    except Exception:
        STEP_SOFT_TIMEOUT_SECS = 5
        ALLOWLIST_TOOLS = {"qa_rag": True, "write_note": True}

# Config propia del módulo (puedes moverla a settings si quieres)
MAX_QUERY_CHARS   = 2000
MAX_OUTPUT_CHARS  = 4000

# Patrones de secretos comunes (añade los tuyos si hace falta)
_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{16,}", re.I),               # estilo OpenAI key
    re.compile(r"AKIA[0-9A-Z]{16}"),                        # AWS Access Key
    re.compile(r"(?i)secret[-_\s]*key\s*[:=]\s*[\w\-]{8,}"),# 'secret key: ...'
    re.compile(r"(?i)api[-_\s]*key\s*[:=]\s*[\w\-]{8,}"),   # 'api key=...'
]

_DANGEROUS = [
    r"rm\s+-rf\s+/", r"del\s+/s\s+/q", r"format\s+c:",
    r"shutdown\s+/s", r"powershell\s+Stop-Computer",
]

def precheck_query(query: str) -> Dict[str, Any]:
    """Valida la consulta antes de ejecutar el loop."""
    q = query or ""
    warnings: List[str] = []
    ok = True

    if len(q) > MAX_QUERY_CHARS:
        warnings.append(f"Consulta muy larga (> {MAX_QUERY_CHARS} chars); podría truncarse.")
    low = q.lower()
    for pat in _DANGEROUS:
        if re.search(pat, low):
            warnings.append("Se detectaron comandos peligrosos en la consulta; se ignorarán.")
            ok = False
            break

    return {"ok": ok, "warnings": warnings, "soft_timeout_secs": STEP_SOFT_TIMEOUT_SECS}

def redact_secrets(text: str) -> str:
    out = text or ""
    for rgx in _PATTERNS:
        out = rgx.sub("***REDACTED***", out)
    return out

def postcheck_output(text: str, *, latency_ms: Optional[int] = None) -> Dict[str, Any]:
    """Redacta secretos y aplica un truncado suave si excede el máximo."""
    actions: List[str] = []
    t = redact_secrets(text)
    if t != (text or ""):
        actions.append("redacted_secrets")
    if len(t) > MAX_OUTPUT_CHARS:
        t = t[:MAX_OUTPUT_CHARS] + " …[truncado]"
        actions.append("truncated")
    if latency_ms is not None and latency_ms > STEP_SOFT_TIMEOUT_SECS * 1000:
        actions.append(f"soft_timeout_exceeded({latency_ms}ms)")
    return {"ok": True, "output": t, "actions": actions}
