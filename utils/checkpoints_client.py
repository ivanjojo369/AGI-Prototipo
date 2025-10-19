# utils/checkpoints_client.py — cliente para /jobs/checkpoint (resiliente)
from __future__ import annotations

import os
from typing import Optional, Any, Dict

import requests

BASE = os.getenv("JOBS_BASE_URL", "http://127.0.0.1:8010").rstrip("/")
URL = os.getenv("JOBS_CHECKPOINT_URL", f"{BASE}/jobs/checkpoint")
API_KEY = (os.getenv("API_KEY") or "").strip()
TIMEOUT = float(os.getenv("JOBS_HTTP_TIMEOUT", "5"))

def _headers() -> Dict[str, str]:
    if not API_KEY:
        return {}
    # Preferimos x-api-key, pero añadimos alternativa Bearer por compatibilidad
    return {"x-api-key": API_KEY, "Authorization": f"Bearer {API_KEY}"}

def push_ckpt(job_id: Optional[str], **fields: Any) -> bool:
    """
    Envía un checkpoint al servidor de jobs. Devuelve True/False.
    No levanta excepciones para no romper el flujo del planner.
    """
    if not job_id:
        return False
    try:
        payload = {"job_id": job_id, **fields}
        r = requests.post(URL, json=payload, headers=_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        return True
    except Exception:
        return False
