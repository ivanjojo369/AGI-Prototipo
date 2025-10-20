import os
import re
import time
import requests

BASE = os.getenv("QUIPU_URL", "http://127.0.0.1:8010")
MODEL = os.getenv("MODEL_NAME", "openchat")

def _post_chat(content: str, temperature: float = 0.2, max_tokens: int = 64):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return requests.post(f"{BASE}/v1/chat", json=body, timeout=20)

def test_health_ok():
    r = requests.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert "model" in j
    assert "provider" in j

def test_chat_contract_basic():
    r = _post_chat("Di hola en una sola frase.")
    assert r.status_code == 200
    j = r.json()
    # contrato mínimo
    assert j["object"] == "chat.completion"
    assert isinstance(j["choices"], list) and len(j["choices"]) >= 1
    msg = j["choices"][0]["message"]
    assert isinstance(msg["content"], str) and len(msg["content"]) > 0
    assert "usage" in j and "request_id" in j

def test_language_spanish_heuristic():
    r = _post_chat("Di hola en una sola frase.")
    assert r.status_code == 200
    txt = r.json()["choices"][0]["message"]["content"]
    # Heurística simple de español (palabras y signos frecuentes)
    assert re.search(r"(hola|¿|¡|á|é|í|ó|ú)", txt.lower()) is not None

def test_body_too_large_gives_413():
    # Por defecto el CI levanta con MAX_BODY_BYTES=1048576; mandamos >1.2MB
    huge = "x" * 1_200_000
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": huge}],
        "temperature": 0.2,
        "max_tokens": 8,
    }
    r = requests.post(f"{BASE}/v1/chat", json=body, timeout=20)
    assert r.status_code == 413

def test_rate_limit_bursty_requests_return_some_429():
    """
    El CI arranca el server con RATE_LIMIT_RPS=1 y BURST=2.
    Disparamos varias peticiones muy seguidas y esperamos al menos un 429.
    """
    statuses = []
    for _ in range(8):
        resp = _post_chat("Ping rápido para probar rate-limit", max_tokens=8)
        statuses.append(resp.status_code)
        # sin dormir para estresar el bucket
    assert any(s == 429 for s in statuses), f"statuses: {statuses}"
