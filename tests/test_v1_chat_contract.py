# tests/test_v1_chat_contract.py
from fastapi.testclient import TestClient
from agi_interface import app

client = TestClient(app)

def test_v1_chat_ok():
    payload = {
        "model": "openchat",
        "messages": [{"role": "user", "content": "Di hola en una frase"}],
        "temperature": 0.2,
        "max_tokens": 64
    }
    r = client.post("/v1/chat", json=payload)
    assert r.status_code == 200
    data = r.json()
    for k in ("id","object","created","model","choices","usage","request_id"):
        assert k in data
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(data["usage"]["total_tokens"], int)
