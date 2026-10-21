# tests/smoke/test_llama_server.py
import os, requests

ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://127.0.0.1:8020/v1/chat/completions")
MODEL = os.environ.get("LLAMA_MODEL", "qwen2.5-3b-instruct-q4_k_m")

def test_up():
    r = requests.post(ENDPOINT, json={
        "model": MODEL,
        "messages": [
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":"Reply with 'pong' only."}
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "stop": ["<|im_end|>"]
    }, timeout=30)
    assert r.status_code == 200
    data = r.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content","").lower()
    assert "pong" in text
