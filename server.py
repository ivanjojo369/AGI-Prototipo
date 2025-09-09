# server.py
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List
# server.py  → alias del servidor principal
from llama_server import app  # reexporta la misma app de FastAPI

app = FastAPI(title="AGI Local Demo")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STORE: List[Dict[str, Any]] = []  # memoria “lite” temporal en RAM

@app.get("/")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(payload: Dict[str, Any] = Body(...)):
    msgs = payload.get("messages", [])
    user = next((m.get("content","") for m in reversed(msgs) if m.get("role")=="user"), "")
    # Respuesta placeholder; tu backend de LLM puede reemplazar esta línea.
    return {"text": f"AGI: {user} <|end_of_turn|>"}

@app.post("/memory/semantic/upsert")
def mem_upsert(payload: Dict[str, Any] = Body(...)):
    facts = payload.get("facts", [])
    for f in facts:
        if f.get("text"): STORE.append(f)
    return {"upserted": len(facts)}

@app.post("/memory/semantic/search")
def mem_search(payload: Dict[str, Any] = Body(...)):
    q = (payload.get("q") or "").lower(); k = int(payload.get("k", 3))
    scored = []
    for f in STORE:
        t = f.get("text","")
        score = (t.lower().count(q) if q else 1)
        scored.append((score, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    return {"results": [x[1] for x in scored[:k]]}
