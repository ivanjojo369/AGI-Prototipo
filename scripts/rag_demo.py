# scripts/rag_demo.py
import os, sys, requests
from memory.semantic_memory import SemanticMemory

import re

def extract_year(text: str) -> str:
    m = re.search(r"\b(1[6-9]\d{2}|20\d{2})\b", text)
    return m.group(0) if m else text

API_URL = os.environ.get("AGENT_API_URL", "http://127.0.0.1:8010/chat")
API_KEY = os.environ.get("API_KEY", "CUKJMXoqkHYS2Zapxfl0tD85wyPnueOLE4sQARNr")

# --- construcción de prompt: SIN ### ---
def build_prompt(context: str, question: str) -> str:
    return (
        "Responde usando SOLO el contexto. "
        "Si no está en el contexto, responde exactamente: no sé.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pregunta: {question}\n"
        "Formato de salida: devuelve un único número de cuatro dígitos, sin nada más.\n"
        "Respuesta: "
    )

# --- llamada al servidor: SIN stop; max_tokens corto pero suficiente ---
def call_chat(prompt: str, max_tokens: int = 16):
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        # nada de "stop"
    }
    r = requests.post(API_URL, json=body, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json().get("text", "").strip()

def ask(question: str, k: int = 3):
    sm = SemanticMemory()
    hits = sm.search(question, top_k=k)
    print(f"\nQ: {question}")
    if not hits:
        print("No hay hits en memoria.")
        ctx = ""
    else:
        print("Top-k memoria:", [(h.get("doc_id") or h.get("id"), round(h.get("score",0.0),3)) for h in hits])
        ctx = "\n".join([h.get("text","") for h in hits])

    prompt = (
        "### Instrucción:\n"
        "Usa SOLO el siguiente contexto para responder breve y correcto.\n\n"
        f"Contexto:\n{ctx}\n\n"
        f"Pregunta: {question}\n### Respuesta: "
    )
    ans = call_chat(prompt)
    print("Respuesta:", ans)
    return ans

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]), k=3)
    else:
        # Demo rápido
        for q in [
            "¿Cuándo se publicó la Crítica de la razón pura?",
            "¿Quién creó el lenguaje Python?",
            "¿Cuál es la capital de Francia?",
        ]:
            ask(q, k=3)
