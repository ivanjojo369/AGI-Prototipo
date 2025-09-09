#!/usr/bin/env python3
import time, json, sys, requests

API = "http://127.0.0.1:8010"
STOP = ["<|end_of_turn|>", "</s>"]
SYS = ("Tú eres un asistente técnico conciso y accionable.\n"
       "- Responde en ≤ 8 líneas salvo que pidan detalle/código.\n"
       "- Prioriza pasos, listas cortas y comandos.\n"
       "- Si hay riesgo de verborrea/lentitud, limita la salida.")

def trim_at_stop(text:str) -> str:
    for s in STOP:
        i = text.find(s)
        if i >= 0:
            return text[:i].strip()
    return text.strip()

def search_memory(q:str, k:int=3):
    try:
        r = requests.post(f"{API}/memory/semantic/search", json={"q": q, "k": k}, timeout=5)
        r.raise_for_status()
        data = r.json()
        return [x.get("text","") for x in data.get("results", []) if x.get("text")]
    except Exception:
        return []

def chat_once(user_text:str, use_memory:bool=True, max_new:int=120, temp:float=0.6):
    messages = [{"role":"system","content":SYS}]
    if use_memory:
        facts = search_memory(user_text)
        if facts:
            ctx = f"Contexto útil (memoria): {' | '.join(facts)[:350]}"
            messages.append({"role":"system","content":ctx})
    messages.append({"role":"user","content":user_text})

    payload = {
        "messages": messages,
        "params": {
            "max_new_tokens": max_new,
            "temperature": temp,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "repeat_penalty": 1.08,
            "stop": STOP,
            "stream": False
        }
    }
    t0 = time.time()
    r = requests.post(f"{API}/chat", json=payload, timeout=60)
    dt = (time.time()-t0)*1000
    r.raise_for_status()
    data = r.json()
    text = data.get("text") or data.get("content") or (data.get("choices",[{}])[0].get("message",{}).get("content",""))
    return trim_at_stop(text), int(dt)

def main():
    print("AGI Demo CLI – escribe y Enter (Ctrl+C para salir).")
    while True:
        try:
            u = input("\nTú> ").strip()
            if not u: continue
            resp, dt = chat_once(u, use_memory=True, max_new=120, temp=0.6)
            print(f"AGI ({dt} ms)> {resp}")
        except KeyboardInterrupt:
            print("\nSaliendo…"); sys.exit(0)
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
