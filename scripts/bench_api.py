# scripts/bench_api.py  (auto-descubre el puerto)
from __future__ import annotations
import time, statistics as st, json, urllib.request

CANDIDATES = [8010, 8000, 8787, 8081]

PROMPTS = [
    "hola",
    "me llamo Alejandro",
    "¿cómo me llamo?",
    "estoy en Guadalajara",
    "me gusta programar AGI",
    "dame 3 ideas rápidas para concentrarme",
]

def get_json(url, timeout=5):
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def post_json(url, payload, timeout=60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def find_api():
    for port in CANDIDATES:
        url = f"http://127.0.0.1:{port}/health"
        try:
            j = get_json(url, timeout=1.5)
            if j.get("status") == "ok":
                return f"http://127.0.0.1:{port}"
        except Exception:
            continue
    raise SystemExit("No encontré la API. ¿Está corriendo agi_webapp.py?")

def main(rounds: int = 3):
    base = find_api()
    chat_url = base + "/chat"
    print(f"[i] Usando API: {chat_url}")

    lat = []; sizes = []
    for _ in range(rounds):
        for p in PROMPTS:
            t0 = time.perf_counter()
            out = post_json(chat_url, {"message": p})
            dt = (time.perf_counter() - t0) * 1000
            reply = (out or {}).get("reply", "") or ""
            lat.append(dt); sizes.append(len(reply))
            preview = reply[:140].replace("\n", " ")
            print(f"[{dt:7.1f} ms] > {p}\n{preview}{' ...' if len(reply)>140 else ''}\n")

    if lat:
        p95 = st.quantiles(lat, n=100)[94]
        print("=== API STATS ===")
        print(f"count={len(lat)}  mean={st.mean(lat):.1f} ms  p50={st.median(lat):.1f} ms  p95={p95:.1f} ms")
        print(f"reply_len mean={st.mean(sizes):.0f} chars")

if __name__ == "__main__":
    main()
