from __future__ import annotations
import time, statistics as st
from pathlib import Path
from agi_initializer import build_context, initialize_all

PROMPTS = [
    "hola",
    "me llamo Alejandro",
    "¿cómo me llamo?",
    "estoy en Guadalajara",
    "me gusta programar AGI",
    "dame 3 ideas rápidas para concentrarme",
]

def main(rounds: int = 3):
    ctx = build_context(Path("settings.json"))
    comps = initialize_all(ctx)
    agi = comps["interface_instance"]

    latencies = []
    sizes = []

    for _ in range(rounds):
        for p in PROMPTS:
            t0 = time.perf_counter()
            r = agi.process_message(p)
            dt = (time.perf_counter() - t0) * 1000
            latencies.append(dt)
            sizes.append(len(r or ""))

            preview = (r or "")[:140].replace("\n", " ")
            ellipsis = " ..." if r and len(r) > 140 else ""
            print(f"[{dt:7.1f} ms] > {p}")
            print(preview + ellipsis)
            print()

    if latencies:
        p95 = st.quantiles(latencies, n=100)[94]
        print("=== STATS ===")
        print(f"count={len(latencies)}  mean={st.mean(latencies):.1f} ms  "
              f"p50={st.median(latencies):.1f} ms  p95={p95:.1f} ms")
        print(f"reply_len mean={st.mean(sizes):.0f} chars")

if __name__ == "__main__":
    main()
