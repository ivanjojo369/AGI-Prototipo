# scripts/run_eval.py
import os, sys, json, csv, time, re, datetime, requests, yaml
from typing import List, Dict

BASE_URL = os.getenv("QUIPU_BASE_URL", "http://localhost:8000")
MODEL = os.getenv("QUIPU_MODEL", "openchat")
CANON_PATH = os.getenv("EVAL_CANON_PATH", "eval/canon.yaml")

OUT_DIR = os.path.join("eval", "results", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def load_canon(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def call_chat(messages: List[Dict]) -> str:
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {"model": MODEL, "messages": messages, "temperature": 0.2}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # Compat con OpenAI-like
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, ensure_ascii=False)

def match_expectations(text: str, case: Dict) -> Dict:
    ok_all = True
    any_ok = False

    exp_all = case.get("expect_all") or []
    exp_any = case.get("expect_any") or []

    details = []

    for pat in exp_all:
        found = re.search(pat, text, flags=re.IGNORECASE|re.DOTALL) is not None
        ok_all = ok_all and found
        details.append({"type":"all", "pattern":pat, "found":found})

    for pat in exp_any:
        found = re.search(pat, text, flags=re.IGNORECASE|re.DOTALL) is not None
        any_ok = any_ok or found
        details.append({"type":"any", "pattern":pat, "found":found})

    passed = (ok_all and (any_ok or not exp_any))
    return {"passed": passed, "checks": details}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    canon = load_canon(CANON_PATH)
    cases = canon.get("cases", [])
    results = []

    for idx, case in enumerate(cases, 1):
        cid = case.get("id") or f"case_{idx}"
        messages = case.get("messages", [])
        t0 = time.time()
        try:
            out = call_chat(messages)
            elapsed = time.time() - t0
            checks = match_expectations(out, case)
            results.append({
                "id": cid,
                "ok": checks["passed"],
                "latency_s": round(elapsed, 3),
                "output": out,
                "checks": checks["checks"]
            })
            print(f"[{cid}] ok={checks['passed']} t={elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "id": cid,
                "ok": False,
                "latency_s": round(elapsed, 3),
                "error": str(e)
            })
            print(f"[{cid}] ERROR {e}")

    # JSON completo
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # CSV resumen
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","ok","latency_s"])
        for r in results:
            w.writerow([r["id"], r.get("ok"), r.get("latency_s")])

    # pequeña métrica agregada
    total = len(results)
    passed = sum(1 for r in results if r.get("ok"))
    with open(os.path.join(OUT_DIR, "aggregate.txt"), "w", encoding="utf-8") as f:
        f.write(f"passed={passed}/{total}\n")

if __name__ == "__main__":
    sys.exit(main())
