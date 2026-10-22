# scripts/bench/llama_bench.py
import time, json, statistics, argparse, requests, uuid
from typing import List

def run_once(endpoint: str, model: str, prompt: str, max_tokens: int = 128, temperature: float = 0.2):
    started = time.time()
    r = requests.post(endpoint, json={
        "model": model,
        "messages": [
            {"role":"system","content":"You are a concise assistant."},
            {"role":"user","content": prompt}
        ],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["<|im_end|>"]
    }, timeout=60)
    elapsed = time.time() - started
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {})
    prompt_toks = usage.get("prompt_tokens")
    compl_toks = usage.get("completion_tokens")
    total_toks = usage.get("total_tokens")
    # Fallback si falta usage
    if compl_toks is None:
        text = data.get("choices", [{}])[0].get("message", {}).get("content","")
        compl_toks = max(1, int(len(text.split()) * 1.33))  # aproximación burda
    tokens_per_s = compl_toks / elapsed if elapsed > 0 else 0.0
    return {
        "elapsed_s": elapsed,
        "tokens_per_s": tokens_per_s,
        "prompt_tokens": prompt_toks,
        "completion_tokens": compl_toks,
        "total_tokens": total_toks,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8020/v1/chat/completions")
    ap.add_argument("--model", default="qwen2.5-3b-instruct-q4_k_m")
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    latencies = []
    speeds = []
    for i in range(args.runs):
        m = run_once(args.endpoint, args.model, f"Say a random unique id then a short haiku. id={uuid.uuid4()}")
        latencies.append(m["elapsed_s"])
        speeds.append(m["tokens_per_s"])
        print(json.dumps(m))

    def p(v: List[float], q: float) -> float:
        return round(statistics.quantiles(v, n=100)[int(q)-1], 4) if len(v) >= 100 else round(sorted(v)[int(len(v)*q/100)], 4)

    print("\n=== SUMMARY ===")
    print(f"latency_p50_s={p(latencies,50)} latency_p95_s={p(latencies,95)}")
    print(f"toks_per_s_p50={p(speeds,50)} toks_per_s_p95={p(speeds,95)}")

if __name__ == "__main__":
    main()
