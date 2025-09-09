# benchmarks/bench_tool_use.py
from __future__ import annotations
import json, pathlib, csv, datetime
from collections import Counter, defaultdict

from tools.tool_selector import choose_tool

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks" / "data" / "tool_use.jsonl"
OUTDIR  = ROOT / "logs" / "benchmarks"
OUTDIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    cases = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip(): 
            continue
        cases.append(json.loads(line))
    return cases

def run_bench():
    cases = load_dataset()
    correct = 0
    rows = []
    conf = defaultdict(int)
    for c in cases:
        pred, reason, score = choose_tool(c["prompt"])
        gold = c["gold_tool"]
        ok = (pred == gold)
        if ok: correct += 1
        conf[(gold or "None", pred or "None")] += 1
        rows.append({
            "case_id": c["id"],
            "prompt": c["prompt"],
            "gold_tool": gold,
            "pred_tool": pred,
            "ok": int(ok),
            "score": f"{score:.2f}",
            "reason": reason
        })
    acc = correct / max(1, len(cases))
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_csv = OUTDIR / f"tool_use_run_{ts}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Total: {len(cases)}  Correctas: {correct}  Accuracy: {acc:.3f}")
    print(f"CSV: {out_csv}")
    print("Confusiones:")
    for (g,p), n in sorted(conf.items()):
        print(f"  gold={g:9s} pred={p:9s} -> {n}")
    return acc

if __name__ == "__main__":
    run_bench()
