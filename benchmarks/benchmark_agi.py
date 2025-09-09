# benchmarks/benchmark_agi.py
from __future__ import annotations
import argparse, csv, json, pathlib, time, datetime
from typing import Dict, Any, List, Optional, Tuple

# --- AGENTE (HTTP al servidor local) ---
from agents.agent import Agent, AgentConfig

# --- SELECTOR DE HERRAMIENTAS ---
try:
    from tools.tool_selector import choose_tool
except Exception:
    choose_tool = None  # corre sin selector si aún no lo agregas

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "logs" / "benchmarks"
LOGDIR.mkdir(parents=True, exist_ok=True)

def load_jsonl(p: Optional[str]) -> List[Dict[str, Any]]:
    if not p:
        return []
    path = pathlib.Path(p)
    if not path.exists():
        print(f"[WARN] dataset no encontrado: {path}")
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out

def norm_text(s: str) -> str:
    return " ".join(str(s).split())

def approx_tokens(s: str) -> int:
    # aproximación rápida (no depende de tokenizer específico)
    return max(1, len(s.split()))

def eval_tool_use(prompt: str, gold_tool: Optional[str]) -> Tuple[Optional[str], float, str, int]:
    """ Devuelve (tool_pred, score, reason, ok_int) """
    if choose_tool is None:
        return None, 0.0, "selector no disponible", int(gold_tool is None)
    pred, reason, score = choose_tool(prompt)
    ok = int(gold_tool is not None and pred == gold_tool)
    return pred, score, reason, ok

def run_case(agent: Agent, case: Dict[str, Any]) -> Dict[str, Any]:
    """ Ejecuta un caso y devuelve fila para CSV. """
    cid = case.get("id") or case.get("case_id") or f"case-{int(time.time()*1000)}"
    prompt = case.get("prompt") or case.get("input") or ""
    gold_tool = case.get("gold_tool")  # sólo para dataset de herramientas

    # Predicción de herramienta (no bloqueante)
    tool_pred, tool_score, tool_reason, ok_tool = eval_tool_use(prompt, gold_tool)

    # Ejecuta el agente (HTTP al servidor GGUF)
    t0 = time.perf_counter()
    out = agent.run(prompt)
    lat_ms = int((time.perf_counter() - t0) * 1000)

    # Métricas mínimas
    tokens_out = approx_tokens(out)

    # pass@1 (para tool_use: si predicción es correcta; para otros casos queda vacío)
    pass_at_1 = ok_tool if gold_tool else ""

    # Campos opcionales que quizá midas después
    retrieval_at_k = ""   # no inferimos aquí; tu agente ya escribe métricas por fuera
    used_fact     = ""    # idem; puede estimarse escaneando [doc:ID] si te interesa

    row = {
        "run_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "case_id": cid,
        "prompt": norm_text(prompt)[:300],
        "pass@1": pass_at_1,
        "lat_ms": lat_ms,
        "ttft_ms": "",             # requiere streaming; dejamos vacío
        "tokens_out": tokens_out,
        "retrieval_at_k": retrieval_at_k,
        "used_fact": used_fact,
        "fixed_on_retry": "",      # si quieres, emítelo desde el agente y pásalo por kwargs
        # Tool-use
        "tool_gt": gold_tool or "",
        "tool_pred": tool_pred or "",
        "tool_score": f"{tool_score:.2f}" if tool_pred is not None else "",
        "tool_reason": tool_reason or "",
        "notes": case.get("notes",""),
        "output": out[:500]  # recorte de ejemplo
    }
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="JSONL principal (golden general)", default="")
    ap.add_argument("--toolset", help="JSONL de tool-use (gold_tool/golden)", default="benchmarks/data/tool_use.jsonl")
    ap.add_argument("--out_csv", help="Ruta CSV de salida", default="")
    ap.add_argument("--api_url", default="http://127.0.0.1:8010/chat")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=110)
    args = ap.parse_args()

    # Instancia del agente
    cfg = AgentConfig(api_url=args.api_url, api_key=args.api_key, max_new_tokens=args.max_new_tokens)
    agent = Agent(cfg)

    cases = []
    cases += load_jsonl(args.dataset)
    cases += load_jsonl(args.toolset)

    if not cases:
        print("[ERROR] No hay casos para correr.")
        return

    rows = [run_case(agent, c) for c in cases]

    # CSV
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_csv = pathlib.Path(args.out_csv) if args.out_csv else (LOGDIR / f"run_{ts}.csv")
    fieldnames = [
        "run_id","case_id","prompt","pass@1","lat_ms","ttft_ms","tokens_out",
        "retrieval_at_k","used_fact","fixed_on_retry","tool_gt","tool_pred",
        "tool_score","tool_reason","notes","output"
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Resumen rápido para tool_use
    tool_rows = [r for r in rows if r["tool_gt"]]
    if tool_rows:
        acc = sum(int(r["pass@1"]==1) for r in tool_rows) / max(1, len(tool_rows))
        print(f"[Tool-use] casos: {len(tool_rows)}  accuracy: {acc:.3f}")
    print(f"CSV → {out_csv}")

if __name__ == "__main__":
    main()
