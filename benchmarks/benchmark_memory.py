# benchmarks/benchmark_memory.py
# Benchmark de Memoria: bootstrap (golden.jsonl) y evaluación (retrieval@k, used_fact)
# - Bootstrap casi intacto (solo cableado mínimo)
# - Eval actualizada: usa golden + marca hit aun sin embeddings
# - Auth robusta (Bearer + X-API-Key + api_key en query/body)
# - Logging inline en logs/benchmarks/memory_metrics_YYYYMMDD.csv
# - Compat: k/top_k en SemanticMemory, _generate flexible para /chat

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from agi_initializer import get_backend


# ----------------------------- utilidades generales -----------------------------

def goal_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:10]


def today_csv(out_dir: str) -> Path:
    return Path(out_dir) / f"memory_metrics_{time.strftime('%Y%m%d')}.csv"


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_queries(backend, queries: List[str], out_dir: str, strict=False, max_steps=3) -> Path:
    for q in queries:
        backend.run(q, strict=strict, max_steps=max_steps)
    return today_csv(out_dir)


# ------------------------------- API KEY helpers -------------------------------

def _resolve_api_key() -> Tuple[Optional[str], str]:
    key = os.getenv("DEMO_API_KEY") or os.getenv("API_KEY")
    if not key:
        return None, "(auth desactivada)"
    tail = key[-6:] if len(key) >= 6 else key
    return key, f"(***…{tail})"


# --------------------- transporte HTTP “robusto” para /chat --------------------

def _post_json_robust(url: str, payload: dict, timeout: float, api_key: Optional[str],
                      retry_http: int = 1, backoff: float = 0.5) -> dict:
    attempts = max(1, int(retry_http))
    last_err = None

    headers = {"Content-Type": "application/json"}
    params = {}
    body = dict(payload)

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
        params["api_key"] = api_key
        body.setdefault("api_key", api_key)

    for i in range(attempts):
        try:
            r = requests.post(url, json=body, params=params, headers=headers, timeout=timeout)
            if r.status_code == 401:
                tail = api_key[-6:] if api_key else ""
                raise RuntimeError(
                    "401 Unauthorized (verifica que la API key del benchmark coincide con la del servidor). "
                    f"Key usada: ***…{tail}  URL: {url}"
                )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if i < attempts - 1:
                time.sleep(float(backoff) * (2 ** i))
    raise last_err


# -------- compat SemanticMemory: acepta k o top_k indistintamente --------------

def _apply_semantic_memory_compat():
    try:
        import memory.semantic_memory as sm
        if not hasattr(sm, "SemanticMemory"):
            return
        _SM = sm.SemanticMemory

        def _wrap(fn):
            def inner(self, *args, **kwargs):
                try:
                    sig = inspect.signature(fn)
                    if "k" in kwargs and "k" not in sig.parameters:
                        if "top_k" in sig.parameters:
                            kwargs["top_k"] = kwargs.pop("k")
                        else:
                            kwargs.pop("k", None)
                except Exception:
                    pass
                return fn(self, *args, **kwargs)
            return inner

        for name in ("retrieve", "search", "query"):
            if hasattr(_SM, name):
                setattr(_SM, name, _wrap(getattr(_SM, name)))
    except Exception as e:
        print("WARNING: compat patch for SemanticMemory failed:", e)


# --------- logging de métricas inline (sin instrumentar agents/agent.py) --------

def _enable_inline_metrics_logging(
    backend,
    k: int,
    out_dir: str,
    gold_map: Optional[Dict[str, List[str]]] = None,  # solo en eval
):
    import os as _os
    from memory.metrics import compute_memory_metrics

    # Instancia de memoria semántica
    try:
        import memory.semantic_memory as sm
        sem = sm.SemanticMemory() if hasattr(sm, "SemanticMemory") else None
    except Exception:
        sem = None

    def _search(qq: str, top_k: int, _sem=sem):
        """Busca top-k pidiendo embeddings con múltiples banderas soportadas."""
        if not _sem:
            return []
        for fn in ("retrieve", "search", "query"):
            g = getattr(_sem, fn, None)
            if g is None:
                continue
            try:
                try:
                    sig = inspect.signature(g)
                except (ValueError, TypeError):
                    sig = None

                kw = {}
                # Acepta k/top_k
                if sig and "k" in sig.parameters:
                    kw["k"] = top_k
                elif sig and "top_k" in sig.parameters:
                    kw["top_k"] = top_k

                # Banderas comunes para pedir embeddings
                for flag in (
                    "include_embeddings", "with_embeddings", "return_embeddings",
                    "return_embs", "with_embs", "include_embs", "return_vectors"
                ):
                    if sig and flag in sig.parameters:
                        kw[flag] = True

                if kw:
                    return g(qq, **kw)

                # Fallback sin introspección: prueba ambos estilos de k
                try:
                    return g(qq, k=top_k)
                except TypeError:
                    return g(qq, top_k=top_k)
            except Exception:
                continue
        return []

    def _pick_emb(d):
        """Extrae el vector desde múltiples nombres posibles."""
        if isinstance(d, dict):
            for key in ("emb", "embedding", "embeddings", "vector", "vec", "dense", "dense_vec"):
                if key in d:
                    return d[key]
            return None
        # objeto
        for key in ("emb", "embedding", "embeddings", "vector", "vec", "dense", "dense_vec"):
            v = getattr(d, key, None)
            if v is not None:
                return v
        return None

    class RetrievalItem:
        def __init__(self, doc_id, score, text, emb=None):
            self.doc_id = doc_id
            self.score = score
            self.text = text
            self.emb = emb

    class MemoryTrace:
        def __init__(self, goal, retrieved):
            self.goal = goal
            self.retrieved = retrieved

    orig_run = backend.agent.run

    def wrapped_run(goal: str, **kwargs):
        # Tolerante al retorno de run (1, 2 o N valores)
        res = orig_run(goal, **kwargs)
        if isinstance(res, tuple):
            answer = res[0]
            raw = res[1] if len(res) > 1 else None
        else:
            answer = res
            raw = None

        # Gold para esta query (solo en eval)
        gold = None
        if gold_map:
            gold = gold_map.get(goal) or gold_map.get(goal.strip())

        # recuperar top-k (con embeddings si se soporta)
        items = []
        try:
            res_list = _search(goal, k)
            for r in (res_list or []):
                if isinstance(r, dict):
                    doc_id = r.get("doc_id") or r.get("id") or r.get("doc") or ""
                    score = float(r.get("score", 0.0))
                    text = r.get("text") or r.get("content") or ""
                    emb = _pick_emb(r)
                else:
                    doc_id = getattr(r, "doc_id", getattr(r, "id", ""))
                    score = float(getattr(r, "score", 0.0))
                    text = getattr(r, "text", getattr(r, "content", ""))
                    emb = _pick_emb(r)
                items.append(RetrievalItem(doc_id, score, text, emb))
        except Exception as e:
            print("WARN: búsqueda top-k falló:", e)

        trace = MemoryTrace(goal=goal, retrieved=items)

        # --- compute + CSV robusto ---
        try:
            has_emb = any(getattr(it, "emb", None) is not None for it in items)
            if has_emb:
                m = compute_memory_metrics(
                    goal=goal,
                    answer=answer,
                    trace=trace,
                    k=k,
                    run_id="bench",
                    gold_doc_ids=gold if gold else None,  # si tu implementación lo ignora, no pasa nada
                )
            else:
                # Fila mínima; si tenemos golden, marcamos hit@k comparando IDs
                hit = False
                if gold:
                    gold_set = set(gold)
                    hit = any((it.doc_id in gold_set) for it in (items[:k] if items else []))

                m = {
                    "timestamp": int(time.time()),
                    "goal_hash": goal_hash(goal),
                    "goal": goal,
                    "answer": str(answer) if answer is not None else "",
                    "k": k,
                    "support_doc_id": items[0].doc_id if items else "",
                    "retrieval_hit@k": bool(hit),
                    "used_fact": False,
                    "best_cos": 0.0,
                }
        except Exception as e:
            print("WARN: compute_memory_metrics falló:", e)
            m = {
                "timestamp": int(time.time()),
                "goal_hash": goal_hash(goal),
                "goal": goal,
                "answer": str(answer) if answer is not None else "",
                "k": k,
                "support_doc_id": items[0].doc_id if items else "",
                "retrieval_hit@k": False,
                "used_fact": False,
                "best_cos": 0.0,
            }

        try:
            _os.makedirs(out_dir, exist_ok=True)
            csv_path = _os.path.join(out_dir, f"memory_metrics_{time.strftime('%Y%m%d')}.csv")
            need_header = not _os.path.exists(csv_path)

            FIELDS = [
                "timestamp", "goal_hash", "goal", "answer", "k",
                "support_doc_id", "retrieval_hit@k", "used_fact", "best_cos"
            ]
            row = {k2: m.get(k2, "") for k2 in FIELDS}

            import csv as _csv
            with open(csv_path, "a", encoding="utf-8", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=FIELDS)
                if need_header:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:
            print("WARN: no se pudieron escribir métricas:", e)

        return answer, raw

    backend.agent.run = wrapped_run


# ----------------------------- preparación de backend ---------------------------

def _prepare_backend(backend):
    cfg = backend.agent.cfg
    cfg.api_key = os.getenv("DEMO_API_KEY") or os.getenv("API_KEY") or getattr(cfg, "api_key", None)

    # Evita 422 si el server es estricto con campos no soportados
    for name in ("temperature", "top_p", "top_k", "repeat_penalty", "max_new_tokens"):
        if hasattr(cfg, name):
            setattr(cfg, name, None)

    # _generate flexible: intenta múltiples payloads y SIEMPRE devuelve (text, resp)
    def _smart_generate(self, prompt: str):
        shapes = [
            {"prompt": prompt},                                   # mínimo
            {"prompt": prompt, "n_predict": 128},                 # llama.cpp / variantes GGUF
            {"prompt": prompt, "max_tokens": 128},                # otros servers
            {"input": prompt}, {"text": prompt}, {"query": prompt}, {"message": prompt},
            {"messages": [{"role": "user", "content": prompt}], "stream": False},  # estilo OpenAI
        ]
        if getattr(self.cfg, "api_key", None):
            shapes += [
                {"prompt": prompt, "api_key": self.cfg.api_key},
                {"messages": [{"role": "user", "content": prompt}], "api_key": self.cfg.api_key},
            ]

        last = None
        for payload in shapes:
            try:
                resp = _post_json_robust(
                    self.cfg.api_url, payload, getattr(self.cfg, "timeout", 30.0), self.cfg.api_key,
                    getattr(self.cfg, "retry_http", 1), getattr(self.cfg, "retry_backoff", 0.5),
                )
                text = (
                    resp.get("text") or resp.get("output") or resp.get("message")
                    or (resp.get("choices", [{}])[0].get("text") if isinstance(resp.get("choices"), list) else None)
                )
                if not isinstance(text, str):
                    text = json.dumps(text, ensure_ascii=False)
                return text, resp
            except Exception as e:
                last = e
                continue
        raise last

    backend.agent._generate = types.MethodType(_smart_generate, backend.agent)


# ---------------------------------- modos CLI -----------------------------------

def bootstrap(seeds_path: str, out_file: str, out_dir: str, k: int):
    seeds = [ln.strip() for ln in Path(seeds_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not seeds:
        raise SystemExit(f"No hay consultas en {seeds_path}")

    key, hint = _resolve_api_key()
    print(f"[bootstrap] API key: {hint}")
    _apply_semantic_memory_compat()
    backend = get_backend("agent")
    _prepare_backend(backend)
    _enable_inline_metrics_logging(backend, k, out_dir)  # bootstrap sin gold_map

    csv_path = run_queries(backend, seeds, out_dir, strict=False, max_steps=3)
    rows = read_rows(csv_path)

    # última fila por goal_hash (sin filtrar por tiempo)
    last_by_hash: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        gh = r.get("goal_hash")
        if gh:
            last_by_hash[gh] = r

    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for q in seeds:
            gh = goal_hash(q)
            row = last_by_hash.get(gh)
            gold = []
            if row:
                sd = row.get("support_doc_id")
                if sd:
                    gold = [sd]
            f.write(json.dumps({"query": q, "gold_doc_ids": gold}, ensure_ascii=False) + "\n")
    print(f"[BOOTSTRAP] golden escrito en: {out_file}  (seeds={len(seeds)})")


def evaluate(golden_path: str, out_dir: str, k: int, target: float):
    cases = [json.loads(ln) for ln in Path(golden_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not cases:
        raise SystemExit(f"No hay casos en {golden_path}")

    # Mapa query -> gold_doc_ids
    gold_map: Dict[str, List[str]] = {c["query"]: c.get("gold_doc_ids", []) for c in cases}

    key, hint = _resolve_api_key()
    print(f"[eval] API key: {hint}")
    _apply_semantic_memory_compat()
    backend = get_backend("agent")
    _prepare_backend(backend)
    _enable_inline_metrics_logging(backend, k, out_dir, gold_map=gold_map)

    queries = [c["query"] for c in cases]
    csv_path = run_queries(backend, queries, out_dir, strict=False, max_steps=3)
    rows = read_rows(csv_path)

    by_hash: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        gh = r.get("goal_hash")
        if gh:
            by_hash.setdefault(gh, []).append(r)

    correct, total = 0, 0
    used_hits, used_total = 0, 0

    for c in cases:
        gh = goal_hash(c["query"])
        lst = by_hash.get(gh, [])
        if not lst:
            continue
        last = lst[-1]
        total += 1
        hit = last.get("retrieval_hit@k")
        if hit in ("True", "False"):
            correct += (hit == "True")
        if last.get("used_fact") in ("True", "False"):
            used_hits += (last["used_fact"] == "True")
            used_total += 1

    ret_at_k = correct / max(total, 1)
    used_rate = used_hits / max(used_total, 1)

    print(f"[EVAL] retrieval@{k}: {ret_at_k:.3f}  |  used_fact: {used_rate:.3f}  |  N={total}")
    if ret_at_k < target:
        raise SystemExit(f"Fallo KPI: retrieval@{k}={ret_at_k:.3f} < {target:.2f}")


# -------------------------------------- main ------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    ap_boot = sub.add_parser("bootstrap", help="Genera golden.jsonl desde seeds.txt")
    ap_boot.add_argument("--seeds", required=True, help="Ruta a seeds.txt (1 consulta por línea)")
    ap_boot.add_argument("--out", default="benchmarks/data/retrieval_golden.jsonl")
    ap_boot.add_argument("--logs", default="logs/benchmarks")
    ap_boot.add_argument("--k", type=int, default=5)

    ap_eval = sub.add_parser("eval", help="Evalúa un golden.jsonl existente")
    ap_eval.add_argument("--golden", required=True)
    ap_eval.add_argument("--logs", default="logs/benchmarks")
    ap_eval.add_argument("--k", type=int, default=5)
    ap_eval.add_argument("--target", type=float, default=0.90)

    args = ap.parse_args()
    if args.cmd == "bootstrap":
        bootstrap(args.seeds, args.out, args.logs, args.k)
    elif args.cmd == "eval":
        evaluate(args.golden, args.logs, args.k, args.target)
    else:
        ap.print_help()
