# rag/query.py
from __future__ import annotations
import os, json, argparse, sys
from typing import List, Dict, Any, Tuple

from rag import retriever
from rag.pipeline import augment_messages  # reutiliza tu Reason Mode + contexto

def _score(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    t = text.strip()
    bullets = len([1 for ln in t.splitlines() if ln.lstrip().startswith(("-", "•"))])
    nums    = len([1 for ln in t.splitlines() if ln.lstrip()[:2].isdigit()])
    s = 1.0 + min(bullets, 4) * 0.5 + min(nums, 4) * 0.3
    if "Respuesta final:" in t: s += 0.7
    if "Por qué" in t or "Por que" in t: s += 0.5
    if len(t) < 40: s -= 0.6
    if len(t) > 4000: s -= 0.5
    return s

def _best(cands: List[str]) -> str:
    return sorted((c for c in cands if isinstance(c, str) and c.strip()),
                  key=_score, reverse=True)[0] if cands else ""

def _post(url: str, headers: Dict[str,str], payload: Dict[str,Any]) -> Tuple[int, Dict[str,Any]]:
    import requests
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    try:
        data = r.json()
    except Exception:
        data = {"text": r.text}
    return r.status_code, data

def answer_via_llama(user_message: str,
                     server_url: str | None = None,
                     api_key: str | None = None,
                     model_id: str | None = None) -> Dict[str, Any]:
    server_url = (server_url or os.getenv("LLAMA_SERVER_URL") or "http://127.0.0.1:8123").rstrip("/")
    api_key    = api_key or os.getenv("LLAMA_API_KEY") or ""
    model_id   = model_id or os.getenv("LLAMA_REMOTE_MODEL_ID") or ".\\models\\Qwen2.5-3B-Instruct-Q4_K_M.gguf"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-api-key"] = api_key

    messages, cits = augment_messages(user_message)
    payload = {"model": model_id, "messages": messages, "temperature": 0.5, "max_tokens": 512, "n": 3}

    code, data = _post(f"{server_url}/v1/chat/completions", headers, payload)
    if 200 <= code < 300:
        cands = [ (ch.get("message") or {}).get("content","") for ch in (data.get("choices") or []) ]
        return {"reply": _best(cands), "citations": cits, "route": "chat", "status": code}

    # fallback a completions (plantilla Qwen)
    preamble = messages[0]["content"]
    ctx      = messages[1]["content"]
    prompt = (
        f"<|im_start|>system\n{preamble}\n<|im_end|>\n"
        f"<|im_start|>system\n{ctx}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_message}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    payload2 = {"model": model_id, "prompt": prompt, "temperature": 0.5, "max_tokens": 512, "n": 3}
    code2, data2 = _post(f"{server_url}/v1/completions", headers, payload2)
    cands2 = [ ch.get("text","") for ch in (data2.get("choices") or []) ]
    return {"reply": _best(cands2), "citations": cits, "route": "completions", "status": code2}

def main(argv=None):
    p = argparse.ArgumentParser("rag.query")
    p.add_argument("question", nargs="?", help="Pregunta a responder")
    p.add_argument("--q", dest="question_opt", help="Pregunta (alternativa a posicional)")
    p.add_argument("--topk", type=int, default=3, help="Documentos a recuperar (RAG)")
    p.add_argument("--min-score", type=float, default=0.05, help="Umbral mínimo de similitud")
    p.add_argument("--answer", action="store_true", help="Llamar al modelo (llama-cpp server) con Reason Mode")
    p.add_argument("--server-url", default=None, help="URL del llama-cpp server")
    p.add_argument("--api-key", default=None, help="API key del server")
    p.add_argument("--model", default=None, help="Ruta/ID del modelo remoto")
    p.add_argument("--json", action="store_true", help="Salida JSON (máquina)")
    args = p.parse_args(argv)

    q = args.question or args.question_opt
    if not q:
        print("Falta pregunta. Ejemplo: python -m rag.query \"¿Qué hay en los docs?\"")
        sys.exit(2)

    # Recuperación (contexto)
    ctx_text, hits = retriever.rag_answer(query=q, top_k=max(1,args.topk), min_score=max(0.0,args.min_score))
    cits = retriever.citations(hits)

    if not args.answer:
        if args.json:
            print(json.dumps({"context": ctx_text, "citations": cits}, ensure_ascii=False, indent=2))
        else:
            print("=== CONTEXTO RAG ==="); print(ctx_text)
            print("\n=== CITAS ==="); print(cits)
        return

    # Respuesta con modelo (Reason Mode + n-best)
    result = answer_via_llama(q, server_url=args.server_url, api_key=args.api_key, model_id=args.model)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("=== RESPUESTA ==="); print(result.get("reply","").strip())
        print("\n=== CITAS ==="); print(result.get("citations", []))
        print(f"\n(route={result.get('route')}, status={result.get('status')})")

if __name__ == "__main__":
    main()
