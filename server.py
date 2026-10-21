# server.py — AGI Prototype (robusto; intenta reply_service o reason_service). Puerto 8020 por defecto.
from __future__ import annotations

import os, math, statistics, inspect
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import Body, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi import Security

# ---------------- Fallbacks si faltan módulos opcionales ----------------
try:
    from memory.unified_memory import UnifiedMemory
except Exception:
    class UnifiedMemory:
        def __init__(self, *a, **kw): pass
        def retrieve_relevant_memories(self, q: str, top_k: int = 5): return []
        def add_to_vector_memory(self, *a, **kw): return {"mem_id": f"mem://{uuid4()}"}
        def get_status(self): return {"ok": True, "index_size": 0}

try:
    from reasoner.orchestrator import SkillOrchestrator
except Exception:
    class SkillOrchestrator:
        def __init__(self, **kw): pass
        def execute(self, query: str, k: int = 3) -> Dict[str, Any]:
            return {"answer": f"Puedo ayudarte con “{query}”. ¿Qué detalle necesitas exactamente?"}

try:
    from reasoner.task_planner import TaskPlanner
except Exception:
    class TaskPlanner:
        def plan(self, goal: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
            ctx = context or {}
            return [
                {"id": "1", "action": "entender_objetivo", "input": goal, "deps": []},
                {"id": "2", "action": "reunir_contexto", "input": list(ctx.keys()), "deps": ["1"]},
                {"id": "3", "action": "resolver", "input": goal, "deps": ["2"]},
            ]

try:
    from reasoner.curriculum import CurriculumBuilder
except Exception:
    class CurriculumBuilder:
        def build(self, failures: List[Dict[str, Any]], max_items: int = 10) -> List[Dict[str, Any]]:
            out = []
            for i, f in enumerate(failures[:max_items], 1):
                q = f.get("query") or f.get("input") or f"tarea_{i}"
                out.append({"id": str(i), "practice": f"Reintenta: {q}", "hint": "Divide y valida sub-pasos."})
            return out

# ---------------- Carga flexible de servicios de respuesta ----------------
_RS_cls = _fn_generate = _fn_reply = None

def _try_load_service(module_name: str):
    global _RS_cls, _fn_generate, _fn_reply
    if _RS_cls or _fn_generate or _fn_reply:
        return
    try:
        mod = __import__(module_name, fromlist=["*"])
    except Exception:
        return
    _RS_cls = getattr(mod, "ReplyService", None)
    _fn_generate = getattr(mod, "generate_reply", None)
    _fn_reply = getattr(mod, "reply", None)

# prefer reply_service, luego reason_service
_try_load_service("services.reply_service")
_try_load_service("services.reason_service")

# -------------------------------- Seguridad --------------------------------
API_KEY_HEADER_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

def swagger_api_key(api_key: str = Security(api_key_header)) -> str:
    # Permite probar /docs sin validar; solo valida en middleware fuera de la whitelist
    return api_key or ""

app = FastAPI(title="AGI Prototype", version="3.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

DOCS_WHITELIST = {"/", "/docs", "/redoc", "/openapi.json", "/status", "/get_status"}

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    required_key = (os.getenv("API_KEY") or "").strip()
    if not required_key or request.url.path in DOCS_WHITELIST:
        return await call_next(request)

    sent_key = request.headers.get(API_KEY_HEADER_NAME)
    if not sent_key:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            sent_key = auth[7:].strip()
    if sent_key != required_key:
        return JSONResponse({"detail": "Forbidden: invalid or missing API Key"}, status_code=403)
    return await call_next(request)

def custom_openapi() -> dict:
    if app.openapi_schema: return app.openapi_schema
    schema = get_openapi(
        title=app.title, version=app.version,
        description="Endpoints de AGI con memoria y orquestación",
        routes=app.routes,
    )
    schema.setdefault("components", {}).setdefault("securitySchemes", {})["ApiKeyAuth"] = {
        "type": "apiKey", "in": "header", "name": API_KEY_HEADER_NAME,
        "description": "Introduce tu API Key (solo el valor).",
    }
    schema["security"] = [{"ApiKeyAuth": []}]
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi  # type: ignore

# ----------------------------- Estado global ------------------------------
UM = UnifiedMemory(memory_dir="memory_store", use_gpu=False)
ORCH = SkillOrchestrator(memory_vector_provider=None, fs_root=".", score_threshold=0.45)
PLANNER = TaskPlanner()
CURRICULUM = CurriculumBuilder()

COUNTERS: Dict[str, int] = {
    "total_requests": 0, "chat_requests": 0,
    "upserts_total": 0, "search_requests": 0,
    "vec_upserts_total": 0, "vec_search_requests": 0,
}
LAT_LOG: Dict[str, List[float]] = {k: [] for k in ["chat","upsert","search","vec_upsert","vec_search"]}

def _record_latency(bucket: str, start_ts: float) -> None:
    LAT_LOG.setdefault(bucket, []).append((perf_counter() - start_ts) * 1000.0)

def _avg(x: List[float]) -> Optional[float]:
    return float(statistics.fmean(x)) if x else None

def _p95(x: List[float]) -> Optional[float]:
    if not x: return None
    s = sorted(x); idx = max(0, math.ceil(0.95 * len(s)) - 1)
    return float(s[idx])

# ------------------------- Utilidades de memoria --------------------------
def _vector_provider(query: str, k: int) -> List[Dict[str, Any]]:
    scale = float(os.environ.get("CONF_SCALE", "200.0"))
    try:
        res = UM.retrieve_relevant_memories(query, top_k=k)
    except Exception as e:
        return [{"text": f"[vector error] {type(e).__name__}: {e}", "score": 0.0, "confidence": 0.0}]
    out: List[Dict[str, Any]] = []
    for r in res:
        raw = float(r.get("score", 0.0))
        conf = max(0.0, min(1.0, raw * scale))
        out.append({
            "text": r.get("text", ""), "score": raw, "confidence": conf,
            "citation_id": r.get("citation_id") or r.get("mem_id"),
            "metadata": r.get("metadata", {})
        })
    return out

# ------------------------------ Endpoints base ----------------------------
@app.get("/")
def health() -> Dict[str, Any]:
    return {"ok": True}

@app.get("/auth/echo", dependencies=[Depends(swagger_api_key)])
def auth_echo() -> Dict[str, Any]:
    return {"ok": True, "auth": "passed"}

@app.get("/get_status")
def get_status() -> Dict[str, Any]:
    try:
        um_status = UM.get_status() if hasattr(UM, "get_status") else {}
    except Exception as e:
        um_status = {"ok": False, "error": type(e).__name__, "msg": str(e)}
    index_size_vec = 0
    if isinstance(um_status, dict):
        for k in ("index_size", "ntotal", "size", "count"):
            v = um_status.get(k)
            if isinstance(v, (int, float)):
                index_size_vec = int(v); break
    return {
        "ok": True,
        "counters": {**COUNTERS},
        "latency_ms": {
            "chat_avg": _avg(LAT_LOG["chat"]), "chat_p95": _p95(LAT_LOG["chat"]),
            "upsert_avg": _avg(LAT_LOG["upsert"]), "upsert_p95": _p95(LAT_LOG["upsert"]),
            "search_avg": _avg(LAT_LOG["search"]), "search_p95": _p95(LAT_LOG["search"]),
            "vec_upsert_avg": _avg(LAT_LOG["vec_upsert"]), "vec_upsert_p95": _p95(LAT_LOG["vec_upsert"]),
            "vec_search_avg": _avg(LAT_LOG["vec_search"]), "vec_search_p95": _p95(LAT_LOG["vec_search"]),
        },
        "vector_status": um_status, "index_size_vector": index_size_vec,
    }

@app.get("/status")
def status_alias() -> Dict[str, Any]:
    return get_status()

# ------------------------------ Chat inteligente --------------------------
def _safe_call(callable_obj, rich_kwargs: Dict[str, Any], base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Intenta con kwargs ricos; si la firma no los soporta, filtra o usa kwargs mínimos.
    Devuelve siempre un dict con 'reply'.
    """
    try:
        sig = inspect.signature(callable_obj)
        def filter_kwargs(kwargs):
            return {k: v for k, v in kwargs.items() if k in sig.parameters}
        try:
            res = callable_obj(**filter_kwargs(rich_kwargs))
        except TypeError:
            res = callable_obj(**filter_kwargs(base_kwargs))
    except Exception:
        try:
            res = callable_obj(**rich_kwargs)
        except Exception:
            res = callable_obj(**base_kwargs)

    if isinstance(res, dict) and "reply" in res: return res
    if isinstance(res, dict) and "text" in res: return {"reply": res["text"], **res}
    if isinstance(res, str): return {"reply": str(res)}
    return {"reply": str(res)}

def _call_reply_service(messages: List[Dict[str, Any]], mode: str, k: int) -> Dict[str, Any]:
    base_kwargs = dict(messages=messages, mode=mode, k=k)
    rich_kwargs = dict(**base_kwargs,
                       orchestrator=ORCH, planner=PLANNER,
                       curriculum=CURRICULUM, vector_provider=_vector_provider)

    # 1) Clase ReplyService
    if _RS_cls is not None:
        try:
            try:
                svc = _RS_cls(**rich_kwargs)
            except TypeError:
                svc = _RS_cls()
            if hasattr(svc, "reply"):
                try:
                    return _safe_call(getattr(svc, "reply"), rich_kwargs, base_kwargs)
                except Exception:
                    pass
            if callable(svc):
                try:
                    return _safe_call(svc, rich_kwargs, base_kwargs)
                except Exception:
                    pass
        except Exception as e:
            return {"reply": f"[reply_service error] {type(e).__name__}: {e}"}

    # 2) Funciones sueltas (generate_reply / reply)
    for fn in (_fn_generate, _fn_reply):
        if fn is None: continue
        try:
            return _safe_call(fn, rich_kwargs, base_kwargs)
        except Exception as e:
            return {"reply": f"[reply_fn error] {type(e).__name__}: {e}"}

    # 3) Fallback: mini-orquestador
    user_msg = next((m.get("content", "") for m in reversed(messages)
                     if m.get("role") == "user"), "").strip()
    if not user_msg: return {"reply": "¿En qué te ayudo?"}
    if mode == "reason":
        out = ORCH.execute(user_msg, k=k)
        txt = out.get("answer") or out.get("result") or out.get("text") or str(out)
        return {"reply": txt, "debug": {"used": "orchestrator_fallback"}}
    return {"reply": f"{user_msg} (recibido)"}

@app.post("/chat")
async def chat(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        COUNTERS["chat_requests"] += 1
        msgs = payload.get("messages", []) or []
        mode = (payload.get("mode") or "reason").lower()
        k = int(payload.get("k", 3))
        result = _call_reply_service(messages=msgs, mode=mode, k=k)
        if isinstance(result, dict) and "reply" in result and "text" not in result:
            result["text"] = result["reply"]
        result.setdefault("ok", True)
        result.setdefault("mode_used", mode)
        return result
    finally:
        _record_latency("chat", t0)

# ------------------- Memoria semántica (RAM simple de compatibilidad) ----
STORE: List[Dict[str, Any]] = []

def _simple_score(q: str, t: str) -> float:
    if not q or not t: return 0.0
    ql, tl = q.lower(), t.lower()
    hits = tl.count(ql)
    return 0.0 if hits <= 0 else min(1.0, hits / max(1, len(tl) / max(1, len(ql))))

@app.post("/memory/semantic/upsert")
async def semantic_upsert(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        facts = payload.get("facts", []) or []
        req_trace = payload.get("trace_id") or str(uuid4())
        upserted, mem_ids = 0, []
        for f in facts:
            text = (f or {}).get("text")
            if not text: continue
            mem_id = f.get("mem_id") or f"mem://{uuid4()}"
            STORE.append({
                "mem_id": mem_id, "doc_id": f.get("doc_id") or mem_id, "text": text,
                "meta": f.get("meta") or {}, "result_quality": f.get("result_quality", "unknown"),
                "confidence": float(f.get("confidence", 0.5)), "trace_id": f.get("trace_id") or req_trace,
            })
            mem_ids.append(mem_id); upserted += 1
        COUNTERS["upserts_total"] += upserted
        return {"ok": True, "upserted": upserted, "trace_id": req_trace, "mem_ids": mem_ids}
    finally:
        _record_latency("upsert", t0)

@app.post("/memory/semantic/search")
async def semantic_search(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        COUNTERS["search_requests"] += 1
        q = (payload.get("q") or "").strip()
        k = int(payload.get("k", 5))
        scored: List[Tuple[float, Dict[str, Any]]] = [(_simple_score(q, f.get("text", "")), f) for f in STORE]
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for s, f in scored[:k]:
            row = dict(f); row["score"] = float(s)
            row["confidence_source"] = row.get("confidence", 0.5)
            row["confidence"] = float(s)
            row["citation_id"] = f.get("mem_id") or f.get("doc_id")
            out.append(row)
        return {"ok": True, "results": out, "query": q, "k": k}
    finally:
        _record_latency("search", t0)

# ------------------------------ Memoria vectorial -------------------------
@app.post("/memory/vector/upsert")
async def vector_upsert(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        facts = payload.get("facts", []) or []
        req_trace = payload.get("trace_id")
        upserted, mem_ids = 0, []
        for f in facts:
            text = (f or {}).get("text")
            if not text: continue
            meta = f.get("metadata") or f.get("meta") or {}
            res = UM.add_to_vector_memory(
                text, metadata=meta,
                result_quality=f.get("result_quality", "unknown"),
                confidence=float(f.get("confidence", 0.5)),
                trace_id=f.get("trace_id") or req_trace,
            )
            mem_id = None
            if isinstance(res, tuple) and len(res) >= 2: mem_id = res[1]
            elif isinstance(res, dict): mem_id = res.get("mem_id")
            elif isinstance(res, str): mem_id = res
            if mem_id: mem_ids.append(mem_id); upserted += 1
        COUNTERS["vec_upserts_total"] += upserted
        return {"ok": True, "upserted": upserted, "mem_ids": mem_ids}
    finally:
        _record_latency("vec_upsert", t0)

@app.post("/memory/vector/search")
async def vector_search(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        q = (payload.get("q") or "").strip()
        k = int(payload.get("k", 5))
        results = _vector_provider(q, k)
        return {"ok": True, "results": results, "query": q, "k": k}
    finally:
        _record_latency("vec_search", t0)

# ----------------------------------- Main ---------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8020"))  # 8020 por defecto
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=True)
