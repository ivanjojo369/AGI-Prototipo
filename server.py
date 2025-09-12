# server.py — AGI Prototype (Fases 1, 2 y 3) con API Key, Swagger "Authorize",
# memoria unificada FAISS-CPU, orquestador, planificador HTN y curriculum.
# -----------------------------------------------------------------------------
# Endpoints:
#   GET  /, /status, /get_status
#   GET  /auth/echo                         (protegido; prueba de API key)
#   POST /chat                              (echo | reason | plan | curriculum)
#   POST /memory/semantic/upsert            (RAM "lite")
#   POST /memory/semantic/search            (RAM "lite")
#   POST /memory/vector/upsert              (UnifiedMemory / FAISS-CPU)
#   POST /memory/vector/search              (UnifiedMemory / FAISS-CPU)
#   POST /reason/execute                    (herramientas / orquestador)
#   POST /plan/solve                        (Fase 3: TaskPlanner HTN)
#   POST /curriculum/build                  (Fase 3: Curriculum auto-generado)
#
# Auth: Header x-api-key o Authorization: Bearer <key>
#       Actívalo con: API_KEY="mi-clave"
#       Swagger muestra botón "Authorize".
#
# Nota: escala de "confidence" de búsquedas vectoriales vía CONF_SCALE
#       Windows (persistente):  setx CONF_SCALE 300
#       PowerShell (sesión):    $env:CONF_SCALE = "300"

from __future__ import annotations

import os
import math
import statistics
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi import Security

# -----------------------------------------------------------------------------#
#  Razón / Autonomía (Fase 2 + Fase 3)
# -----------------------------------------------------------------------------#
# Memoria unificada (vector) y orquestador de skills
from memory.unified_memory import UnifiedMemory

try:
    # Layout "reasoner/"
    from reasoner.orchestrator import SkillOrchestrator  # type: ignore
except Exception:
    # Fallback si está en raíz
    from orchestrator import SkillOrchestrator  # type: ignore

# Fase 3: Planner & Curriculum (con fallback simple si no existen)
try:
    from reasoner.task_planner import TaskPlanner  # type: ignore
except Exception:
    class TaskPlanner:  # fallback mínimo
        def plan(self, goal: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
            ctx = context or {}
            # Descompone naive el objetivo en 3 pasos genéricos
            return [
                {"id": "1", "action": "entender_objetivo", "input": goal, "deps": []},
                {"id": "2", "action": "buscar_contexto", "input": list(ctx.keys()), "deps": ["1"]},
                {"id": "3", "action": "resolver", "input": goal, "deps": ["2"]},
            ]

try:
    from reasoner.curriculum import CurriculumBuilder  # type: ignore
except Exception:
    class CurriculumBuilder:  # fallback mínimo
        def build(self, failures: List[Dict[str, Any]], max_items: int = 10) -> List[Dict[str, Any]]:
            out = []
            for i, fail in enumerate(failures[:max_items], 1):
                q = fail.get("query") or fail.get("input") or f"tarea_{i}"
                out.append({
                    "id": str(i),
                    "practice": f"Reintenta: {q}",
                    "hint": "Simplifica el problema y valida sub-resultados.",
                    "tags": ["autogen", "from_fail"],
                })
            return out

# -----------------------------------------------------------------------------#
#  App + Seguridad (API Key) y Swagger "Authorize"
# -----------------------------------------------------------------------------#
API_KEY_HEADER_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

def swagger_api_key(api_key: str = Security(api_key_header)) -> str:
    # Dependencia para proteger endpoints en Swagger si la usas.
    return api_key or ""

app = FastAPI(title="AGI Prototype (FAISS-CPU)", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Rutas *libres* sin API Key (para que Swagger cargue)
DOCS_WHITELIST = {"/", "/docs", "/redoc", "/openapi.json", "/status", "/get_status"}

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    required_key = (os.getenv("API_KEY") or "").strip()
    if not required_key:
        return await call_next(request)  # sin API_KEY => libre

    if request.url.path in DOCS_WHITELIST:
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
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description="AGI endpoints (Fases 1, 2 y 3)",
        routes=app.routes,
    )
    schema.setdefault("components", {}).setdefault("securitySchemes", {})["ApiKeyAuth"] = {
        "type": "apiKey",
        "in": "header",
        "name": API_KEY_HEADER_NAME,
        "description": "Introduce tu API Key (solo el valor).",
    }
    schema["security"] = [{"ApiKeyAuth": []}]  # botón Authorize
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi  # type: ignore[assignment]

# -----------------------------------------------------------------------------#
#  Estado, Memorias y Orquestador
# -----------------------------------------------------------------------------#
UM = UnifiedMemory(memory_dir="memory_store", use_gpu=False)

# Orquestador (Fase 2)
def _vector_provider(query: str, k: int) -> List[Dict[str, Any]]:
    """
    Adapta la salida de UnifiedMemory:
      - score: crudo del índice
      - confidence = clamp(score * CONF_SCALE, 0..1)
    """
    scale = float(os.environ.get("CONF_SCALE", "200.0"))
    try:
        res = UM.retrieve_relevant_memories(query, top_k=k)
    except Exception as e:
        return [{"text": f"[vector error] {type(e).__name__}: {e}", "score": 0.0, "confidence": 0.0}]

    out: List[Dict[str, Any]] = []
    for r in res:
        raw = float(r.get("score", 0.0))
        conf = max(0.0, min(1.0, raw * scale))
        meta = r.get("metadata", {})
        if isinstance(meta, list) and len(meta) > 20:
            meta = meta[:20] + ["…"]
        out.append({
            "text": r.get("text", ""),
            "score": raw,
            "confidence": conf,
            "citation_id": r.get("citation_id") or r.get("mem_id"),
            "metadata": meta,
            "result_quality": r.get("result_quality", "unknown"),
            "trace_id": r.get("trace_id"),
        })
    return out

ORCH = SkillOrchestrator(
    memory_vector_provider=_vector_provider,
    fs_root=".",                # para filesystem_read
    score_threshold=0.45,
)

PLANNER = TaskPlanner()
CURRICULUM = CurriculumBuilder()

# -----------------------------------------------------------------------------#
#  Métricas (contadores + latencias)
# -----------------------------------------------------------------------------#
COUNTERS: Dict[str, int] = {
    "total_requests": 0,
    "chat_requests": 0,

    "upserts_total": 0,          # semantic RAM
    "search_requests": 0,

    "vec_upserts_total": 0,      # vector
    "vec_search_requests": 0,

    "reason_requests": 0,        # tools/orchestrator
    "plan_requests": 0,          # Fase 3
    "curriculum_requests": 0,    # Fase 3
}

LAT_LOG: Dict[str, List[float]] = {
    "chat": [], "upsert": [], "search": [],
    "vec_upsert": [], "vec_search": [], "reason": [],
    "plan": [], "curriculum": [],
}

def _record_latency(bucket: str, start_ts: float) -> None:
    LAT_LOG.setdefault(bucket, []).append((perf_counter() - start_ts) * 1000.0)

def _avg(x: List[float]) -> Optional[float]:
    return float(statistics.fmean(x)) if x else None

def _p95(x: List[float]) -> Optional[float]:
    if not x:
        return None
    s = sorted(x)
    idx = max(0, math.ceil(0.95 * len(s)) - 1)
    return float(s[idx])

# -----------------------------------------------------------------------------#
#  Memoria semántica "lite" (RAM) — Fase 1 compat
# -----------------------------------------------------------------------------#
STORE: List[Dict[str, Any]] = []

def _simple_score(q: str, t: str) -> float:
    if not q or not t:
        return 0.0
    ql, tl = q.lower(), t.lower()
    hits = tl.count(ql)
    if hits <= 0:
        return 0.0
    return min(1.0, hits / max(1, len(tl) / max(1, len(ql))))

# -----------------------------------------------------------------------------#
#  Rutas
# -----------------------------------------------------------------------------#
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
                index_size_vec = int(v)
                break

    return {
        "ok": True,
        "counters": {
            **COUNTERS,
            "index_size_semantic": len(STORE),
            "index_size_vector": index_size_vec,
        },
        "latency_ms": {
            "chat_avg": _avg(LAT_LOG["chat"]), "chat_p95": _p95(LAT_LOG["chat"]),
            "upsert_avg": _avg(LAT_LOG["upsert"]), "upsert_p95": _p95(LAT_LOG["upsert"]),
            "search_avg": _avg(LAT_LOG["search"]), "search_p95": _p95(LAT_LOG["search"]),
            "vec_upsert_avg": _avg(LAT_LOG["vec_upsert"]), "vec_upsert_p95": _p95(LAT_LOG["vec_upsert"]),
            "vec_search_avg": _avg(LAT_LOG["vec_search"]), "vec_search_p95": _p95(LAT_LOG["vec_search"]),
            "reason_avg": _avg(LAT_LOG["reason"]), "reason_p95": _p95(LAT_LOG["reason"]),
            "plan_avg": _avg(LAT_LOG["plan"]), "plan_p95": _p95(LAT_LOG["plan"]),
            "curriculum_avg": _avg(LAT_LOG["curriculum"]), "curriculum_p95": _p95(LAT_LOG["curriculum"]),
        },
        "vector_status": um_status,
    }

@app.get("/status")
def status_alias() -> Dict[str, Any]:
    return get_status()

# ------------------------------ Chat -----------------------------------------#
@app.post("/chat")
async def chat(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        COUNTERS["chat_requests"] += 1

        msgs = payload.get("messages", []) or []
        user_msg = next((m.get("content", "") for m in reversed(msgs)
                         if m.get("role") == "user"), "").strip()

        mode = (payload.get("mode") or "echo").lower()
        k = int(payload.get("k", 3))

        if mode == "reason":
            return {"ok": True, "query": user_msg, **ORCH.execute(user_msg, k=k)}

        if mode == "plan":
            plan = PLANNER.plan(user_msg, context=payload.get("context"))
            return {"ok": True, "skill": "task_planner", "goal": user_msg, "plan": plan}

        if mode == "curriculum":
            failures = payload.get("failures") or []
            tasks = CURRICULUM.build(failures, max_items=int(payload.get("max", 10)))
            return {"ok": True, "skill": "curriculum", "items": tasks}

        return {"text": f"AGI: {user_msg} <|end_of_turn|>"}
    finally:
        _record_latency("chat", t0)

# -------------------- Memoria semántica "lite" (RAM) -------------------------#
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
            if not text:
                continue
            mem_id = f.get("mem_id") or f"mem://{uuid4()}"
            STORE.append({
                "mem_id": mem_id,
                "doc_id": f.get("doc_id") or mem_id,
                "text": text,
                "meta": f.get("meta") or {},
                "result_quality": f.get("result_quality", "unknown"),
                "confidence": float(f.get("confidence", 0.5)),
                "trace_id": f.get("trace_id") or req_trace,
            })
            mem_ids.append(mem_id)
            upserted += 1

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

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for f in STORE:
            scored.append((_simple_score(q, f.get("text", "")), f))
        scored.sort(key=lambda x: x[0], reverse=True)

        out = []
        for s, f in scored[:k]:
            row = dict(f)
            row["score"] = float(s)
            row["confidence_source"] = row.get("confidence", 0.5)
            row["confidence"] = float(s)
            row["citation_id"] = f.get("mem_id") or f.get("doc_id")
            out.append(row)

        return {"ok": True, "results": out, "query": q, "k": k}
    finally:
        _record_latency("search", t0)

# ------------------------- Memoria vectorial (UM) -----------------------------#
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
            if not text:
                continue

            meta = f.get("metadata") or f.get("meta") or {}
            res = UM.add_to_vector_memory(
                text,
                metadata=meta,
                result_quality=f.get("result_quality", "unknown"),
                confidence=float(f.get("confidence", 0.5)),
                trace_id=f.get("trace_id") or req_trace,
            )

            mem_id = None
            if isinstance(res, tuple) and len(res) >= 2:
                mem_id = res[1]
            elif isinstance(res, dict):
                mem_id = res.get("mem_id")
            elif isinstance(res, str):
                mem_id = res

            if mem_id:
                mem_ids.append(mem_id)
                upserted += 1

        COUNTERS["vec_upserts_total"] += upserted
        return {"ok": True, "upserted": upserted, "mem_ids": mem_ids}
    finally:
        _record_latency("vec_upsert", t0)

@app.post("/memory/vector/search")
async def vector_search(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        COUNTERS["vec_search_requests"] += 1

        q = (payload.get("q") or "").strip()
        k = int(payload.get("k", 5))

        results = _vector_provider(q, k)
        return {"ok": True, "results": results, "query": q, "k": k}
    finally:
        _record_latency("vec_search", t0)

# --------------------------- Razonamiento (Fase 2) ---------------------------#
@app.post("/reason/execute")
async def reason_execute(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        COUNTERS["reason_requests"] += 1

        q = (payload.get("query") or "").strip()
        k = int(payload.get("k", 5))

        return ORCH.execute(q, k=k)
    finally:
        _record_latency("reason", t0)

# ------------------------------ Fase 3 ---------------------------------------#
@app.post("/plan/solve")
async def plan_solve(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Genera un plan jerárquico (HTN) para un objetivo."""
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        COUNTERS["plan_requests"] += 1

        goal = (payload.get("goal") or "").strip()
        if not goal:
            raise HTTPException(400, "goal requerido")
        context = payload.get("context") or {}
        plan = PLANNER.plan(goal, context=context)
        return {"ok": True, "goal": goal, "plan": plan}
    finally:
        _record_latency("plan", t0)

@app.post("/curriculum/build")
async def curriculum_build(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Construye micro-tareas de práctica a partir de fallos recientes."""
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        COUNTERS["curriculum_requests"] += 1

        failures = payload.get("failures") or []
        max_items = int(payload.get("max", 10))
        items = CURRICULUM.build(failures, max_items=max_items)
        return {"ok": True, "items": items, "count": len(items)}
    finally:
        _record_latency("curriculum", t0)

# -----------------------------------------------------------------------------#
#  Main (solo si lo ejecutas como script local)
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8010"))
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=True)
