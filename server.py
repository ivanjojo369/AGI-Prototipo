# server.py — AGI Prototype Fase 6: jobs persistence/rotation/metrics integradas
from __future__ import annotations

import os
import statistics
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from server_jobs_router import jobs_router  # incluye endpoints Fase 6

# ------------------------ Fallbacks seguros (por si faltan módulos) ----------#
try:
    from memory.unified_memory import UnifiedMemory  # type: ignore
except Exception:  # pragma: no cover
    class UnifiedMemory:  # minimal stub
        def __init__(self, memory_dir: str = "memory_store", use_gpu: bool = False) -> None:
            pass
        def retrieve_relevant_memories(self, query: str, top_k: int = 5):
            return []
        def add_to_vector_memory(self, text: str, metadata=None, result_quality="unknown", confidence=0.5, trace_id=None):
            return {"mem_id": f"mem://{uuid4()}"}
        def get_status(self):
            return {"ok": True, "index_size": 0}

try:
    from reasoner.orchestrator import SkillOrchestrator  # type: ignore
except Exception:  # pragma: no cover
    class SkillOrchestrator:  # minimal stub
        def __init__(self, **kwargs) -> None:
            pass
        def execute(self, query: str, k: int = 5):
            return {"status": "ok", "answer": f"echo:{query}", "k": k}

try:
    from planner.task_planner import (
        Plan as HTNPlan,
        Step as HTNStep,
        TaskPlanner,
        BuiltinActions,
    )  # type: ignore
except Exception:  # pragma: no cover
    # Stubs mínimos para no romper import
    class HTNStep:  # type: ignore
        def __init__(self, id: str, kind: str, name: str, inputs=None, postconditions=None) -> None:
            self.id, self.kind, self.name = id, kind, name
            self.inputs = inputs or {}
            self.postconditions = postconditions or []
    class HTNPlan:  # type: ignore
        def __init__(self, goal: str, steps: List[HTNStep], metadata=None) -> None:
            self.goal, self.steps, self.metadata = goal, steps, (metadata or {})
        def model_dump(self):  # emula pydantic
            return {"goal": self.goal, "steps": [vars(s) for s in self.steps], "metadata": self.metadata}
    class BuiltinActions:  # type: ignore
        @staticmethod
        def python_exec(**kw): return {"ok": True}
        @staticmethod
        def filesystem_read(**kw): return {"ok": True, "result": ""}
        @staticmethod
        def search_web(**kw): return {"ok": True, "hits": []}
    class TaskPlanner:  # type: ignore
        def __init__(self, actions=None, curriculum_path=""): pass
        def execute_plan(self, plan, context=None, job_id=None):
            return {"status": "ok", "goal": plan.goal, "results": {}, "errors": [], "curriculum_entries": 0, "confidence": 1.0}

# Persistencia: métricas de Fase 6
from jobs_storage import get_persistence_metrics

# -----------------------------------------------------------------------------
# App & Seguridad (API Key)
# -----------------------------------------------------------------------------
API_KEY_HEADER_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

app = FastAPI(title="AGI Prototype (FASE 6)", version="6.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(jobs_router)

# Whitelist para docs/health sin API key
DOCS_WHITELIST = {"/", "/docs", "/redoc", "/openapi.json", "/status", "/get_status"}

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    # Bypass controlado en tests/CI
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("ALLOW_TEST_NO_AUTH") == "1":
        return await call_next(request)

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
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description="AGI endpoints (Fases 1–6, incluye /jobs persistence/rotation/metrics)",
        routes=app.routes,
    )
    schema.setdefault("components", {}).setdefault("securitySchemes", {})["ApiKeyAuth"] = {
        "type": "apiKey",
        "in": "header",
        "name": API_KEY_HEADER_NAME,
        "description": "Introduce tu API Key.",
    }
    schema["security"] = [{"ApiKeyAuth": []}]
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# Estado / Utilidades
# -----------------------------------------------------------------------------
# Memoria vectorial (con fallback)
UM = UnifiedMemory(memory_dir="memory_store", use_gpu=False)

def _vector_provider(query: str, k: int) -> List[Dict[str, Any]]:
    scale = float(os.environ.get("CONF_SCALE", "200.0"))
    try:
        res = UM.retrieve_relevant_memories(query, top_k=k)
    except Exception as e:  # pragma: no cover
        return [{"text": f"[vector error] {type(e).__name__}: {e}", "score": 0.0, "confidence": 0.0}]
    out: List[Dict[str, Any]] = []
    for r in res:
        raw = float(r.get("score", 0.0))
        conf = max(0.0, min(1.0, raw * scale))
        meta = r.get("metadata", {})
        if isinstance(meta, list) and len(meta) > 20:
            meta = meta[:20] + ["…"]
        out.append(
            {
                "text": r.get("text", ""),
                "score": raw,
                "confidence": conf,
                "citation_id": r.get("citation_id") or r.get("mem_id"),
                "metadata": meta,
                "result_quality": r.get("result_quality", "unknown"),
                "trace_id": r.get("trace_id"),
            }
        )
    return out

ORCH = SkillOrchestrator(
    memory_vector_provider=_vector_provider,
    fs_root=".",
    score_threshold=0.45,
)

def memory_action(**kwargs):
    q = (kwargs.get("query") or "").strip()
    k = int(kwargs.get("k", 5))
    return _vector_provider(q, k)

actions = {
    "python_exec": BuiltinActions.python_exec,
    "filesystem_read": BuiltinActions.filesystem_read,
    "memory_search": memory_action,
    "memory_vector": memory_action,
    "search_web": BuiltinActions.search_web,
}

try:
    PLANNER = TaskPlanner(actions=actions, curriculum_path="data/curriculum/planner_curriculum.jsonl")  # type: ignore
except Exception:  # pragma: no cover
    PLANNER = TaskPlanner(actions=actions)  # type: ignore

try:
    from reasoner.curriculum import CurriculumBuilder  # type: ignore
except Exception:  # pragma: no cover
    class CurriculumBuilder:  # fallback mínimo
        def build(self, failures: List[Dict[str, Any]], max_items: int = 10) -> List[Dict[str, Any]]:
            out = []
            for i, fail in enumerate(failures[:max_items], 1):
                q = fail.get("query") or fail.get("input") or f"tarea_{i}"
                out.append({"id": str(i), "practice": f"Reintenta: {q}", "hint": "Divide y valida.", "tags": ["autogen"]})
            return out

CURRICULUM = CurriculumBuilder()

COUNTERS: Dict[str, int] = {
    k: 0
    for k in [
        "total_requests",
        "chat_requests",
        "upserts_total",
        "search_requests",
        "vec_upserts_total",
        "vec_search_requests",
        "reason_requests",
        "plan_requests",
        "curriculum_requests",
    ]
}
LAT_LOG: Dict[str, List[float]] = {
    k: []
    for k in ["chat", "upsert", "search", "vec_upsert", "vec_search", "reason", "plan", "curriculum"]
}

def _record_latency(bucket: str, t0: float) -> None:
    LAT_LOG.setdefault(bucket, []).append((perf_counter() - t0) * 1000.0)

def _avg(x: List[float]) -> Optional[float]:
    return float(statistics.fmean(x)) if x else None

def _p95(x: List[float]) -> Optional[float]:
    if not x:
        return None
    s = sorted(x)
    from math import ceil
    return float(s[max(0, ceil(0.95 * len(s)) - 1)])

# Memoria semántica simple (RAM)
STORE: List[Dict[str, Any]] = []

def _simple_score(q: str, t: str) -> float:
    if not q or not t:
        return 0.0
    ql, tl = q.lower(), t.lower()
    hits = tl.count(ql)
    if hits <= 0:
        return 0.0
    return min(1.0, hits / max(1, len(tl) / max(1, len(ql))))


# -----------------------------------------------------------------------------
# Endpoints básicos
# -----------------------------------------------------------------------------
@app.get("/")
def health() -> Dict[str, Any]:
    return {"ok": True}

@app.get("/status")
def status_alias() -> Dict[str, Any]:
    return get_status()

@app.get("/get_status")
def get_status() -> Dict[str, Any]:
    try:
        um_status = UM.get_status() if hasattr(UM, "get_status") else {}
    except Exception as e:  # pragma: no cover
        um_status = {"ok": False, "error": type(e).__name__, "msg": str(e)}
    index_size_vec = 0
    if isinstance(um_status, dict):
        for k in ("index_size", "ntotal", "size", "count"):
            v = um_status.get(k)
            if isinstance(v, (int, float)):
                index_size_vec = int(v)
                break

    # Métricas de persistencia (Fase 6)
    persist = get_persistence_metrics()

    return {
        "ok": True,
        "counters": {**COUNTERS, "index_size_semantic": len(STORE), "index_size_vector": index_size_vec},
        "latency_ms": {
            "chat_avg": _avg(LAT_LOG["chat"]),
            "chat_p95": _p95(LAT_LOG["chat"]),
            "upsert_avg": _avg(LAT_LOG["upsert"]),
            "upsert_p95": _p95(LAT_LOG["upsert"]),
            "search_avg": _avg(LAT_LOG["search"]),
            "search_p95": _p95(LAT_LOG["search"]),
            "vec_upsert_avg": _avg(LAT_LOG["vec_upsert"]),
            "vec_upsert_p95": _p95(LAT_LOG["vec_upsert"]),
            "vec_search_avg": _avg(LAT_LOG["vec_search"]),
            "vec_search_p95": _p95(LAT_LOG["vec_search"]),
            "reason_avg": _avg(LAT_LOG["reason"]),
            "reason_p95": _p95(LAT_LOG["reason"]),
            "plan_avg": _avg(LAT_LOG["plan"]),
            "plan_p95": _p95(LAT_LOG["plan"]),
            "curriculum_avg": _avg(LAT_LOG["curriculum"]),
            "curriculum_p95": _p95(LAT_LOG["curriculum"]),
        },
        # Fase 6: claves nuevas
        **persist,
        "vector_status": um_status,
    }

@app.get("/auth/echo", dependencies=[Security(api_key_header)])
def auth_echo() -> Dict[str, Any]:
    return {"ok": True, "auth": "passed"}


# ------------------------------ Chat -----------------------------------------#
@app.post("/chat")
async def chat(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        COUNTERS["chat_requests"] += 1
        msgs = payload.get("messages", []) or []
        user_msg = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "").strip()
        mode = (payload.get("mode") or "echo").lower()
        k = int(payload.get("k", 3))

        if mode == "reason":
            return {"ok": True, "query": user_msg, **ORCH.execute(user_msg, k=k)}

        if mode == "plan":
            plan = _solve_goal_to_plan(user_msg, payload.get("context") or {})
            return {"ok": True, "skill": "task_planner", "goal": user_msg, "plan": plan.model_dump()}

        if mode == "curriculum":
            failures = payload.get("failures") or []
            return {"ok": True, "skill": "curriculum", "items": CURRICULUM.build(failures, max_items=int(payload.get("max", 10)))}

        return {"text": f"AGI: {user_msg} <|end_of_turn|>"}
    finally:
        _record_latency("chat", t0)


# --------------- Memoria semántica (RAM) -------------------------------------#
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
            STORE.append(
                {
                    "mem_id": mem_id,
                    "doc_id": f.get("doc_id") or mem_id,
                    "text": text,
                    "meta": f.get("meta") or {},
                    "result_quality": f.get("result_quality", "unknown"),
                    "confidence": float(f.get("confidence", 0.5)),
                    "trace_id": f.get("trace_id") or req_trace,
                }
            )
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
        scored: List[Tuple[float, Dict[str, Any]]] = [(_simple_score(q, f.get("text", "")), f) for f in STORE]
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
            mem_id = (
                res[1]
                if isinstance(res, tuple) and len(res) >= 2
                else (res.get("mem_id") if isinstance(res, dict) else (res if isinstance(res, str) else None))
            )
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
        return {"ok": True, "results": _vector_provider(q, k), "query": q, "k": k}
    finally:
        _record_latency("vec_search", t0)


# ------------------------------ Planner (HTN) --------------------------------#
from pydantic import BaseModel, Field

class PlanExecuteRequest(BaseModel):
    plan: HTNPlan = Field(...)
    context: Dict[str, Any] = Field(default_factory=dict)
    job_id: Optional[str] = None

class PlanExecuteResponse(BaseModel):
    status: str
    goal: str
    results: Dict[str, Any]
    errors: list
    curriculum_entries: int
    confidence: float


def _normalize_goal_text(goal: Any) -> str:
    if isinstance(goal, list):
        s = " ".join(str(x) for x in goal)
    elif isinstance(goal, str):
        s = goal
    else:
        s = str(goal)
    return s


def _solve_goal_to_plan(goal: Any, context: Dict[str, Any]) -> HTNPlan:
    goal_str = _normalize_goal_text(goal)
    gl = goal_str.lower()
    if "buscar" in gl:
        return HTNPlan(
            goal=goal_str,
            steps=[
                HTNStep(
                    id="T",
                    kind="task",
                    name="buscar_y_leer",
                    inputs={"query": context.get("query", goal_str), "path": context.get("path", "server.py")},
                )
            ],
            metadata={"auto": True},
        )
    if "leer" in gl or "read" in gl:
        return HTNPlan(
            goal=goal_str,
            steps=[
                HTNStep(
                    id="r1",
                    kind="action",
                    name="filesystem_read",
                    inputs={"path": context.get("path", "server.py")},
                    postconditions=["len(result) > 0"],
                )
            ],
            metadata={"auto": True},
        )
    # Default
    return HTNPlan(
        goal=goal_str,
        steps=[
            HTNStep(id="m1", kind="action", name="memory_search", inputs={"query": goal_str, "k": 3}),
            HTNStep(id="s1", kind="action", name="python_exec", inputs={"code": "result = 40+2"}, postconditions=["result == 42"]),
            HTNStep(
                id="r1",
                kind="action",
                name="filesystem_read",
                inputs={"path": context.get("path", "server.py")},
                postconditions=["len(result) > 0"],
            ),
        ],
        metadata={"auto": True},
    )


@app.post("/plan/solve")
async def plan_solve(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        COUNTERS["plan_requests"] += 1
        goal = payload.get("goal")
        if goal is None or (isinstance(goal, str) and not goal.strip()):
            raise HTTPException(400, "goal requerido")
        plan = _solve_goal_to_plan(goal, payload.get("context") or {})
        return {"ok": True, "goal": _normalize_goal_text(goal), "plan": plan.model_dump()}
    finally:
        _record_latency("plan", t0)


@app.post("/plan/execute", response_model=PlanExecuteResponse, tags=["planner"])
async def plan_execute(req: PlanExecuteRequest) -> Dict[str, Any]:
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        COUNTERS["plan_requests"] += 1
        return PLANNER.execute_plan(req.plan, context=req.context, job_id=req.job_id)
    finally:
        _record_latency("plan", t0)


@app.post("/curriculum/build")
async def curriculum_build(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8010"))
    ak = (os.getenv("API_KEY") or "")
    mask = ("*" * (len(ak) - 4) + ak[-4:]) if ak else "(none)"
    print(f"[server] API_KEY={mask}  http://127.0.0.1:{port}")
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=True)
