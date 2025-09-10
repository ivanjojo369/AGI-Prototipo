# server.py — Servidor FastAPI “lite” sin GPU
# Fase 1 + cierre fino:
#   - Etiquetado de recuerdos: result_quality, confidence, trace_id
#   - mem_id estable por recuerdo y alias citation_id en /search
#   - get_status(): contadores y latencias avg/p95
#   - Búsqueda “CPU” simple con score normalizado

from __future__ import annotations
from fastapi import FastAPI, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List, Optional, Tuple
from time import perf_counter
from uuid import uuid4
import math
import statistics

# ---------------------------
# App & CORS
# ---------------------------
app = FastAPI(title="AGI Local Demo (FAISS-CPU compatible)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------------------------
# Memoria “lite” en RAM
# Estructura de cada item:
# {
#   "mem_id": "mem://<uuid>",     # id estable para auditoría (“cita”)
#   "doc_id": str,                # alias (para compat)
#   "text": str,
#   "meta": {...},                # opcional
#   "result_quality": str,        # "good" | "ok" | "bad" | "unknown"
#   "confidence": float,          # [0,1] (almacenada)
#   "trace_id": str
# }
# ---------------------------
STORE: List[Dict[str, Any]] = []

# ---------------------------
# Métricas de servidor (contadores y latencias)
# ---------------------------
COUNTERS: Dict[str, int] = {
    "total_requests": 0,
    "chat_requests": 0,
    "upserts_total": 0,
    "search_requests": 0,
}

# latencias por ruta (en ms)
LAT_LOG: Dict[str, List[float]] = {
    "chat": [],
    "upsert": [],
    "search": [],
}

def _record_latency(bucket: str, start: float) -> None:
    dur_ms = (perf_counter() - start) * 1000.0
    LAT_LOG.setdefault(bucket, []).append(dur_ms)

def _avg(lat_list: List[float]) -> Optional[float]:
    return float(statistics.fmean(lat_list)) if lat_list else None

def _p95(lat_list: List[float]) -> Optional[float]:
    if not lat_list:
        return None
    s = sorted(lat_list)
    k = max(0, math.ceil(0.95 * len(s)) - 1)
    return float(s[k])

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
def health() -> Dict[str, Any]:
    return {"ok": True}

@app.get("/get_status")
def get_status() -> Dict[str, Any]:
    """Exponer estado con contadores e índice."""
    index_size = len(STORE)
    return {
        "ok": True,
        "counters": {
            **COUNTERS,
            "index_size": index_size,
        },
        "latency_ms": {
            "chat_avg": _avg(LAT_LOG["chat"]),
            "chat_p95": _p95(LAT_LOG["chat"]),
            "upsert_avg": _avg(LAT_LOG["upsert"]),
            "upsert_p95": _p95(LAT_LOG["upsert"]),
            "search_avg": _avg(LAT_LOG["search"]),
            "search_p95": _p95(LAT_LOG["search"]),
        },
    }

@app.post("/chat")
async def chat(request: Request, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        COUNTERS["chat_requests"] += 1

        msgs = payload.get("messages", [])
        user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        # Placeholder (reemplaza por tu backend LLM)
        return {"text": f"AGI: {user} <|end_of_turn|>"}
    finally:
        _record_latency("chat", t0)

@app.post("/memory/semantic/upsert")
async def mem_upsert(
    request: Request,
    payload: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    """
    Inserta/actualiza hechos en la memoria.
    Etiquetado automático:
      - result_quality: default "unknown"
      - confidence: default 0.5
      - trace_id: del payload o generado
      - mem_id: generado si no viene
      - doc_id: alias (usa mem_id por defecto)
    """
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        facts = payload.get("facts", []) or []
        req_trace = payload.get("trace_id") or str(uuid4())

        upserted = 0
        created_ids: List[str] = []
        for f in facts:
            text = (f or {}).get("text")
            if not text:
                continue

            mem_id = f.get("mem_id") or f"mem://{uuid4()}"
            item = {
                "mem_id": mem_id,
                "doc_id": f.get("doc_id") or mem_id,  # alias de compatibilidad
                "text": text,
                "meta": f.get("meta") or {},
                "result_quality": f.get("result_quality", "unknown"),
                "confidence": float(f.get("confidence", 0.5)),
                "trace_id": f.get("trace_id") or req_trace,
            }
            STORE.append(item)
            created_ids.append(mem_id)
            upserted += 1

        COUNTERS["upserts_total"] += upserted
        return {"ok": True, "upserted": upserted, "trace_id": req_trace, "mem_ids": created_ids}
    finally:
        _record_latency("upsert", t0)

def _simple_score(q: str, t: str) -> float:
    """Puntaje muy simple (CPU): frecuencia de término / longitud.
    Devuelve [0,1] aprox. para normalizar como 'confidence' de búsqueda."""
    if not q:
        return 0.0
    ql = q.lower()
    tl = t.lower()
    hits = tl.count(ql)
    if hits <= 0:
        return 0.0
    # normalización básica por longitud para no sesgar a textos largos
    return min(1.0, hits / max(1, len(tl) / max(1, len(ql))))

@app.post("/memory/semantic/search")
async def mem_search(
    request: Request,
    payload: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    """
    Búsqueda CPU con score normalizado.
    Devuelve los recuerdos con sus etiquetas (result_quality, confidence_source, trace_id)
    y las **citas**: mem_id + alias citation_id.
    """
    t0 = perf_counter()
    try:
        COUNTERS["total_requests"] += 1
        COUNTERS["search_requests"] += 1

        q = (payload.get("q") or "").strip()
        k = int(payload.get("k", 5))

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for f in STORE:
            s = _simple_score(q, f.get("text", ""))
            scored.append((s, f))

        # Ordenar por score desc
        scored.sort(key=lambda x: x[0], reverse=True)
        topk = []
        for s, f in scored[:k]:
            out = dict(f)
            out["score"] = float(s)
            # 'confidence' aquí refleja la confianza de la búsqueda;
            # mantenemos 'confidence' original en 'confidence_source'
            out["confidence_source"] = out.get("confidence", 0.5)
            out["confidence"] = float(s)
            # “cita” para auditoría
            out["citation_id"] = f.get("mem_id") or f.get("doc_id")
            topk.append(out)

        return {"ok": True, "results": topk, "query": q, "k": k}
    finally:
        _record_latency("search", t0)
