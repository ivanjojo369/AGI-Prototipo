# agi_webapp.py
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ---- Inicialización de componentes (modelo/memoria/reflexión) ----
from pathlib import Path

try:
    from agi_initializer import build_context
except Exception as e:
    raise RuntimeError(
        "No puedo importar agi_initializer.build_context. Asegúrate de que agi_initializer.py existe."
    ) from e


def _pick(ctx, *names, default=None):
    """Devuelve el primer atributo o clave existente en ctx con los nombres dados."""
    for name in names:
        # objeto con atributo
        if hasattr(ctx, name):
            val = getattr(ctx, name)
            if val is not None:
                return val
        # dict con clave
        if isinstance(ctx, dict) and name in ctx and ctx[name] is not None:
            return ctx[name]
    return default


def _normalize_components(ctx):
    """Devuelve un dict canónico con las piezas que necesitamos."""
    return {
        # algunos inicializadores llaman "adapter" o "llm" al modelo
        "model": _pick(ctx, "model", "adapter", "llm"),
        # a veces queda expuesto como interface_instance
        "interface": _pick(ctx, "interface", "interface_instance"),
        "memory": _pick(ctx, "memory", "memory_engine", "mem"),
        "reflection": _pick(ctx, "reflection", "reflection_instance"),
        "config": _pick(ctx, "config", "cfg"),
        "web": _pick(ctx, "web", "web_config"),
    }


# 1) Construir el contexto
CTX = build_context(Path("settings.json"))

# 2) Normalizar a un dict uniforme
COMP = _normalize_components(CTX)

# 3) Variables de módulo usadas por los endpoints
MODEL = COMP["model"]
INTERFACE = COMP["interface"]
MEMORY = COMP["memory"]
REFLECTION = COMP["reflection"]
CFG = COMP["config"]
WEB = COMP["web"]

# ------------------------------------------------------------------
# Memoria episódica (fallback simple)
# ------------------------------------------------------------------
@dataclass
class Episode:
    id: int
    ts: float
    text: str
    meta: Dict[str, Any]


class EpisodeMemoryLite:
    def __init__(self):
        self._data: List[Episode] = []
        self._idx = 0

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None) -> int:
        self._idx += 1
        self._data.append(Episode(id=self._idx, ts=time.time(), text=text, meta=meta or {}))
        return self._idx

    def recent(self, k: int = 20) -> List[Dict[str, Any]]:
        rows = self._data[-k:][::-1]
        return [asdict(x) for x in rows]

    def search(self, q: str, k: int = 5) -> List[Dict[str, Any]]:
        ql = q.lower().strip()
        scored = []
        for ep in self._data:
            score = ep.text.lower().count(ql) if ql else 0
            if score > 0:
                scored.append((score, ep))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [asdict(ep) | {"score": sc} for sc, ep in scored[:k]]


# ------------------------------------------------------------------
# Memoria semántica “lite” (bolsa de palabras + coseno)
# ------------------------------------------------------------------
@dataclass
class SemItem:
    id: int
    text: str
    meta: Dict[str, Any]
    vec: Dict[str, float]


class SemanticMemoryLite:
    def __init__(self):
        self._data: List[SemItem] = []
        self._idx = 0

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in s).split() if t]

    @classmethod
    def _to_vec(cls, s: str) -> Dict[str, float]:
        v: Dict[str, float] = {}
        for tok in cls._tokenize(s):
            v[tok] = v.get(tok, 0.0) + 1.0
        # normaliza L2
        norm = sum(x * x for x in v.values()) ** 0.5 or 1.0
        return {k: v[k] / norm for k in v}

    @staticmethod
    def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        # producto punto sobre intersección
        keys = a.keys() & b.keys()
        return float(sum(a[k] * b[k] for k in keys))

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None) -> int:
        self._idx += 1
        self._data.append(SemItem(self._idx, text, meta or {}, self._to_vec(text)))
        return self._idx

    def search(self, q: str, k: int = 5) -> List[Dict[str, Any]]:
        vq = self._to_vec(q)
        scored = []
        for it in self._data:
            sc = self._cos(vq, it.vec)
            scored.append((sc, it))
        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for sc, it in scored[:k]:
            out.append({"id": it.id, "score": sc, "text": it.text, "meta": it.meta})
        return out


# ------------------------------------------------------------------
# Modelos pydantic para /chat y /semantic
# ------------------------------------------------------------------
class ChatIn(BaseModel):
    message: str
    system: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class ChatOut(BaseModel):
    reply: str


class SemAddIn(BaseModel):
    text: str
    meta: Optional[Dict[str, Any]] = None


class SemAddOut(BaseModel):
    ok: bool
    id: int


# ------------------------------------------------------------------
# Helpers de compatibilidad con build_context()
# ------------------------------------------------------------------
from dataclasses import asdict, is_dataclass

def _get(ctx: Any, *keys: str, default: Any = None) -> Any:
    """
    Obtiene ctx[key] si ctx es dict; o getattr(ctx, key) si ctx es objeto.
    Devuelve el primer valor no None; si ninguno existe, devuelve 'default'.
    """
    for k in keys:
        try:
            if isinstance(ctx, dict):
                val = ctx.get(k, None)
            else:
                # objeto (p.ej. AGIContext)
                val = getattr(ctx, k, None)
        except Exception:
            val = None
        if val is not None:
            return val
    return default

def _to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convierte objetos de config en dict para uso cómodo.
    Soporta dataclasses, pydantic (model_dump/dict) y objetos regulares.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    # pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # pydantic v1
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()
        except Exception:
            pass
    # objeto normal
    if hasattr(obj, "__dict__"):
        try:
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        except Exception:
            pass
    # último recurso
    return {"value": obj}


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
LOG = logging.getLogger("agi_webapp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

CTX = build_context(Path("settings.json"))
CFG = _to_dict(_get(CTX, "config", "cfg", default={}))

# Model (preferimos 'llm' > 'model' > 'engine')
LLM = _get(CTX, "llm", "model", "engine", default=None)

# ---------------- Fallback: carga modelo con llama_cpp si no vino en el contexto ----------------
def _cfg_lookup(cfg: dict, *paths: str, default=None):
    """
    Lee cfg con rutas tipo "model.path". Prueba múltiples rutas,
    devuelve default si no encuentra.
    """
    for path in paths:
        cur = cfg
        ok = True
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok:
            return cur
    return default

if LLM is None:
    try:
        from adapters.llama_cpp_adapter import LlamaCppChat, LlamaCppConfig

        model_path = _cfg_lookup(
            CFG,
            "model.path", "llama.model_path", "llm.path", "models.default_path",
            default="models/openchat-3.5-1210.Q4_K_M.gguf",
        )
        n_ctx = int(_cfg_lookup(CFG, "model.n_ctx", "llama.n_ctx", default=8192))
        n_gpu_layers = int(_cfg_lookup(CFG, "model.n_gpu_layers", "llama.n_gpu_layers", default=-1))
        temperature = float(_cfg_lookup(CFG, "model.temperature", default=0.7))
        top_p = float(_cfg_lookup(CFG, "model.top_p", default=0.95))
        max_tokens = int(_cfg_lookup(CFG, "model.max_tokens", default=256))
        chat_template = _cfg_lookup(CFG, "model.chat_template", default=None)

        cfg = LlamaCppConfig(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            chat_template=chat_template,
        )
        LLM = LlamaCppChat(cfg)
        LOG.info(f"Modelo cargado por fallback: {model_path}")
    except Exception as e:  # pragma: no cover
        LOG.exception("No pude cargar modelo por fallback; seguiré sin LLM (modo eco).")
        LLM = None

# -----------------------------------------------------------------------------------------------
# Episode memory
EP_MEM = _get(CTX, "episode_memory", "memory", default=None)
if EP_MEM is None or not hasattr(EP_MEM, "add"):
    EP_MEM = EpisodeMemoryLite()
    LOG.info("Memory episódica no encontrada en contexto. Usando EpisodeMemoryLite.")

# Semantic memory
SEM_MEM = _get(CTX, "semantic_memory", "vector_memory", default=None)
if SEM_MEM is None or not hasattr(SEM_MEM, "add"):
    SEM_MEM = SemanticMemoryLite()
    LOG.info("Memoria semántica no encontrada en contexto. Usando SemanticMemoryLite.")

# Reflection (opcional)
REFL = _get(CTX, "reflection", "reflection_engine", default=None)

# Web config
web_cfg = CFG.get("web", {})
HOST = web_cfg.get("host", "127.0.0.1")
PORT = int(web_cfg.get("port", 8010))

LOG.info(
    f"Contexto cargado. Modelo: {'llama_cpp' if LLM else 'None'} | Memoria: {'ok' if EP_MEM else 'none'} | Web: {HOST}:{PORT}"
)

app = FastAPI(title="AGI Unificado v2 — API")


# ------------------------------ Endpoints ------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    html = """<h1>AGI Unificado v2 — API</h1>
<p>Servidor listo. Endpoints útiles:</p>
<ul>
<li><code>/health</code> — Health</li>
<li><code>/status</code> — Status</li>
<li><code>/chat</code> — Chat Get Hint</li>
<li><b>POST</b> <code>/chat</code> — Chat (body: {"message":"hola"})</li>
<li><code>/memory/episodes</code> — Episodios recientes</li>
<li><code>/memory/episodes/search?q=algo&k=5</code> — Buscar episodios</li>
<li><b>POST</b> <code>/memory/semantic/add</code> — Agregar semántica (text/meta)</li>
<li><code>/memory/semantic/search?q=algo&k=5</code> — Buscar semántica</li>
<li><code>/routes</code> — Lista de rutas</li>
</ul>
<p>Si ves <code>{"detail":"Not Found"}</code>, revisa la ruta o usa el método correcto (GET vs POST).</p>
"""
    return HTMLResponse(html)


@app.get("/routes")
def routes():
    return [{"path": r.path, "name": r.name, "methods": list(r.methods)} for r in app.router.routes]


@app.get("/health")
def health():
    return {"ok": True, "model": bool(LLM)}


@app.get("/status")
def status():
    return {
        "ok": True,
        "config": CFG,
        "has_model": bool(LLM),
        "memory_counts": {
            "episodes": len(getattr(EP_MEM, "_data", [])),
            "semantic": len(getattr(SEM_MEM, "_data", [])),
        },
    }


@app.get("/chat")
def chat_hint():
    return {"hint": "POST /chat con JSON {'message': 'hola'}"}


@app.post("/chat", response_model=ChatOut)
def chat_api(body: ChatIn):
    if not body.message or not isinstance(body.message, str):
        return JSONResponse({"detail": "message (string) es requerido"}, status_code=422)

    if LLM is None:
        # Sin modelo: responde fijo (útil para diagnóstico)
        EP_MEM.add(f"[USER] {body.message}", {"type": "chat"})
        return {"reply": "Modelo no cargado. Verifica configuración."}

    system_msg = body.system or "Responde en español, de forma breve y útil. No inventes identidades."
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": body.message}]
    gen_kwargs = {
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
        "top_p": body.top_p,
    }

    text, info = LLM.chat(messages, **gen_kwargs)
    EP_MEM.add(f"[USER] {body.message}\n[ASSISTANT] {text}", {"type": "chat", "gen": info})
    return {"reply": text}


@app.get("/memory/episodes")
def memory_recent(k: int = Query(20, ge=1, le=200)):
    return EP_MEM.recent(k=k)


@app.get("/memory/episodes/search")
def memory_search(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=100)):
    return EP_MEM.search(q=q, k=k)


@app.post("/memory/semantic/add", response_model=SemAddOut)
def memory_sem_add(body: SemAddIn):
    _id = SEM_MEM.add(text=body.text, meta=body.meta or {})
    return {"ok": True, "id": _id}


@app.get("/memory/semantic/search")
def memory_sem_search(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=100)):
    return SEM_MEM.search(q=q, k=k)


# ------------------------------ Main ------------------------------
if __name__ == "__main__":
    import uvicorn

    LOG.info("Levantando servidor FastAPI…")
    uvicorn.run("agi_webapp:app", host=HOST, port=PORT, reload=False)
