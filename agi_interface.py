# agi_interface.py
# Interfaz AGI alineada con initializer + webapp.
from __future__ import annotations
from typing import Any, Dict, Optional
import re

# ---- Stopwords/guards típicos para modelos instruct (llama.cpp / OpenChat)
STOP_DEFAULT = [
    "<|end_of_turn|>", "</s>",
    "Usuario:", "User:", "\nUsuario:", "\nUser:",
    "MENSAJE DEL USUARIO", "MENSAJE DEL ASISTENTE",
    "[RESPONDER]", "RESPONDER:", "Instrucciones:", "INSTRUCCIONES:"
]

def _dedupe_lines(text: str) -> str:
    if not text: return text
    out = []
    for ln in text.splitlines():
        if not out or ln.strip() != out[-1].strip():
            out.append(ln.rstrip())
    return "\n".join(out).strip()

def _remove_user_echo(text: str, user_text: str) -> str:
    if not text or not user_text: return text or ""
    t = text.replace(user_text.strip(), "")
    ut = re.escape(user_text.strip().rstrip("?.!"))
    t = re.sub(rf"\b{ut}\b[?.!…]*", "", t, flags=re.IGNORECASE)
    return t.strip()

def _strip_markers(text: str) -> str:
    if not text: return ""
    SYS = [
        "[SISTEMA]", "[SYSTEM]", "[CONTEXTO]", "[CONTEXT]", "[USUARIO]", "[USER]",
        "[ASISTENTE]", "[ASSISTANT]", "[RESPONDER]", "[RESPONSE]",
        "<think>", "</think>", "[INSTRUCCIONES]", "INSTRUCCIONES", "INSTRUCTIONS",
        "MENSAJE DEL USUARIO", "MENSAJE DEL ASISTENTE"
    ]
    t = text
    for m in SYS: t = t.replace(m, "")
    # quita líneas de “no imprimas / no cites…”
    lines = []
    for ln in t.splitlines():
        low = ln.lower().strip()
        if low.startswith("- no ") or "no imprimas" in low or "no cites" in low:
            continue
        if low in ("[responder]", "[response]"):
            continue
        lines.append(ln)
    t = "\n".join(lines)
    t = re.sub(r"(?:^|\n)[\-\*\•]\s*$", "", t.strip())     # bullets huérfanos
    t = re.sub(r"(?:\n\s*){3,}", "\n\n", t)                # colapsa saltos extra
    return t.strip()

def _sanitize(text: str, user_text: str = "") -> str:
    t = (text or "").strip()
    t = _strip_markers(t)
    t = _remove_user_echo(t, user_text)
    t = _dedupe_lines(t)
    if len(t) > 2000:
        t = t[:2000].rstrip() + "…"
    return t

def _max_tokens_by_input(user_text: str) -> int:
    n = len(user_text or "")
    if n <= 40:  return 96
    if n <= 160: return 192
    return 256

# ---------------- Stubs de respaldo (por si el initializer no inyecta algo) ----------------
class _DummyModel:
    def generate(self, prompt: str, temperature: float = 0.6, max_tokens: int = 256, **kw) -> str:
        return "Estoy listo y funcionando. ¿En qué te ayudo?"

class _DummyMemory:
    def __init__(self, persist_path: Optional[str] = None, save_to_disk: bool = True, max_history: int = 64):
        self._hist = []
        self.max_history = max_history
    def add_history(self, role: str, content: str):
        self._hist.append(type("Msg", (), {"role": role, "content": content}))
        self._hist = self._hist[-self.max_history:]
    def recall_recent(self, k: int = 6):
        return self._hist[-k:]
    def summary_text(self) -> str:
        return ""
    def get_slot(self, key: str): return None
    def update_slot(self, key: str, val: Any): pass
    def add_preference(self, p: str): pass
    def get_preferences(self): return []
    def persist(self): pass

class _DummyPlanner:
    def detect(self, user_text: str, memory: Any = None) -> Dict[str, Any]:
        # Heurísticas mínimas
        txt = (user_text or "").strip().lower()
        if txt.startswith(("guarda:", "guardar:", "save:")):
            return {"intent": "save_note", "next": "memory.semantic.add", "slots": {"text": user_text.split(":",1)[1].strip()}}
        if txt.startswith(("recuerda", "qué hablamos")):
            return {"intent": "recall", "next": "memory.report", "slots": {}}
        return {"intent": "chat", "next": "llm.reply", "slots": {}}

# ---------------- AGIInterface principal ----------------
class AGIInterface:
    """
    Interfaz AGI flexible:
    - Soporta __init__ sin argumentos (modo simple).
    - Soporta inyección de model/memory/planner/reflection/episodic/semantic.
    - Ofrece process_message() y add_memory().
    """
    def __init__(self,
                 model: Any = None,
                 memory: Any = None,
                 planner: Any = None,
                 task_manager: Any = None,
                 reflection_engine: Any = None,
                 episodic: Optional[Any] = None,
                 reply_style: str = "conversational",
                 use_meta_agent: bool = True):
        self.model = model or _DummyModel()
        self.memory = memory or _DummyMemory()
        self.planner = planner or _DummyPlanner()
        self.task_manager = task_manager
        self.re = reflection_engine  # puede traer build_prompt/postprocess o analyze_response
        self.episodic = episodic
        self.reply_style = reply_style
        self.use_meta_agent = use_meta_agent
        self.tools: Dict[str, Any] = {}
        # La memoria semántica (si existe) la puede inyectar el initializer como atributo:
        # self.semantic = <SemanticMemoryLite>

    # ---------- utilitario para herramientas opcionales ----------
    def _tool(self, key: str, func: str, *args, **kwargs) -> str:
        try:
            mod = self.tools.get(key)
            if not mod: return "Herramienta no disponible."
            f = getattr(mod, func, None)
            if not f: return "Acción no disponible."
            return str(f(*args, **kwargs))
        except Exception as e:
            return f"Error de herramienta: {e}"

    # ---------- API pública ----------
    def add_memory(self, text: str, metadata: dict = None) -> None:
        """Agrega una nota a la memoria semántica si existe; si no, la deja en historial."""
        metadata = metadata or {}
        sem = getattr(self, "semantic", None)
        if sem:
            try:
                sem.add(text, metadata)
                return
            except Exception:
                pass
        # fallback: historial
        self.memory.add_history("user", f"[nota] {text}")
        self.memory.persist()

    def process_message(self, user_text: str) -> str:
        plan = self.planner.detect(user_text, memory=self.memory)
        intent = plan.get("intent", "chat")
        next_step = plan.get("next", "llm.reply")
        slots = plan.get("slots", {})

        # -------- Acciones “sin LLM” / semántica directa --------
        if next_step == "memory.semantic.add":
            text = slots.get("text") or user_text.split(":", 1)[-1].strip()
            self.add_memory(text, {"tag": "nota", "source": "user"})
            self.memory.add_history("user", user_text); self.memory.persist()
            reply = "Anotado. ¿Algo más que quieras guardar?"
            self._log_episode(intent, user_text, reply)
            return reply

        if next_step == "memory.report":
            get = getattr(self.memory, "get_slot", lambda *_: None)
            name = get("name"); loc = get("location")
            prefs = getattr(self.memory, "get_preferences", lambda : [])() or []
            parts = []
            if name: parts.append(f"Nombre: {name}")
            if loc: parts.append(f"Ubicación: {loc}")
            if prefs: parts.append("Preferencias: " + ", ".join(prefs))
            if not parts: parts.append("Aún no tengo datos guardados.")
            self.memory.add_history("user", user_text); self.memory.persist()
            reply = " / ".join(parts)
            self._log_episode(intent, user_text, reply)
            return reply

        # -------- Llamada al modelo (con reflexión si existe) --------
        self.memory.add_history("user", user_text)
        prompt = user_text

        # Si el ReflectionEngine tiene build_prompt, úsalo
        if self.re and hasattr(self.re, "build_prompt"):
            try:
                prompt = self.re.build_prompt(user_text, self.memory)
            except Exception:
                prompt = user_text

        # Generación primaria
        max_new = _max_tokens_by_input(user_text)
        try:
            raw = self.model.generate(
                prompt=prompt,
                temperature=0.50,
                max_tokens=max_new,
                top_p=0.92,
                repeat_penalty=1.15,
                stop=STOP_DEFAULT
            )
        except TypeError:
            # adaptadores con firma distinta
            try:
                raw = self.model.generate(prompt=prompt, temperature=0.50, max_tokens=max_new, stop=STOP_DEFAULT)
            except Exception:
                raw = self.model.generate(prompt=prompt)

        # Post-procesado con ReflectionEngine si ofrece postprocess/analyze_response
        reply = None
        if self.re and hasattr(self.re, "postprocess"):
            try:
                reply = self.re.postprocess(raw, user_text, self.memory)
            except Exception:
                reply = _sanitize(raw, user_text)
        elif self.re and hasattr(self.re, "analyze_response"):
            try:
                reply = self.re.analyze_response(user_text, _sanitize(raw, user_text))
            except Exception:
                reply = _sanitize(raw, user_text)
        else:
            reply = _sanitize(raw, user_text)

        # Guarda historial y episodic
        self.memory.add_history("assistant", reply); self.memory.persist()
        self._log_episode(intent, user_text, reply)

        # Auto-ingesta semántica ligera (si existe)
        self._auto_ingest_semantic(reply)

        return reply

    # ---------- Helpers internos ----------
    def _auto_ingest_semantic(self, reply: str) -> None:
        sem = getattr(self, "semantic", None)
        if not sem: return
        try:
            def _is_useful(text: str) -> bool:
                t = (text or "").strip()
                if len(t) < 120: return False
                banned = ["¿en qué puedo ayudarte", "hola", "gracias por", "encantado"]
                return not any(b in t.lower() for b in banned)
            if _is_useful(reply):
                key = (reply or "").strip().lower()[:200]
                cache = getattr(self, "_sem_cache_keys", set())
                if key not in cache:
                    sem.add(reply, meta={"tag": "auto", "source": "chat"})
                    cache.add(key)
                    self._sem_cache_keys = cache
        except Exception:
            pass

    def _log_episode(self, intent: str, user_text: str, reply: str):
        epi = getattr(self, "episodic", None)
        if not epi: return
        try:
            epi.add(intent=intent, text=user_text, reply=reply, tags=[intent])
        except Exception:
            pass
