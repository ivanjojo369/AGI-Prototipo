# agi_interface.py
# Interfaz AGI central para AGI-Prototipo / AgentBeats.
# Diseñada para ser robusta a cambios en model, memory, planner y reflection_engine,
# con dummies de respaldo y sin dependencias fuertes de la "initializer".

from __future__ import annotations

from typing import Any, Dict, Optional
import re
import logging


logger = logging.getLogger(__name__)


# ---- Stopwords/guards típicos para modelos instruct (llama.cpp / OpenChat) ----
STOP_DEFAULT = [
    "<|end_of_turn|>", "</s>",
    "Usuario:", "User:", "\nUsuario:", "\nUser:",
    "MENSAJE DEL USUARIO", "MENSAJE DEL ASISTENTE",
    "[RESPONDER]", "RESPONDER:", "Instrucciones:", "INSTRUCCIONES:"
]


# ---------------------------------------------------------------------------
# Utilidades de sanitización de respuestas
# ---------------------------------------------------------------------------

def _dedupe_lines(text: str) -> str:
    if not text:
        return text
    out: list[str] = []
    for ln in text.splitlines():
        if not out or ln.strip() != out[-1].strip():
            out.append(ln.rstrip())
    return "\n".join(out).strip()


def _remove_user_echo(text: str, user_text: str) -> str:
    if not text or not user_text:
        return text or ""
    # 1) reemplazo directo
    t = text.replace(user_text.strip(), "")
    # 2) patrón regex (texto del usuario sin signos finales)
    ut = re.escape(user_text.strip().rstrip("?.!…"))
    if ut:
        t = re.sub(rf"\b{ut}\b[?.!…]*", "", t, flags=re.IGNORECASE)
    return t.strip()


def _strip_markers(text: str) -> str:
    if not text:
        return ""
    SYS = [
        "[SISTEMA]", "[SYSTEM]", "[CONTEXTO]", "[CONTEXT]", "[USUARIO]", "[USER]",
        "[ASISTENTE]", "[ASSISTANT]", "[RESPONDER]", "[RESPONSE]",
        "<think>", "</think>", "[INSTRUCCIONES]", "INSTRUCCIONES", "INSTRUCTIONS",
        "MENSAJE DEL USUARIO", "MENSAJE DEL ASISTENTE"
    ]
    t = text
    for m in SYS:
        t = t.replace(m, "")

    # elimina líneas que son instrucciones internas
    lines: list[str] = []
    for ln in t.splitlines():
        low = ln.lower().strip()
        if low.startswith("- no ") or "no imprimas" in low or "no cites" in low:
            continue
        if low in ("[responder]", "[response]"):
            continue
        lines.append(ln)
    t = "\n".join(lines)

    # limpia bullets huérfanos y saltos extra
    t = re.sub(r"(?:^|\n)[\-\*\•]\s*$", "", t.strip())
    t = re.sub(r"(?:\n\s*){3,}", "\n\n", t)
    return t.strip()


def _sanitize(text: str, user_text: str = "") -> str:
    """Limpia eco del usuario, marcadores internos y ruido."""
    t = (text or "").strip()
    t = _strip_markers(t)
    t = _remove_user_echo(t, user_text)
    t = _dedupe_lines(t)
    if len(t) > 2000:
        t = t[:2000].rstrip() + "…"
    return t


def _max_tokens_by_input(user_text: str) -> int:
    n = len(user_text or "")
    if n <= 40:
        return 96
    if n <= 160:
        return 192
    return 256


# ---------------------------------------------------------------------------
# Stubs de respaldo (por si el initializer no inyecta algo)
# ---------------------------------------------------------------------------

class _DummyModel:
    def generate(
        self,
        prompt: str,
        temperature: float = 0.6,
        max_tokens: int = 256,
        **kw: Any,
    ) -> str:
        return "Estoy listo y funcionando. ¿En qué te ayudo?"


class _DummyMemory:
    """
    Memoria mínima en RAM, solo para que el sistema no truene
    cuando no se inyecta una implementación real.
    """
    def __init__(
        self,
        persist_path: Optional[str] = None,
        save_to_disk: bool = True,
        max_history: int = 64,
    ) -> None:
        self._hist: list[Any] = []
        self.max_history = max_history

    # API "friendly" para AGIInterface
    def add_history(self, role: str, content: str) -> None:
        self._hist.append(type("Msg", (), {"role": role, "content": content}))
        self._hist = self._hist[-self.max_history :]

    def recall_recent(self, k: int = 6) -> list[Any]:
        return self._hist[-k:]

    def summary_text(self) -> str:
        return ""

    def get_slot(self, key: str) -> Any:
        return None

    def update_slot(self, key: str, val: Any) -> None:
        return None

    def add_preference(self, p: str) -> None:
        return None

    def get_preferences(self) -> list[str]:
        return []

    def persist(self) -> None:
        return None


class _DummyPlanner:
    def detect(self, user_text: str, memory: Any = None) -> Dict[str, Any]:
        """
        Heurísticas mínimas:
        - "guarda:" | "guardar:" | "save:" → nota a memoria semántica
        - "recuerda" / "qué hablamos" → reporte de memoria
        - resto → chat normal
        """
        txt = (user_text or "").strip().lower()
        if txt.startswith(("guarda:", "guardar:", "save:")):
            return {
                "intent": "save_note",
                "next": "memory.semantic.add",
                "slots": {
                    "text": user_text.split(":", 1)[1].strip() if ":" in user_text else user_text
                },
            }
        if txt.startswith(("recuerda", "qué hablamos")):
            return {"intent": "recall", "next": "memory.report", "slots": {}}
        return {"intent": "chat", "next": "llm.reply", "slots": {}}


# ---------------------------------------------------------------------------
# AGIInterface principal
# ---------------------------------------------------------------------------

class AGIInterface:
    """
    Interfaz AGI flexible para AGI-Prototipo / AgentBeats.

    - Soporta __init__ sin argumentos (modo simple con dummies).
    - Soporta inyección de model / memory / planner / reflection_engine / episodic / semantic.
    - Ofrece:
        * process_message(user_text: str) -> str
        * add_memory(text: str, metadata: dict | None = None) -> None
    """

    def __init__(
        self,
        model: Any = None,
        memory: Any = None,
        planner: Any = None,
        task_manager: Any = None,
        reflection_engine: Any = None,
        episodic: Optional[Any] = None,
        reply_style: str = "conversational",
        use_meta_agent: bool = True,
    ) -> None:
        self.model = model or _DummyModel()
        self.memory = memory or _DummyMemory()
        self.planner = planner or _DummyPlanner()
        self.task_manager = task_manager
        self.re = reflection_engine  # alias corto
        self.reflection_engine = reflection_engine
        self.episodic = episodic
        self.reply_style = reply_style
        self.use_meta_agent = use_meta_agent
        self.tools: Dict[str, Any] = {}
        # La memoria semántica (si existe) la puede inyectar el initializer como atributo:
        # self.semantic = <SemanticMemoryLite>  (si no existe, _auto_ingest_semantic la ignora)
        self._sem_cache_keys: set[str] = set()

    # ---------- utilitario para herramientas opcionales ----------

    def _tool(self, key: str, func: str, *args: Any, **kwargs: Any) -> str:
        try:
            mod = self.tools.get(key)
            if not mod:
                return "Herramienta no disponible."
            f = getattr(mod, func, None)
            if not callable(f):
                return "Acción no disponible."
            return str(f(*args, **kwargs))
        except Exception as e:  # pragma: no cover - defensivo
            logger.debug("Error en herramienta %s.%s: %s", key, func, e)
            return f"Error de herramienta: {e}"

    # ---------- helpers internos: memoria segura ----------

    def _mem_add_history(self, role: str, content: str) -> None:
        msg = str(content)
        fn = getattr(self.memory, "add_history", None)
        if callable(fn):
            try:
                fn(role, msg)
                return
            except Exception as e:  # pragma: no cover - defensivo
                logger.debug("Error en memory.add_history: %s", e)
        # fallback interno
        hist = getattr(self, "_fallback_history", [])
        hist.append({"role": role, "content": msg})
        setattr(self, "_fallback_history", hist)

    def _mem_persist(self) -> None:
        fn = getattr(self.memory, "persist", None)
        if callable(fn):
            try:
                fn()
            except Exception as e:  # pragma: no cover - defensivo
                logger.debug("Error en memory.persist: %s", e)

    # ---------- API pública ----------

    def add_memory(self, text: str, metadata: Optional[dict] = None) -> None:
        """
        Agrega una nota a la memoria semántica si existe; si no, usa historial.
        """
        metadata = metadata or {}
        sem = getattr(self, "semantic", None)
        if sem:
            try:
                add = getattr(sem, "add", None)
                if callable(add):
                    add(text, metadata)
                    return
            except Exception as e:  # pragma: no cover - defensivo
                logger.debug("Error en semantic.add: %s", e)

        # fallback: historial
        self._mem_add_history("user", f"[nota] {text}")
        self._mem_persist()

    def process_message(self, user_text: str) -> str:
        """
        Punto de entrada principal para la conversación.
        Orquesta planner → modelo → reflexión → memoria.
        """
        # -------- PLANIFICACIÓN / INTENCIÓN --------
        try:
            try:
                plan = self.planner.detect(user_text, memory=self.memory)
            except TypeError:
                # planner.detect puede no aceptar 'memory='
                plan = self.planner.detect(user_text)  # type: ignore[misc]
        except Exception as e:  # pragma: no cover - defensivo
            logger.debug("Error en planner.detect: %s", e)
            plan = {"intent": "chat", "next": "llm.reply", "slots": {}}

        intent = plan.get("intent", "chat")
        next_step = plan.get("next", "llm.reply")
        slots = plan.get("slots", {}) or {}

        # -------- Acciones “sin LLM” / semántica directa --------
        if next_step == "memory.semantic.add":
            text = slots.get("text") or user_text.split(":", 1)[-1].strip()
            self.add_memory(text, {"tag": "nota", "source": "user"})
            self._mem_add_history("user", user_text)
            self._mem_persist()
            reply = "Anotado. ¿Algo más que quieras guardar?"
            self._log_episode(intent, user_text, reply)
            return reply

        if next_step == "memory.report":
            get_slot = getattr(self.memory, "get_slot", None)
            name = get_slot("name") if callable(get_slot) else None
            loc = get_slot("location") if callable(get_slot) else None

            get_prefs = getattr(self.memory, "get_preferences", None)
            prefs = get_prefs() if callable(get_prefs) else []

            parts = []
            if name:
                parts.append(f"Nombre: {name}")
            if loc:
                parts.append(f"Ubicación: {loc}")
            if prefs:
                parts.append("Preferencias: " + ", ".join(prefs))

            if not parts:
                parts.append("Aún no tengo datos guardados.")

            self._mem_add_history("user", user_text)
            self._mem_persist()
            reply = " / ".join(parts)
            self._log_episode(intent, user_text, reply)
            return reply

        # -------- Llamada al modelo (con reflexión si existe) --------
        self._mem_add_history("user", user_text)

        prompt = user_text
        # Si el ReflectionEngine tiene build_prompt, úsalo
        if self.re and hasattr(self.re, "build_prompt"):
            try:
                prompt = self.re.build_prompt(user_text, self.memory)  # type: ignore[assignment]
            except Exception as e:  # pragma: no cover - defensivo
                logger.debug("Error en reflection.build_prompt: %s", e)
                prompt = user_text

        # Generación primaria (robusta frente a variaciones de firma)
        max_new = _max_tokens_by_input(user_text)
        raw = ""
        gen = getattr(self.model, "generate", None)

        if callable(gen):
            try:
                raw = gen(
                    prompt=prompt,
                    temperature=0.50,
                    max_tokens=max_new,
                    top_p=0.92,
                    repeat_penalty=1.15,
                    stop=STOP_DEFAULT,
                )
            except TypeError:
                try:
                    raw = gen(
                        prompt=prompt,
                        temperature=0.50,
                        max_tokens=max_new,
                        stop=STOP_DEFAULT,
                    )
                except TypeError:
                    raw = gen(prompt)
            except Exception as e:  # pragma: no cover - defensivo
                logger.debug("Error en model.generate: %s", e)
                raw = _DummyModel().generate(prompt)
        else:
            # modelos que son directamente "callables"
            try:
                raw = self.model(prompt)
            except Exception as e:  # pragma: no cover - defensivo
                logger.debug("Error en model callable: %s", e)
                raw = _DummyModel().generate(prompt)

        # Post-procesado con ReflectionEngine si ofrece postprocess/analyze_response
        reply: Optional[str] = None
        if self.re:
            if hasattr(self.re, "postprocess"):
                try:
                    reply = self.re.postprocess(raw, user_text, self.memory)  # type: ignore[assignment]
                except Exception as e:  # pragma: no cover - defensivo
                    logger.debug("Error en reflection.postprocess: %s", e)
                    reply = _sanitize(raw, user_text)
            elif hasattr(self.re, "analyze_response"):
                try:
                    reply = self.re.analyze_response(user_text, _sanitize(raw, user_text))  # type: ignore[assignment]
                except Exception as e:  # pragma: no cover - defensivo
                    logger.debug("Error en reflection.analyze_response: %s", e)
                    reply = _sanitize(raw, user_text)

        if reply is None:
            reply = _sanitize(raw, user_text)

        # Guarda historial y episodic
        self._mem_add_history("assistant", reply)
        self._mem_persist()
        self._log_episode(intent, user_text, reply)

        # Auto-ingesta semántica ligera (si existe)
        self._auto_ingest_semantic(reply)

        return reply

    # ---------- Helpers internos ----------

    def _auto_ingest_semantic(self, reply: str) -> None:
        sem = getattr(self, "semantic", None)
        if not sem:
            return
        try:
            def _is_useful(text: str) -> bool:
                t = (text or "").strip()
                if len(t) < 120:
                    return False
                banned = ["¿en qué puedo ayudarte", "hola", "gracias por", "encantado"]
                return not any(b in t.lower() for b in banned)

            if _is_useful(reply):
                key = (reply or "").strip().lower()[:200]
                cache = self._sem_cache_keys
                if key not in cache:
                    add = getattr(sem, "add", None)
                    if callable(add):
                        add(reply, meta={"tag": "auto", "source": "chat"})
                        cache.add(key)
        except Exception as e:  # pragma: no cover - defensivo
            logger.debug("Error en _auto_ingest_semantic: %s", e)

    def _log_episode(self, intent: str, user_text: str, reply: str) -> None:
        epi = getattr(self, "episodic", None)
        if not epi:
            return
        try:
            add = getattr(epi, "add", None)
            if callable(add):
                add(intent=intent, text=user_text, reply=reply, tags=[intent])
        except Exception as e:  # pragma: no cover - defensivo
            logger.debug("Error en episodic.add: %s", e)
