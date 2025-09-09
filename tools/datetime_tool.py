# ==============================
# ==============================
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import locale

try:
    from zoneinfo import ZoneInfo  # Py>=3.9
except Exception:  # pragma: no cover
    ZoneInfo = None

# Ajuste opcional de locale (no falla si no existe)
for loc in ("es_MX.UTF-8", "es_ES.UTF-8", "es_MX", "es_ES"):
    try:
        locale.setlocale(locale.LC_TIME, loc)
        break
    except Exception:
        pass

DEFAULT_TZ = "America/Mexico_City"


def get_datetime(tz: str = DEFAULT_TZ, fmt: str | None = None) -> str:
    if ZoneInfo:
        try:
            now = datetime.now(ZoneInfo(tz)) if tz else datetime.now()
        except Exception:
            now = datetime.now(timezone.utc)
    else:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
    return now.strftime(fmt) if fmt else now.isoformat()


def now_str(tz: str = DEFAULT_TZ) -> str:
    return get_datetime(tz=tz, fmt="Hoy es %A, %d de %B de %Y, %H:%M:%S")


def relative(day: str, tz: str = DEFAULT_TZ) -> str:
    base = datetime.now(ZoneInfo(tz)) if ZoneInfo else datetime.now()
    if "mañana" in (day or "").lower():
        d = base + timedelta(days=1)
    elif "ayer" in (day or "").lower():
        d = base - timedelta(days=1)
    else:
        d = base
    return d.strftime("%A, %d de %B de %Y")


def handle(query: str) -> str:
    q = (query or "").lower()
    if "qué hora" in q or "hora" in q:
        return now_str()
    if any(w in q for w in ("fecha", "hoy", "mañana", "ayer")):
        if "mañana" in q:
            return "Mañana: " + relative("mañana")
        if "ayer" in q:
            return "Ayer: " + relative("ayer")
        return "Hoy: " + relative("hoy")
    return now_str()


def run(prompt: str = "", tz: str = DEFAULT_TZ, fmt: str | None = None, **kw) -> str:
    """Wrapper estándar para el selector (compatible con Agent)."""
    # Si se pasa fmt/tz, respeta; si no, devuelve ISO local
    return get_datetime(tz=tz, fmt=fmt)