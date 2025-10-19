# memory/__init__.py
from __future__ import annotations
import os
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # si no está instalado, seguimos sin YAML

from .unified_memory import UnifiedMemory

_DEF_CFG = {
    "memory_dir": "memory_store",
    "vector_dim": 1536,
    "use_gpu": False,
    "ttl_days": 0,              # 0 = desactivado
    "half_life_days": 14,       # decaimiento
    "max_items": 5000,
    "summarize_after_days": None,
}

def _load_yaml_cfg() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if yaml is None or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

class MemoryStore(UnifiedMemory):
    """
    Capa de compatibilidad para tu servidor:
    - Acepta parámetros en segundos y los traduce a días si hace falta.
    - Lee opcionalmente memory/config.yaml (si existe).
    - Mantiene los nombres que ya usas en agi_interface (ttl_seconds, max_items, decay).
    """

    def __init__(
        self,
        max_items: int = 500,
        ttl_seconds: int = 0,
        decay: float = 0.98,  # no mapea 1:1 a half-life; usamos half_life_days de config/env
        **kwargs: Any,
    ):
        cfg = dict(_DEF_CFG)
        cfg.update(_load_yaml_cfg())

        # .env → YAML fallback
        cfg["memory_dir"] = os.getenv("MEMORY_DIR", cfg["memory_dir"])
        cfg["vector_dim"] = _env_int("MEMORY_VECTOR_DIM", cfg["vector_dim"])
        cfg["use_gpu"] = os.getenv("MEMORY_USE_GPU", "0").lower() in {"1", "true", "yes"}

        # Mapear tus flags existentes (segundos → días)
        ttl_days_env = int((_env_int("MEMORY_TTL_SECONDS", ttl_seconds) or 0) / 86400)
        cfg["ttl_days"] = cfg.get("ttl_days", ttl_days_env)
        cfg["half_life_days"] = _env_int("MEMORY_HALF_LIFE_DAYS", cfg.get("half_life_days", 14))
        cfg["max_items"] = _env_int("MEMORY_MAX_ITEMS", max_items)

        # Resumen denso opcional
        if "summarize_after_days" not in cfg:
            # usa env si existe
            sad = os.getenv("MEMORY_SUMMARIZE_AFTER_DAYS")
            cfg["summarize_after_days"] = int(sad) if (sad and sad.isdigit()) else None

        super().__init__(
            memory_dir=cfg["memory_dir"],
            vector_dim=cfg["vector_dim"],
            use_gpu=cfg["use_gpu"],
            ttl_days=cfg["ttl_days"],
            half_life_days=cfg["half_life_days"],
            max_items=cfg["max_items"],
            summarize_after_days=cfg.get("summarize_after_days"),
            summarizer=None,  # puedes inyectar uno real más adelante
        )
