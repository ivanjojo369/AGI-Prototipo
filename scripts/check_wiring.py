# scripts/check_wiring.py
from __future__ import annotations
import json, sys
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).name != "check_wiring.py" else Path(__file__).resolve().parent.parent
print(f"[i] ROOT = {ROOT}")

def exists(p): 
    p = ROOT / p
    ok = p.exists()
    print(f" - {'✓' if ok else '✗'} {p}")
    return ok

print("\n[1] Archivos esperados:")
exists("agi_initializer.py")
exists("agi_interface.py")
exists("agi_webapp.py")
exists("settings.json")
exists("adapters/llama_cpp_adapter.py")
exists("agents/reflection_engine.py")
exists("memory/context_memory.py")
exists("planner/planner.py")
exists("task_manager.py")

print("\n[2] settings.json (si existe):")
cfgp = ROOT / "settings.json"
if cfgp.exists():
    try:
        cfg = json.loads(cfgp.read_text(encoding="utf-8"))
        model_path = cfg.get("model", {}).get("model_path")
        print(" - model_path:", model_path)
        print(" - n_ctx:", cfg.get("model", {}).get("n_ctx"))
        print(" - use_gpu:", cfg.get("model", {}).get("use_gpu"))
    except Exception as e:
        print(" ! Error leyendo settings.json:", e)
else:
    print(" - (no existe; lo crea el initializer en la primera corrida)")

def try_import(file_rel: str, mod_name: str):
    p = ROOT / file_rel
    if not p.exists():
        print(f" ! Falta {file_rel}")
        return None
    try:
        spec = importlib.util.spec_from_file_location(mod_name, str(p))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        print(f" ✓ import {mod_name}")
        return mod
    except Exception as e:
        print(f" ✗ import {mod_name} -> {e}")
        return None

print("\n[3] Imports clave:")
m_adapter = try_import("adapters/llama_cpp_adapter.py", "llama_cpp_adapter")
m_reflex = try_import("agents/reflection_engine.py", "reflection_engine")
m_ctxmem = try_import("memory/context_memory.py", "context_memory")
m_planner = try_import("planner/planner.py", "planner")

print("\n[4] Clases expuestas:")
def has(mod, cls):
    try:
        ok = hasattr(mod, cls)
        print(f" - {mod.__name__}.{cls}: {'✓' if ok else '✗'}")
    except Exception as e:
        print(f" - {mod} -> {e}")

if m_reflex: has(m_reflex, "ReflectionEngine")
if m_ctxmem: has(m_ctxmem, "ContextMemory")
if m_planner: has(m_planner, "Planner")
if m_adapter:
    print(" - llama_cpp_adapter.load_model:", "✓" if hasattr(m_adapter, "load_model") else "✗")
