# scripts/inspect_semantic_api.py
import os, sys, inspect
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from memory.semantic_memory import SemanticMemory

sm = SemanticMemory()
print("Clase:", type(sm).__name__)
print("\nMétodos públicos:")
names = [n for n in dir(sm) if not n.startswith("_")]
for n in sorted(names):
    attr = getattr(sm, n)
    if callable(attr):
        try:
            sig = str(inspect.signature(attr))
        except Exception:
            sig = "(sin signature)"
        print(f"  - {n}{sig}")
