import inspect
from memory import memory as m

print("__all__:", m.__all__)
for name in m.__all__:
    obj = getattr(m, name, None)
    if callable(obj):
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "()"
        print(f"{name}{sig}")
    else:
        print(f"{name} (class/const)")
