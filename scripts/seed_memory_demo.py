from memory.semantic_memory import SemanticMemory

def write_any(sm, id, text, **meta):
    for name in ("write_episode", "write", "add", "upsert"):
        fn = getattr(sm, name, None)
        if not fn:
            continue
        # Intenta firmas comunes, de más completa a mínima
        try:
            return fn(id=id, text=text, meta=meta)          # (id, text, meta)
        except TypeError:
            pass
        try:
            return fn(text=text, meta=dict(id=id, **meta))  # (text, meta) con id en meta
        except TypeError:
            pass
        try:
            return fn(text)                                 # (text)
        except Exception:
            pass
    raise RuntimeError("No encontré write_episode/write/add/upsert en SemanticMemory.")

sm = SemanticMemory()
docs = [
    ("doc://1", "Kant publicó la Crítica de la razón pura en 1781."),
    ("doc://2", "La Ilustración enfatiza la razón."),
    ("doc://3", "El siglo XVIII transformó Europa."),
]
for did, txt in docs:
    write_any(sm, did, txt, source="demo")
print("Ingestados:", len(docs))
