from memory.semantic_memory import SemanticMemory

sm = SemanticMemory()
docs = [
    {"id":"1","text":"La Crítica de la razón pura de Kant se publicó en 1781.","tags":["historia","filosofia"],"meta":{}},
    {"id":"2","text":"Python fue creado por Guido van Rossum.","tags":["programacion"],"meta":{}},
    {"id":"3","text":"París es la capital de Francia.","tags":["geografia"],"meta":{}},
]
for d in docs:
    sm.upsert(d)
print("count:", sm.count())
print("paths:", sm.paths)
