from memory.semantic_memory import SemanticMemory
sm = SemanticMemory()
sm.upsert({"id":"6","text":"El Quijote fue escrito por Miguel de Cervantes.","tags":["literatura"],"meta":{}})
print("count:", sm.count())
