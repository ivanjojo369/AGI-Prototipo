from memory.semantic_memory import SemanticMemory
sm = SemanticMemory()
print("hits:", sm.search("¿Quién escribió Don Quijote?", top_k=3))
