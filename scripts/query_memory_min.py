from memory.semantic_memory import SemanticMemory

sm = SemanticMemory()
qs = [
    "¿Cuándo se publicó la Crítica de la razón pura?",
    "Lenguaje de programación creado por Guido van Rossum",
    "Capital de Francia"
]
for q in qs:
    hits = sm.search(q, top_k=5)
    print("Q:", q)
    print("Top:", [(h["doc_id"], round(float(h.get("score",0)),3)) for h in hits])
