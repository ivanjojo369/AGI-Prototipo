import os
import json
import numpy as np
import faiss
from memory.unified_memory import UnifiedMemory
from stress_tests.check_memory_status import check_memory_status

def migrate_vector_meta():
    print("🔄 Iniciando migración de vector_meta.json...")

    data_dir = os.path.join("data", "memory")
    vector_path = os.path.join(data_dir, "vector_meta.json")
    index_path = os.path.join(data_dir, "vector.index")

    if not os.path.exists(vector_path):
        print("❌ No se encontró vector_meta.json, nada que migrar.")
        return

    # Cargar recuerdos
    with open(vector_path, "r", encoding="utf-8") as f:
        memories = json.load(f)

    memory_system = UnifiedMemory()
    updated_memories = []
    embeddings = []
    repaired_count = 0
    ok_count = 0

    for m in memories:
        try:
            if "embedding" not in m or not isinstance(m["embedding"], list):
                emb = memory_system._embed_text(m.get("content", ""))
                m["embedding"] = emb.tolist()
                repaired_count += 1
                print(f"🛠️ Embedding regenerado para ID: {m.get('id', 'sin_id')}")
            else:
                ok_count += 1

            embeddings.append(np.array(m["embedding"], dtype=np.float32))
            updated_memories.append(m)
        except Exception as e:
            print(f"⚠️ Error procesando memoria ID {m.get('id', 'sin_id')}: {e}")

    # Guardar recuerdos reparados
    with open(vector_path, "w", encoding="utf-8") as f:
        json.dump(updated_memories, f, indent=2, ensure_ascii=False)

    # Reconstruir índice FAISS
    if embeddings:
        embeddings_array = np.vstack(embeddings).astype(np.float32)
        index = faiss.IndexFlatL2(embeddings_array.shape[1])
        index.add(embeddings_array)
        faiss.write_index(index, index_path)
        print(f"📦 Índice FAISS reconstruido y guardado en {index_path}")

    print("\n✅ Migración completada.")
    print(f"   📌 Recuerdos reparados: {repaired_count}")
    print(f"   📌 Recuerdos que ya estaban correctos: {ok_count}")
    print(f"   📦 Total recuerdos: {len(updated_memories)}")

    # Ejecutar verificación final
    print("\n🔎 Ejecutando verificación del estado de la memoria...")
    check_memory_status()

if __name__ == "__main__":
    migrate_vector_meta()
