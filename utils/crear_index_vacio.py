import os
import faiss
import numpy as np

# Ruta de destino
DATA_DIR = os.path.join("data", "memory")
INDEX_FILE = os.path.join(DATA_DIR, "vector.index")
META_FILE = os.path.join(DATA_DIR, "vector_meta.json")

# Dimensión de embeddings (ajusta si tu sistema usa otra)
EMBEDDING_DIM = 768

def recreate_files():
    print("🧹 Limpiando archivos anteriores...")
    
    # Eliminar vector.index existente
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
        print("🗑️ Archivo vector.index anterior eliminado.")

    # Crear nuevo vector_meta.json vacío
    with open(META_FILE, "w", encoding="utf-8") as f:
        f.write("[]")
    print("🆕 Archivo vector_meta.json vacío creado.")

    # Crear índice FAISS limpio
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    faiss.write_index(index, INDEX_FILE)
    print("✅ Nuevo vector.index vacío creado correctamente.")

if __name__ == "__main__":
    recreate_files()
