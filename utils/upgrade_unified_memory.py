import os
import json
import shutil
import faiss
import numpy as np

DATA_PATH = os.path.join("data", "memory")
META_FILE = os.path.join(DATA_PATH, "vector_meta.json")
BACKUP_FILE = os.path.join(DATA_PATH, "vector_meta_backup_pre_upgrade.json")
INDEX_FILE = os.path.join(DATA_PATH, "vector.index")
VECTOR_SIZE = 128

def backup_file():
    if os.path.exists(META_FILE):
        shutil.copy2(META_FILE, BACKUP_FILE)
        print(f"📦 Backup creado: {BACKUP_FILE}")
    else:
        print("⚠️ No existe archivo vector_meta.json, se iniciará vacío.")

def load_and_clean_json():
    if not os.path.exists(META_FILE):
        return []

    try:
        with open(META_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        cleaned_data = []
        for i, item in enumerate(data):
            if isinstance(item.get("embedding"), list) and len(item["embedding"]) == VECTOR_SIZE:
                cleaned_data.append(item)
            else:
                print(f"⚠️ Registro corrupto ignorado en línea {i}")

        print(f"✅ Total registros válidos: {len(cleaned_data)}")
        print(f"🗑️ Registros corruptos eliminados: {len(data) - len(cleaned_data)}")
        return cleaned_data

    except Exception as e:
        print(f"❌ Error al leer JSON: {e}")
        return []

def rebuild_faiss_index(memories):
    print("🔄 Reconstruyendo índice FAISS...")
    index = faiss.IndexFlatL2(VECTOR_SIZE)
    if memories:
        embeddings = np.array([m["embedding"] for m in memories], dtype="float32")
        index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print(f"✅ Índice FAISS reconstruido con {len(memories)} vectores.")

def save_cleaned_data(memories):
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)
    print("💾 vector_meta.json actualizado y limpio.")

def main():
    print("🚀 Iniciando migración y actualización de Unified Memory...")
    os.makedirs(DATA_PATH, exist_ok=True)

    backup_file()
    memories = load_and_clean_json()
    save_cleaned_data(memories)
    rebuild_faiss_index(memories)

    print("🎉 Migración completada exitosamente. El sistema ya es compatible con el nuevo unified_memory.py.")

if __name__ == "__main__":
    main()
