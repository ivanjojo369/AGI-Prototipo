import os
import json
import re
import shutil
import faiss
from datetime import datetime

DATA_DIR = os.path.join("data", "memory")
LOG_DIR = os.path.join("logs", "benchmarks", "memory")
os.makedirs(LOG_DIR, exist_ok=True)

CURRENT_META = os.path.join(DATA_DIR, "vector_meta.json")
BACKUP_CORRUPT = os.path.join(DATA_DIR, "vector_meta_backup_corrupto.json")
RESTORED_META = os.path.join(DATA_DIR, "vector_meta_restaurado.json")
FAISS_INDEX = os.path.join(DATA_DIR, "vector.index")

def backup_and_restore():
    print("🛠️ Renombrando y restaurando archivos...")
    if os.path.exists(CURRENT_META):
        shutil.move(CURRENT_META, CURRENT_META.replace(".json", "_old.json"))
        print("✅ Archivo actual renombrado como _old.json")

    if os.path.exists(BACKUP_CORRUPT):
        shutil.copy(BACKUP_CORRUPT, RESTORED_META)
        print("✅ Backup corrupto copiado a vector_meta_restaurado.json")
    else:
        print("❌ No existe backup corrupto.")
        return False
    return True

def clean_json(file_path):
    print("🧹 Limpiando y reparando JSON...")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_data = f.read()

    # Intento de reparación
    repaired_data = re.sub(r'(\}\s*\{)', '},{', raw_data).strip()
    if not repaired_data.startswith("["):
        repaired_data = "[" + repaired_data
    if not repaired_data.endswith("]"):
        repaired_data = repaired_data + "]"

    try:
        data = json.loads(repaired_data)
        valid_data = [obj for obj in data if isinstance(obj, dict) and "embedding" in obj]
        print(f"📦 Recuperados {len(valid_data)} recuerdos válidos / {len(data)} totales")

        with open(CURRENT_META, "w", encoding="utf-8") as out:
            json.dump(valid_data, out, indent=2, ensure_ascii=False)
        return valid_data
    except json.JSONDecodeError as e:
        print(f"❌ Error al reparar JSON: {e}")
        return []

def rebuild_faiss(memories):
    if not memories:
        print("⚠️ No hay recuerdos válidos para FAISS.")
        return
    print("🔄 Reconstruyendo índice FAISS...")
    dimension = len(memories[0]["embedding"])
    index = faiss.IndexFlatL2(dimension)

    vectors = []
    for mem in memories:
        try:
            vectors.append(mem["embedding"])
        except KeyError:
            continue

    import numpy as np
    vectors_np = np.array(vectors).astype('float32')
    index.add(vectors_np)

    faiss.write_index(index, FAISS_INDEX)
    print(f"✅ Índice FAISS reconstruido con {len(vectors)} recuerdos.")

def save_report(recovered, total):
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_recuerdos": total,
        "recuerdos_recuperados": recovered,
        "archivo_resultante": CURRENT_META,
        "faiss_index": FAISS_INDEX
    }
    report_path = os.path.join(LOG_DIR, "migrate_and_clean_report.json")
    with open(report_path, "w", encoding="utf-8") as rep:
        json.dump(report, rep, indent=2, ensure_ascii=False)
    print(f"📝 Reporte guardado en {report_path}")

def main():
    print("🚀 Iniciando migración y limpieza de vector_meta...")
    if not backup_and_restore():
        return
    memories = clean_json(RESTORED_META)
    if memories:
        rebuild_faiss(memories)
        save_report(len(memories), len(memories))
        print("🎉 Migración y limpieza completadas correctamente.")
    else:
        print("❌ No se pudieron recuperar recuerdos.")

if __name__ == "__main__":
    main()
