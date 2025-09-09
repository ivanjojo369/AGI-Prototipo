import os
import json
import time
import numpy as np
from tqdm import tqdm
from memory.unified_memory import UnifiedMemory

TOTAL_RECUERDOS = 50000
BACKUP_PATH = os.path.join("data", "memory", "backups", "stress_memory_backup.json")
CHECKPOINT_DIR = os.path.join("logs", "benchmarks", "memory")
REPORT_PATH = os.path.join(CHECKPOINT_DIR, "stress_memory_report.json")

def find_last_checkpoint():
    """Buscar el último checkpoint guardado."""
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint_")]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
    return os.path.join(CHECKPOINT_DIR, latest), int(latest.split("_")[1].split(".")[0])

def run_resume_test():
    print("🔄 Reanudando stress test de memoria...")
    memory = UnifiedMemory()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Restaurar desde backup si está vacío
    try:
        current_data = memory.export_memories()
        if len(current_data) == 0 and os.path.exists(BACKUP_PATH):
            with open(BACKUP_PATH, "r", encoding="utf-8") as f:
                restored = json.load(f)
                memory.import_memories(restored)
            print("🟢 Memoria restaurada desde backup.")
    except Exception as e:
        print(f"⚠️ Error restaurando memoria: {e}")

    # Buscar último checkpoint
    checkpoint_file, start_index = (None, 0)
    found = find_last_checkpoint()
    if found:
        checkpoint_file, start_index = found
        print(f"📌 Reanudando desde checkpoint: {checkpoint_file} (recuerdo {start_index})")

    for i in tqdm(range(start_index, TOTAL_RECUERDOS), desc="Insertando recuerdos", unit="rec"):
        dummy_embedding = np.random.rand(768).tolist()
        memory.add_memory(f"Recuerdo {i}", dummy_embedding)

        if (i + 1) % 5000 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{i+1}.json")
            memory.save(checkpoint_path)

    print("🧹 Ejecutando compactación final...")
    memory.compact_memory()

    total = memory.get_memory_count()
    report = {
        "total_recuerdos": total,
        "status": "completado",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)

    print(f"✅ Stress test completado con éxito. Total recuerdos: {total}")
    print(f"📄 Reporte guardado en: {REPORT_PATH}")

if __name__ == "__main__":
    run_resume_test()
