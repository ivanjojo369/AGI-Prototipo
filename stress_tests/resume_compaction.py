import os
import json
from datetime import datetime
from memory.unified_memory import UnifiedMemory

def resume_compaction():
    memory = UnifiedMemory()

    # Conteo antes de compactar
    total_before = memory.get_memory_count()
    vector_before = len(memory.vector_memories)
    reflections_before = len(memory.reflections)
    context_before = len(memory.context_short_term)

    print("🔵 Reanudando compactación de memoria...")
    print(f"🧠 Recuerdos actuales: {total_before}")
    print("⚙️ Ejecutando compactación...")

    # Ejecutar compactación
    memory.compact_memory()

    # Conteo después de compactar
    total_after = memory.get_memory_count()
    vector_after = len(memory.vector_memories)
    reflections_after = len(memory.reflections)
    context_after = len(memory.context_short_term)

    print("✅ Compactación finalizada.")

    # Crear carpeta para reportes
    report_dir = os.path.join("logs", "benchmarks", "memory")
    os.makedirs(report_dir, exist_ok=True)

    # Guardar reporte JSON
    report_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "before": {
            "total": total_before,
            "vector_memories": vector_before,
            "reflections": reflections_before,
            "context_short_term": context_before
        },
        "after": {
            "total": total_after,
            "vector_memories": vector_after,
            "reflections": reflections_after,
            "context_short_term": context_after
        }
    }

    report_path = os.path.join(report_dir, "resume_compaction_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    print(f"📝 Reporte final guardado en {report_path}")

if __name__ == "__main__":
    resume_compaction()
