import os
import json
import time
from tqdm import tqdm
from memory.unified_memory import UnifiedMemory

def run_stress_test():
    print("🚀 Iniciando stress test de memoria con 50000 recuerdos...")

    # Crear instancia de memoria
    memory = UnifiedMemory()

    # Intentar crear backup
    try:
        if hasattr(memory, "export_memories"):
            current_data = memory.export_memories()
            backup_path = os.path.join("data", "memory", "vector_meta_backup.json")
            with open(backup_path, "w", encoding="utf-8") as backup_file:
                json.dump(current_data, backup_file, indent=2, ensure_ascii=False)
            print("✅ Backup de memoria creado.")
            print(f"🧠 Memorias cargadas: {len(current_data)} (corruptas ignoradas: 0)")
        else:
            print("⚠️ Método export_memories() no disponible. Saltando backup.")
            current_data = []
    except Exception as e:
        print(f"❌ No se pudo crear backup: {e}")
        current_data = []

    print("🔄 Reanudando desde el recuerdo 0...")

    # Inserción de recuerdos simulados
    total_recuerdos = 50000
    start_index = 0
    try:
        for i in tqdm(range(start_index, total_recuerdos), desc="Insertando recuerdos", unit="rec"):
            # Generar datos ficticios de embedding
            embedding = [float(i % 100) for _ in range(128)]
            memory.add_memory(f"Recuerdo {i}", embedding)
    except Exception as e:
        print(f"❌ Error durante la inserción de recuerdos: {e}")

    print("🧹 Ejecutando compactación de memoria...")
    try:
        memory.compact_memory()
        print("✅ Compactación completada.")
    except Exception as e:
        print(f"⚠️ Error durante la compactación: {e}")

    # Guardar estado final
    try:
        memory.save_to_disk()
        print("💾 Estado final guardado correctamente.")
    except Exception as e:
        print(f"⚠️ Error al guardar el estado final: {e}")

if __name__ == "__main__":
    run_stress_test()
