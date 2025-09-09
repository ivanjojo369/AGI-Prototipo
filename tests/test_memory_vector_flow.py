import sys
import os
import json
import time

# Ajustar ruta para importar módulos del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memory.unified_memory import UnifiedMemory

def main():
    print("🚀 Iniciando test de flujo de memoria vectorial...")

    # Inicializar memoria
    memoria = UnifiedMemory()
    print("[UM] Memoria inicializada.")

    # Guardar un recuerdo de prueba
    texto_prueba = "Mi película favorita es Interestelar."
    print(f"💾 Guardando recuerdo: '{texto_prueba}'")
    memoria.add_memory(texto_prueba)

    # Esperar un momento para asegurar escritura
    time.sleep(1)

    # Buscar el recuerdo
    print(f"🔍 Buscando recuerdo con query: 'película favorita'")
    resultados = memoria.search_memory("película favorita", top_k=3)

    # Mostrar resultados
    if resultados:
        print(f"✅ Recuerdos encontrados: {len(resultados)}")
        for r in resultados:
            print(f" - Texto: {r['text']} | Distancia: {r['distance']:.4f}")
    else:
        print("❌ No se encontraron recuerdos similares.")

    # Verificar estado
    total_memories = memoria.get_memory_count()
    status = {
        "total_memories": total_memories,
        "faiss_entries": getattr(memoria, "faiss_entries", 0),
        "gpu_enabled": getattr(memoria, "gpu_enabled", False),
        "corrupt_log_exists": getattr(memoria, "corrupt_log_exists", False)
    }
    print(f"🧠 Estado final de la memoria: {status}")

if __name__ == "__main__":
    main()
