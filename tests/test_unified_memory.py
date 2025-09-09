import os
import sys
import json
from memory.unified_memory import UnifiedMemory

def run_basic_tests():
    print("🚀 Iniciando pruebas básicas de UnifiedMemory...\n")
    
    # Crear instancia limpia
    memory = UnifiedMemory()
    memory.clear_all()
    
    # 1️⃣ Agregar interacciones
    print("🧠 Agregando interacciones...")
    memory.add_interaction("user", "Hola, esto es un mensaje de prueba.")
    memory.add_interaction("assistant", "Hola, mensaje recibido correctamente.")

    # 2️⃣ Verificar contexto
    context = memory.get_context()
    print(f"✅ Contexto recuperado: {context}\n")

    # 3️⃣ Probar búsqueda vectorial
    print("🔎 Probando búsqueda vectorial...")
    memory.add_to_vector_memory("Recuerdo importante de prueba", metadata={"priority": 2})
    results = memory.search_vector_memory("importante", top_k=3)
    print(f"✅ Resultados de búsqueda: {results}\n")

    # 4️⃣ Almacenar evento episódico
    print("📅 Almacenando evento episódico...")
    memory.store_event("evento_prueba", "Se registró un evento de prueba.")
    events = memory.get_recent_events()
    print(f"✅ Eventos recuperados: {events}\n")

    # 5️⃣ Almacenar reflexión
    print("🪞 Almacenando reflexión...")
    memory.store_reflection("Esta es una reflexión de prueba.")
    reflections = memory.retrieve_recent_reflections()
    print(f"✅ Reflexiones recuperadas: {reflections}\n")

    # 6️⃣ Verificar estado
    status = memory.get_status()
    print(f"📊 Estado actual de memoria: {json.dumps(status, indent=2)}\n")

    print("✅ Todas las pruebas básicas completadas con éxito.\n")

if __name__ == "__main__":
    run_basic_tests()
