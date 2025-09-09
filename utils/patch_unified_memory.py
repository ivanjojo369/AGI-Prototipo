import os
import re

def patch_unified_memory():
    print("🔧 Aplicando parche a unified_memory.py...")

    file_path = os.path.join("memory", "unified_memory.py")
    
    if not os.path.exists(file_path):
        print("❌ Archivo unified_memory.py no encontrado.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Buscar y reemplazar la función get_status si existe, o agregarla si no está
    new_function = '''
def get_status(self):
    """
    Devuelve el estado actual de la memoria unificada,
    asegurando que siempre existan las claves esperadas.
    """
    try:
        status = {
            "vector_memories": len(self.vector_memories) if hasattr(self, "vector_memories") else 0,
            "reflections": len(self.reflections) if hasattr(self, "reflections") else 0,
            "context": len(self.short_term_context) if hasattr(self, "short_term_context") else 0
        }
    except Exception as e:
        print(f"⚠️ Error obteniendo estado de la memoria: {e}")
        status = {
            "vector_memories": 0,
            "reflections": 0,
            "context": 0
        }
    
    return status
'''

    if "def get_status" in content:
        # Reemplazar la función existente
        content = re.sub(
            r"def get_status\(.*?\):.*?return .*?\n",
            new_function,
            content,
            flags=re.S
        )
        print("✏️ Función get_status reemplazada.")
    else:
        # Insertar al final de la clase UnifiedMemory
        content = re.sub(
            r"(class UnifiedMemory\(.*?\):)",
            r"\1\n" + new_function,
            content,
            flags=re.S
        )
        print("➕ Función get_status agregada.")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Parche aplicado correctamente a unified_memory.py")

if __name__ == "__main__":
    patch_unified_memory()
