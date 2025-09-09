from memory.unified_memory import UnifiedMemory

def main():
    print("=== 🔍 Inspeccionando métodos de UnifiedMemory ===")
    memoria = UnifiedMemory()

    # Listar métodos y atributos públicos
    metodos = [m for m in dir(memoria) if not m.startswith("_")]
    for metodo in metodos:
        print("•", metodo)

if __name__ == "__main__":
    main()
