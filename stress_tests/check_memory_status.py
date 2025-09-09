import sys
import os
import json
from memory.unified_memory import UnifiedMemory

def check_memory_status():
    print("🔵 Verificando estado de la memoria...")
    try:
        memory = UnifiedMemory()
        status = memory.get_status()
        print("✅ Estado actual de la memoria:")
        print(json.dumps(status, indent=2))
    except Exception as e:
        print(f"❌ Error verificando memoria: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_memory_status()
