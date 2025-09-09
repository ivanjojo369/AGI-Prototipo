import pynvml
import json
import time
import os

LOG_DIR = "logs/benchmarks/gpu/"
os.makedirs(LOG_DIR, exist_ok=True)

def monitor_gpu():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    print("📊 Midiendo VRAM de la GPU...")
    start = time.time()
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    end = time.time()

    result = {
        "total": round(memory_info.total / (1024**2), 2),
        "used": round(memory_info.used / (1024**2), 2),
        "free": round(memory_info.free / (1024**2), 2),
        "timestamp": round(time.time()),
        "measurement_time": round(end - start, 4)
    }

    filename = os.path.join(LOG_DIR, f"benchmark_vram_{result['timestamp']}.json")
    with open(filename, "w") as f:
        json.dump(result, f, indent=4)

    print(f"✅ VRAM usada: {result['used']} MB / {result['total']} MB")
    print(f"📁 Resultados guardados en: {filename}")

    pynvml.nvmlShutdown()

if __name__ == "__main__":
    monitor_gpu()
