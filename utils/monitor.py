import threading
import time
import os

try:
    import psutil
except ImportError:
    raise ImportError("Necesitas instalar psutil: pip install psutil")

try:
    import GPUtil
    gpu_available = True
except ImportError:
    gpu_available = False

def show_resource_status():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("📊 USO DE RECURSOS DEL SISTEMA AGI")
        print("-" * 40)

        # CPU
        cpu_usage = psutil.cpu_percent(interval=0.5)
        print(f"🧠 CPU Usage: {cpu_usage:.1f}%")

        # RAM
        ram = psutil.virtual_memory()
        print(f"📦 RAM Usage: {ram.percent:.1f}% ({round(ram.used / (1024**3), 2)} GB / {round(ram.total / (1024**3), 2)} GB)")

        # GPU
        if gpu_available:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                print(f"🎮 GPU {i} ({gpu.name}): {gpu.load * 100:.1f}% | VRAM: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        else:
            print("⚠️ GPU no disponible o GPUtil no instalado.")

        print("-" * 40)
        time.sleep(2)

def monitor_resources(background=False):
    if background:
        thread = threading.Thread(target=show_resource_status, daemon=True)
        thread.start()
    else:
        show_resource_status()

# Uso directo desde terminal
if __name__ == "__main__":
    monitor_resources()
