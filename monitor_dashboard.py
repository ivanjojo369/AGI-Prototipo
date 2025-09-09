import os
import time
import json
import pathlib
import threading
import psutil
from plyer import notification

try:
    import GPUtil
except ImportError:
    GPUtil = None

try:
    import winsound
    BEEP_ENABLED = True
except ImportError:
    BEEP_ENABLED = False

BASE_DIR = pathlib.Path(__file__).resolve().parent
MEMORY_PATH = BASE_DIR / "data" / "memory" / "vector_meta.json"

def beep_alert():
    if BEEP_ENABLED:
        winsound.Beep(1000, 500)

def notify(title, message):
    notification.notify(
        title=title,
        message=message,
        timeout=3
    )

def get_agi_memory_size():
    if not MEMORY_PATH.exists():
        return 0
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return len(data)
    except Exception:
        return -1  # Corrupt or empty

def monitor():
    print("🔍 Iniciando monitor del sistema AGI...\n")
    print(f"{'Tiempo':<8} | CPU % | RAM % | GPU % | VRAM (MB) | Recuerdos AGI")
    print("-" * 65)

    while True:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        agi_mem = get_agi_memory_size()

        gpu_usage = "N/A"
        vram_usage = "N/A"

        if GPUtil:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
                vram_usage = gpus[0].memoryUsed

        now = time.strftime("%H:%M:%S")

        print(f"{now:<8} | {cpu:>5.1f} | {ram:>5.1f} | "
              f"{gpu_usage if isinstance(gpu_usage, str) else f'{gpu_usage:>5.1f}'} | "
              f"{vram_usage if isinstance(vram_usage, str) else f'{vram_usage:>9.1f}'} | "
              f"{agi_mem if agi_mem != -1 else '⚠️ Error'}")

        if isinstance(gpu_usage, float) and gpu_usage > 80:
            beep_alert()
            notify("🚨 GPU Alta", f"Uso de GPU: {gpu_usage:.1f}%")

        if ram > 85:
            beep_alert()
            notify("🚨 RAM Alta", f"Uso de RAM: {ram:.1f}%")

        time.sleep(1.5)

if __name__ == "__main__":
    monitor()
