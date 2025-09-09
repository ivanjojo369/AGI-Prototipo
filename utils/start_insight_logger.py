import subprocess
import os
import time

UTILS_DIR = "utils"

Logger.info("🚀 Iniciando sistema automático de insights...")

processes = []

try:
    processes.append(subprocess.Popen(["python", os.path.join(UTILS_DIR, "clipboard_watcher.py")]))
    Logger.info("📋 Clipboard Watcher iniciado.")

    time.sleep(2)  # esperar un poco antes de lanzar el otro proceso

    processes.append(subprocess.Popen(["python", os.path.join(UTILS_DIR, "sync_insights.py")]))
    Logger.info("🔄 Sync Insights iniciado.")

    # Mantener procesos corriendo
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    Logger.info("\n🛑 Deteniendo procesos...")
    for p in processes:
        p.terminate()

except Exception as e:
    Logger.info(f"⚠️ Error al iniciar Insight Logger: {e}")
    for p in processes:
        p.terminate()
