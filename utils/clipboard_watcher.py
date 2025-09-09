from utils.logger import Logger
import os
import time
import json
import pyperclip
from datetime import datetime

# Ruta a la carpeta de inputs
INPUT_DIR = os.path.join("sessions", "inputs")
os.makedirs(INPUT_DIR, exist_ok=True)

last_clipboard_content = ""

Logger.info("📋 Clipboard Watcher iniciado correctamente...")
Logger.info(f"📂 Guardando archivos en: {INPUT_DIR}")

while True:
    try:
        # Lee contenido del portapapeles
        clipboard_content = pyperclip.paste().strip()

        if clipboard_content and clipboard_content != last_clipboard_content:
            last_clipboard_content = clipboard_content

            # Crear nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"insight_{timestamp}.json"
            file_path = os.path.join(INPUT_DIR, file_name)

            # Guardar contenido en JSON
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({"timestamp": timestamp, "content": clipboard_content}, f, indent=4, ensure_ascii=False)

            Logger.info(f"✅ Insight detectado y guardado: {file_path}")

        time.sleep(1)  # espera 1 segundo antes de volver a leer

    except Exception as e:
        Logger.info(f"⚠️ Error en clipboard_watcher: {e}")
        time.sleep(2)
