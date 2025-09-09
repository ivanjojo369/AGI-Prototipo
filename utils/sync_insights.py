from utils.logger import Logger
import os
import time
import json
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from win10toast import ToastNotifier

INPUT_DIR = "sessions/inputs"
PROCESSED_DIR = "sessions/processed"

class InsightHandler(FileSystemEventHandler):
    def __init__(self, notifier):
        self.notifier = notifier

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".json"):
            return
        
        file_name = os.path.basename(event.src_path)
        Logger.info(f"📄 Nuevo archivo detectado: {file_name}")

        try:
            # Leer contenido
            with open(event.src_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            Logger.info(f"📑 Contenido del insight: {data}")

            # Mover archivo a processed
            dest_path = os.path.join(PROCESSED_DIR, file_name)
            shutil.move(event.src_path, dest_path)
            Logger.info(f"📂 Movido a: {dest_path}")

            # Notificación toast
            self.notifier.show_toast(
                "Insight Procesado",
                f"Archivo {file_name} movido a processed",
                duration=3,
                threaded=True
            )

        except Exception as e:
            Logger.info(f"❌ Error procesando {file_name}: {e}")

def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    notifier = ToastNotifier()
    Logger.info("🚀 Iniciando sincronización de insights...")
    Logger.info(f"📂 Monitoreando carpeta: {INPUT_DIR}")

    event_handler = InsightHandler(notifier)
    observer = Observer()
    observer.schedule(event_handler, INPUT_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

if __name__ == "__main__":
    main()
