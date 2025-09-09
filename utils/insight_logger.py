import os
import json
import time

class InsightLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, message):
        """Agrega un mensaje simple al log"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def log_from_file(self, file_path):
        """Lee un JSON y lo agrega al log"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            timestamp = data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            source = data.get("source", "Desconocido")
            insight = data.get("insight", "No content")

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] ({source}) {insight}\n")

            Logger.info(f"📝 Insight registrado en log desde archivo: {file_path}")

        except Exception as e:
            Logger.info(f"❌ Error registrando insight desde archivo {file_path}: {e}")
