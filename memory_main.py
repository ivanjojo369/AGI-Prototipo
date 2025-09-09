from utils.logger import Logger
import json
import os

class ContextMemory:
    def __init__(self, max_length=10, file_path="memory.json"):
        self.history = []
        self.max_length = max_length
        self.file_path = file_path
        self.load()
        self.save()  # Fuerza la creación del archivo al iniciar

    def add_interaction(self, user_input, response):
        self.history.append({"user": user_input, "agi": response})
        self.history = self.history[-self.max_length:]
        self.save()

    def get_context(self):
        return self.history

    def save(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            Logger.info(f"[✓] Memoria guardada en {self.file_path}")
        except Exception as e:
            Logger.info(f"[!] Error al guardar memoria: {e}")

    def load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
                Logger.info(f"[✓] Memoria cargada desde {self.file_path}")
            except Exception as e:
                Logger.info(f"[!] Error al cargar memoria: {e}")
        else:
            Logger.info(f"[ℹ️] {self.file_path} no existe. Se creará al guardar.")
