from utils.logger import Logger
import json
import os
import re

class TaskPlanner:
    def __init__(self, file_path="tasks.json"):
        self.tasks = []
        self.file_path = file_path
        self.load()
        self.save()  # Fuerza creación del archivo al iniciar

    def detect_task(self, user_input):
        if re.search(r"\b(crea|haz|programa|construye)\b", user_input, re.IGNORECASE):
            return f"Tarea de creación detectada: '{user_input}'"
        if re.search(r"\b(busca|investiga|encuentra)\b", user_input, re.IGNORECASE):
            return f"Tarea de búsqueda detectada: '{user_input}'"
        if re.search(r"\b(explica|define|cuenta|describe)\b", user_input, re.IGNORECASE):
            return f"Tarea de explicación detectada: '{user_input}'"
        return None

    def register_task(self, task):
        self.tasks.append(task)
        self.save()

    def get_tasks(self):
        return self.tasks

    def clear(self):
        self.tasks = []
        self.save()
        Logger.info(f"[✓] Tareas limpiadas y archivo {self.file_path} actualizado.")

    def save(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.tasks, f, indent=2, ensure_ascii=False)
            Logger.info(f"[✓] Tareas guardadas en {self.file_path}")
        except Exception as e:
            Logger.info(f"[!] Error al guardar tareas: {e}")

    def load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.tasks = json.load(f)
                Logger.info(f"[✓] Tareas cargadas desde {self.file_path}")
            except Exception as e:
                Logger.info(f"[!] Error al cargar tareas: {e}")
        else:
            Logger.info(f"[ℹ️] {self.file_path} no existe. Se creará al guardar.")
