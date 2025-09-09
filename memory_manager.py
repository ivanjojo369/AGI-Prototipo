import os
import json
from datetime import datetime

SESSIONS_FOLDER = "sessions"

def ensure_sessions_dir():
    if not os.path.exists(SESSIONS_FOLDER):
        os.makedirs(SESSIONS_FOLDER)

def nueva_memoria():
    ensure_sessions_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archivo = os.path.join(SESSIONS_FOLDER, f"session_{timestamp}.json")
    with open(archivo, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    return archivo

def load_memory(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def guardar_memoria(path, memoria):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memoria, f, ensure_ascii=False, indent=2)

def list_sessions():
    ensure_sessions_dir()
    return [f for f in os.listdir(SESSIONS_FOLDER) if f.endswith(".json")]
