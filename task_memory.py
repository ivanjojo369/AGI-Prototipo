import json
import os
from datetime import datetime

TASKS_FILE = "data/tareas.json"  # Asegúrate de que exista la carpeta `data`

def crear_archivo_si_no_existe():
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

def cargar_tareas():
    crear_archivo_si_no_existe()
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def guardar_tareas(tareas):
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tareas, f, indent=2, ensure_ascii=False)

def agregar_tarea(descripcion, prioridad="media"):
    tareas = cargar_tareas()
    nueva = {
        "descripcion": descripcion,
        "estado": "pendiente",
        "prioridad": prioridad,
        "fecha_creacion": datetime.now().isoformat(),
        "fecha_completada": None
    }
    tareas.append(nueva)
    guardar_tareas(tareas)
    return nueva

def marcar_completada(indice):
    tareas = cargar_tareas()
    if 0 <= indice < len(tareas):
        tareas[indice]["estado"] = "hecha"
        tareas[indice]["fecha_completada"] = datetime.now().isoformat()
        guardar_tareas(tareas)
        return True
    return False

def obtener_tareas_pendientes():
    return [t for t in cargar_tareas() if t["estado"] == "pendiente"]

def obtener_tareas_por_prioridad(prioridad):
    return [t for t in cargar_tareas() if t["prioridad"] == prioridad and t["estado"] == "pendiente"]
