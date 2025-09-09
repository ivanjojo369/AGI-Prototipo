# task_planner.py

class TaskPlanner:
    def __init__(self):
        self.tareas = []

    def analizar(self, mensaje: str, respuesta: str):
        """Detecta posibles tareas en el mensaje o respuesta"""
        if "recordar" in mensaje.lower():
            self.tareas.append(mensaje)
        # Puedes agregar reglas más complejas si deseas

    def obtener_tareas(self):
        return self.tareas
