# context_memory.py

class ContextMemory:
    def __init__(self):
        self.historial = []

    def recuperar(self) -> str:
        """Devuelve el contexto acumulado como texto plano"""
        return "\n".join(self.historial[-5:])  # Puedes ajustar la longitud del contexto

    def actualizar(self, entrada_usuario: str, respuesta_agente: str):
        """Agrega la nueva interacción al historial"""
        self.historial.append(f"Usuario: {entrada_usuario}")
        self.historial.append(f"Asistente: {respuesta_agente}")
