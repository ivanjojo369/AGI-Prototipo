import asyncio

class AGIAgent:
    def __init__(self, name="Agente"):
        self.name = name

    async def handle_message(self, message):
        """Simula el procesamiento del mensaje y devuelve la respuesta."""
        await asyncio.sleep(0.2)  # Simulación de tiempo de procesamiento
        return f"[{self.name}] Respuesta procesada para: {message}"
