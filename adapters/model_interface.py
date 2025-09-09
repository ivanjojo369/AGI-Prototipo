# adapters/model_interface.py
import random
from typing import Optional

class DummyModel:
    """
    Modelo de fallback cuando no hay un LLM real disponible.
    Ahora responde de forma más variada y usa contexto si se le pasa.
    """

    def chat(self, prompt: str, contexto: Optional[str] = None) -> str:
        # Respuestas rápidas y adaptadas
        base_respuestas = [
            "Interesante lo que mencionas.",
            "Puedo entenderlo, cuéntame más.",
            "Eso suena bien, ¿quieres que profundicemos?",
            "Comprendo. ¿Qué más te gustaría explorar?"
        ]

        # Si hay contexto, úsalo
        if contexto:
            return f"Tengo presente que {contexto.lower()}. En cuanto a lo que dices: {random.choice(base_respuestas)}"

        # Si no hay contexto, usa la respuesta base
        return random.choice(base_respuestas)


# Modelo real (ejemplo de integración futura)
class RealModel:
    def __init__(self, client):
        self.client = client  # Aquí podrías pasar openai, llama_cpp, etc.

    def chat(self, prompt: str) -> str:
        # Ejemplo: integración con modelo real
        # response = self.client.chat.completions.create(
        #     model="nombre_modelo",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return response.choices[0].message["content"].strip()
        return "Aquí respondería el modelo real."
    

# Función para seleccionar modelo
def load_model(use_real_model=False, client=None):
    if use_real_model and client:
        return RealModel(client)
    return DummyModel()
