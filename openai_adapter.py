from adapters.model_interface import ModelInterface
import openai

class OpenAIAdapter(ModelInterface):
    def __init__(self, api_key: str, model_name="gpt-4o"):
        openai.api_key = api_key
        self.model_name = model_name

    def generar_respuesta(self, prompt: str) -> str:
        respuesta = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return respuesta.choices[0].message.content.strip()
