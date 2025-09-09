from adapters.model_interface import ModelInterface
import requests

class OllamaAdapter(ModelInterface):
    def __init__(self, model_name="llama3", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def generar_respuesta(self, prompt: str) -> str:
        response = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model_name, "prompt": prompt}
        )
        return response.json().get("response", "").strip()
