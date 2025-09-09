# adapters/ollama_adapter.py

import requests
from adapters.model_interface import ModelInterface

class OllamaAdapter(ModelInterface):
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.endpoint = "http://localhost:11434/api/generate"

    def generate_response(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(self.endpoint, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            raise Exception(f"Ollama API error: {response.text}")
