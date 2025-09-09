# adapters/openai_adapter.py

import openai
from adapters.model_interface import ModelInterface

class OpenAIAdapter(ModelInterface):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
