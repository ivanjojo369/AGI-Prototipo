from utils.logger import Logger
import openai

# Redirige el cliente OpenAI a tu servidor local
openai.api_base = "http://localhost:7860/v1"
openai.api_key = "tu-clave-falsa"  # No se valida, puede ser cualquier texto

# Llamada estilo ChatGPT
response = openai.ChatCompletion.create(
    model="openchat-3.5-1210.Q4_K_M.gguf",
    messages=[
        {"role": "user", "content": "¿Qué es la conciencia?"}
    ],
    temperature=0.7,
    max_tokens=300
)

Logger.info(response.choices[0].message["content"])
