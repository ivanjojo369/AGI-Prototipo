from utils.logger import Logger
import openai

# Configura la API para usar tu servidor local
openai.api_base = "http://localhost:7860/v1"
openai.api_key = "sk-fake"  # Puede ser cualquier string

# Mensaje para el modelo
response = openai.ChatCompletion.create(
    model="openchat-3.5-1210.Q4_K_M.gguf",
    messages=[
        {"role": "user", "content": "¿Qué es la conciencia?"}
    ],
    temperature=0.7,
    max_tokens=300
)

# Muestra la respuesta
Logger.info("\n🧠 Respuesta del modelo:\n")
Logger.info(response.choices[0].message["content"])
