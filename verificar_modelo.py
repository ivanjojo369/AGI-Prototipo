from utils.logger import Logger
from llama_cpp import Llama

# Ruta al modelo (ajústala si renombraste el archivo)
modelo_path = "models/openchat-3.5-1210.Q4_K_M.gguf"

# Cargar el modelo
llm = Llama(
    model_path=modelo_path,
    n_ctx=2048,
    n_threads=4,   # Ajusta este valor según tu CPU (usa menos si se satura)
    verbose=True
)

# Prompt de prueba
prompt = "Hola, ¿quién eres?"

# Ejecutar el modelo
respuesta = llm(prompt, max_tokens=256, stop=["</s>"])

# Mostrar resultado
Logger.info("\nRespuesta del modelo:")
Logger.info(respuesta["choices"][0]["text"])
