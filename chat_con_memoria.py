from utils.logger import Logger
import json
from llama_cpp import Llama

# Ruta al modelo
modelo_path = "models/openchat-3.5-1210.Q4_K_M.gguf"

# Cargar modelo
llm = Llama(
    model_path=modelo_path,
    n_ctx=2048,
    n_threads=4,
    verbose=True
)

# Ruta del archivo de memoria
archivo_memoria = "memoria.json"

# Cargar o iniciar memoria
try:
    with open(archivo_memoria, "r", encoding="utf-8") as f:
        memoria = json.load(f)
except FileNotFoundError:
    memoria = []

# Función para construir el prompt contextual
def construir_prompt(memoria, nueva_entrada):
    prompt = ""
    for entrada in memoria[-5:]:  # Limita a las últimas 5 interacciones
        prompt += f"Usuario: {entrada['usuario']}\nAsistente: {entrada['asistente']}\n"
    prompt += f"Usuario: {nueva_entrada}\nAsistente:"
    return prompt

# Loop de conversación
while True:
    entrada_usuario = input("Tú: ").strip()
    if entrada_usuario.lower() in ["salir", "exit", "quit"]:
        Logger.info("Asistente: ¡Hasta luego!")
        break

    prompt = construir_prompt(memoria, entrada_usuario)

    respuesta = llm(prompt, max_tokens=256, stop=["Usuario:", "</s>"])
    texto_respuesta = respuesta["choices"][0]["text"].strip()

    Logger.info(f"Asistente: {texto_respuesta}")

    # Guardar en memoria
    memoria.append({"usuario": entrada_usuario, "asistente": texto_respuesta})
    with open(archivo_memoria, "w", encoding="utf-8") as f:
        json.dump(memoria, f, indent=2, ensure_ascii=False)
