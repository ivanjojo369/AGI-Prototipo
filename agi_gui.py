# agi_gui.py (versión autónoma sin OpenAI)

import gradio as gr
from agi_interface import AGIInterface
from adapters.llama_cpp_adapter import LlamaCppAdapter
import json

# Inicializar AGI local
agi = AGIInterface()

# Cargar modelo local (GGUF) mediante llama_cpp_adapter
modelo_local = LlamaCppAdapter(
    model_path="models/nous-hermes-2-mixtral-8x7b-dpo.gguf",  # Ajusta si usas otro
    n_ctx=4096,
    temperature=0.7
)

def generar_analisis_local(contexto_json):
    """
    Usa el modelo local para generar Insight y Sugerencia sin depender de OpenAI.
    """
    prompt = f"""
    Analiza el comportamiento del sistema AGI y sus reflexiones.
    Datos del sistema:
    {json.dumps(contexto_json, indent=2, ensure_ascii=False)}

    Responde solo en formato JSON:
    {{
      "insight": "Breve observación sobre el estado y reflexiones del AGI.",
      "suggestion": "Recomendación de mejora o próximo paso para acelerar el desarrollo del proyecto."
    }}
    """
    respuesta = modelo_local.generate_text(prompt, max_tokens=300)

    try:
        datos = json.loads(respuesta)
    except:
        # Si el modelo no devuelve JSON válido, devolvemos fallback
        datos = {
            "insight": "No se pudo parsear JSON, revisar prompt.",
            "suggestion": "Optimizar el analizador local para mayor precisión."
        }

    return datos

def responder_agi(mensaje_usuario, historial):
    # Respuesta y contexto desde AGI local
    respuesta_local = agi.procesar_comando(mensaje_usuario)

    # Extraer contexto
    try:
        partes = respuesta_local.split("📤 Contexto enviado al GPT:")
        contexto = partes[1].strip() if len(partes) > 1 else "{}"
        contexto_json = json.loads(contexto.replace("'", '"'))
    except Exception:
        contexto_json = {}

    # Generar Insight y Sugerencia usando modelo local
    analisis = generar_analisis_local(contexto_json)
    insight = analisis.get("insight", "Sin observaciones")
    sugerencia = analisis.get("suggestion", "Sin sugerencia")

    salida = (
        f"🤖 **Respuesta Local:**\n{respuesta_local}\n\n"
        f"🧠 **Insight (Modelo Local):** {insight}\n"
        f"🚀 **Sugerencia:** {sugerencia}"
    )

    historial.append((mensaje_usuario, salida))
    return "", historial

# Construcción de la GUI
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("## 🚀 AGI Accelerator Hub - Interfaz Autónoma")

    chatbot = gr.Chatbot(label="AGI Local")
    entrada = gr.Textbox(placeholder="Escribe un mensaje...", label="Entrada del Usuario")
    enviar = gr.Button("Enviar")

    estado = gr.State([])

    enviar.click(fn=responder_agi, inputs=[entrada, estado], outputs=[entrada, chatbot])

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860)
