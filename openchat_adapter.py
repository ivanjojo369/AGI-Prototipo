from llama_cpp import Llama

class OpenChatAdapter:
    def __init__(self, model_path):
        self.model = Llama(model_path=model_path, n_ctx=4096, verbose=True)

    def generar_respuesta(self, mensaje, historial):
        prompt = self._formatear_prompt(mensaje, historial)
        respuesta = self.model(prompt, max_tokens=512, stop=["</s>"])
        texto = respuesta["choices"][0]["text"].strip()
        return texto

    def _formatear_prompt(self, mensaje, historial):
        contexto = ""
        for entrada, salida in historial:
            contexto += f"Usuario: {entrada}\nAsistente: {salida}\n"
        contexto += f"Usuario: {mensaje}\nAsistente:"
        return contexto
