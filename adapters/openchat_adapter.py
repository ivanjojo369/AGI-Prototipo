from utils.logger import Logger
import os
from llama_cpp import Llama


class OpenChatAdapter:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")

        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,         # Usa GPU si está disponible
            n_ctx=2048,              # Tamaño del contexto
            seed=42,                 # Para resultados reproducibles
            f16_kv=True,             # Usa menos RAM
            logits_all=False,        # Solo el último token
            use_mlock=True,          # Evita que el modelo sea sacado de RAM
            verbose=True             # Muestra detalles de carga
        )

    def generar_respuesta(self, prompt):
        respuesta = self.model(
            prompt=prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stop=["<|endoftext|>", "</s>", "\n\n"]
        )
        return respuesta["choices"][0]["text"].strip()

    def get_logits(self):
        """
        Retorna los logits actuales del modelo.
        Nota: solo válido si 'logits_all=True' fue activado en Llama().
        """
        try:
            return self.model.get_logits()
        except AttributeError:
            Logger.info("⚠️ No se puede acceder a los logits. Asegúrate de haber usado logits_all=True en la configuración.")
            return None
