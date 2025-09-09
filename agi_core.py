from utils.logger import Logger
import json

class AGIInterface:
    def __init__(self, model_adapter, memory, planner):
        self.model = model_adapter
        self.memory = memory
        self.planner = planner

    def resumir_texto(self, texto_largo):
        prompt = f"""
        Resume el siguiente texto en párrafos breves y claros, sin perder la idea principal:

        {texto_largo}
        """
        return self.model.generate(prompt)

    def planear_meta(self, meta):
        prompt = f"""
        Divide la siguiente meta en una lista de pasos concretos que una IA pueda ejecutar o ayudar:

        Meta: {meta}
        """
        return self.model.generate(prompt)

    def start(self):
        Logger.info("\n🤖 AGI lista. Escribe algo o escribe 'salir' para terminar.\n")

        while True:
            user_input = input("Tú: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["salir", "exit"]:
                Logger.info("AGI: ¡Hasta luego!")
                break

            # Comandos especiales
            if user_input.lower() == "ver tareas":
                tasks = self.planner.get_tasks()
                Logger.info("AGI:\nÚltimas tareas registradas:")
                for i, t in enumerate(tasks[-5:], 1):
                    Logger.info(f"{i}. {t}")
                continue

            if user_input.lower() in ["memoria", "¿qué sabes de mí?"]:
                context = self.memory.get_context()
                Logger.info("AGI:\nEsto es lo que recuerdo:")
                for turn in context[-3:]:
                    Logger.info(f"Tú: {turn['user']}\nAGI: {turn['agi']}")
                continue

            if user_input.lower() == "reset tareas":
                self.planner.clear()
                Logger.info("AGI: Tareas borradas.")
                continue

            if user_input.lower() == "reset memoria":
                self.memory.history = []
                self.memory.save()
                Logger.info("AGI: Memoria borrada.")
                continue

            if user_input.lower().startswith("resume:"):
                texto_largo = user_input[len("resume:"):].strip()
                resumen = self.resumir_texto(texto_largo)
                Logger.info("AGI:\nResumen generado:\n", resumen)
                continue

            if user_input.lower().startswith("meta:"):
                objetivo = user_input[len("meta:"):].strip()
                plan = self.planear_meta(objetivo)
                Logger.info("AGI:\nAquí tienes un plan para lograrlo:\n", plan)
                self.planner.register_task(f"Meta: {objetivo}\nPlan:\n{plan}")
                continue

            # ✅ CONTEXTO LIMITADO PARA VELOCIDAD
            context = self.memory.get_context()[-3:]  # últimas 3 interacciones
            prompt = ""
            for turn in context:
                prompt += f"Tú: {turn['user']}\nAGI: {turn['agi']}\n"
            prompt += f"Tú: {user_input}\nAGI:"

            Logger.info("AGI: (procesando...)")  # alerta rápida
            response = self.model.generate(prompt)
            Logger.info("AGI:", response)

            self.memory.add_interaction(user_input, response)

            task = self.planner.detect_task(user_input)
            if task:
                self.planner.register_task(task)
                Logger.info(f"[✓] Tarea detectada: {task}")
