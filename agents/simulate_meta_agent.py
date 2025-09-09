from utils.logger import Logger
# simulate_meta_agent.py

import random
import time
from agents.meta_agent import MetaAgent

class MetaAgentSimulator:
    def __init__(self):
        self.meta_agent = MetaAgent()
        self.historial_resultados = []

    def simular_interaccion(self, num_interacciones=20):
        """Simula varias interacciones con resultados aleatorios."""
        for i in range(num_interacciones):
            entrada = f"Tarea simulada {i+1}"

            # Simular resultado con probabilidad de éxito del 70%
            exito = random.random() < 0.7
            resultado = "Completado correctamente" if exito else "Error: Fallo de ejecución"

            Logger.info(f"\n🧠 Interacción {i+1}")
            Logger.info(f"Entrada: {entrada}")
            Logger.info(f"Resultado simulado: {resultado}")

            reflexion_interna = self.meta_agent.reflexionar_sobre_entrada_usuario(entrada)
            self.meta_agent.reflexionar_sobre_tarea(entrada, resultado)

            self.historial_resultados.append({
                "entrada": entrada,
                "resultado": resultado,
                "exito": exito
            })

            Logger.info(f"🤖 Reflexión: {reflexion_interna}")
            Logger.info(f"📈 Puntuación acumulada: {self.meta_agent.score}")
            Logger.info(f"⚖️ Estrategia: {self.meta_agent.strategy_weights}")
            Logger.info("-" * 60)

            time.sleep(0.2)  # pequeña pausa para visualización

    def resumen_aprendizaje(self):
        """Muestra la tasa de éxito y reflexiones aprendidas."""
        reflexion_final = self.meta_agent.aprender_de_experiencias()
        Logger.info("\n📊 Resumen de simulación:")
        Logger.info(f"- Total interacciones: {len(self.historial_resultados)}")
        Logger.info(f"- Éxitos: {sum(r['exito'] for r in self.historial_resultados)}")
        Logger.info(f"- Fracasos: {sum(not r['exito'] for r in self.historial_resultados)}")
        Logger.info(f"- Estrategia final: {self.meta_agent.strategy_weights}")
        Logger.info(f"- Reflexión final: {reflexion_final}")

if __name__ == "__main__":
    sim = MetaAgentSimulator()
    sim.simular_interaccion(num_interacciones=30)
    sim.resumen_aprendizaje()
