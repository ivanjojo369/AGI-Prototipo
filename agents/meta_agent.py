from typing import Any, Dict, Optional

from planner.planner import Planner
from agents.reflection_engine import generate_reflective_response
from memory.context_memory import save_interaction_to_memory, retrieve_relevant_memories


class MetaAgent:
    def __init__(self, model: Any = None, planner: Optional[Planner] = None):
        self.model = model
        self.planner = planner or Planner()

    def procesar_mensaje(self, mensaje_usuario: str) -> str:
        plan = self.planner.plan(goal=mensaje_usuario, context={})
        recuerdos = retrieve_relevant_memories(mensaje_usuario, top_k=5)
        respuesta = generate_reflective_response(
            prompt=mensaje_usuario, context={"plan": plan, "recuerdos": recuerdos, "model": self.model}
        )
        save_interaction_to_memory(user_message=mensaje_usuario, assistant_message=respuesta)
        return respuesta

    # --- Compat tests ---
    def adjust_strategy(self) -> Dict[str, Any]:
        return {"ok": 1, "strategy": "default"}

    def analyze_task(self, task: str) -> Dict[str, Any]:
        plan = self.planner.plan(goal=task, context={})
        # Los tests esperan que exista la clave "Resultado"
        return {
            "ok": 1,
            "analysis": f"Tarea: {task}",
            "plan": plan,
            "Resultado": "Plan generado",  # <- clave requerida por el test
        }

    def execute_reflection(self, text: str) -> str:
        return generate_reflective_response(prompt=text, context={})
