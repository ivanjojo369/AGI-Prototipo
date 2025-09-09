from typing import Any, Dict, List, Optional

try:
    from utils.logger import Logger
except Exception:
    Logger = None  # type: ignore

from memory.unified_memory import UnifiedMemory

def _safe_log(logger: Any, level: str, msg: str) -> None:
    if logger is None:
        print(msg); return
    for name in (level, "log", "success", "info"):
        fn = getattr(logger, name, None)
        if callable(fn):
            try: fn(msg); return
            except Exception: pass
    print(msg)

class TaskPlanner:
    def __init__(self, memory: Optional[UnifiedMemory] = None):
        self.logger = Logger("TaskPlanner") if Logger else None  # type: ignore
        self.memory = memory or UnifiedMemory(data_dir="memory_store")
        _safe_log(self.logger, "info", "TaskPlanner inicializado correctamente.")

    def create_plan(self, tasks: List[str]) -> List[Dict[str, Any]]:
        _safe_log(self.logger, "info", f"Creando plan para {len(tasks or [])} tareas.")
        plan: List[Dict[str, Any]] = []
        for t in tasks or []:
            related = []
            try:
                related = self.memory.retrieve_relevant_memories(t, top_k=3)  # type: ignore[attr-defined]
            except Exception:
                related = []
            plan.append({"task": t, "status": "pending", "related_experiences": related})
        return plan

    def update_plan(self, plan: List[Dict[str, Any]], task_result: Any) -> List[Dict[str, Any]]:
        _safe_log(self.logger, "info", f"Actualizando plan con resultado: {task_result}")
        updated = list(plan or [])
        updated.append({"result": task_result})
        try:
            if hasattr(self.memory, "store_reflection"):
                self.memory.store_reflection("Resultado de tarea", task_result)  # type: ignore[attr-defined]
                _safe_log(self.logger, "success", "Resultado guardado en memoria.")
        except Exception as e:
            _safe_log(self.logger, "error", f"Error guardando reflexión: {e}")
        return updated

    def prioritize_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        _safe_log(self.logger, "info", "Priorizando tareas usando memoria histórica.")
        try:
            # Si hay clave 'priority', ordenar por ella (desc)
            if tasks and isinstance(tasks[0], dict) and "priority" in tasks[0]:
                return sorted(tasks, key=lambda x: x.get("priority", 0), reverse=True)
            # Si no, usar experiencias relacionadas (desc)
            return sorted(tasks or [], key=lambda x: len(x.get("related_experiences", [])), reverse=True)
        except Exception as e:
            _safe_log(self.logger, "error", f"Error priorizando tareas: {e}")
            return list(tasks or [])
