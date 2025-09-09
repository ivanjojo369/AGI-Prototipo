import unittest
from agents.meta_agent import MetaAgent
from planner.task_planner import TaskPlanner
from memory.unified_memory import UnifiedMemory

class TestIntegrationCore(unittest.TestCase):
    def setUp(self):
        self.memory = UnifiedMemory(data_dir="tests/temp_memory_integration")
        self.meta_agent = MetaAgent()
        self.task_planner = TaskPlanner()
        self.tasks = ["Analizar logs", "Optimizar plan", "Actualizar memoria"]

    def test_core_workflow(self):
        # Crear plan
        plan = self.task_planner.create_plan(self.tasks)
        self.assertEqual(len(plan), len(self.tasks), "El plan no contiene todas las tareas")

        # Priorizar
        prioritized_plan = self.task_planner.prioritize_tasks(plan)
        self.assertTrue(isinstance(prioritized_plan, list), "Priorizar tareas falló")

        # Analizar tareas y guardar reflexiones
        for task in prioritized_plan:
            result = self.meta_agent.analyze_task(task['task'])
            self.memory.store_reflection(f"Resultado de {task['task']}: {result}")
            updated_plan = self.task_planner.update_plan(prioritized_plan, result)
            self.assertTrue(any("result" in step for step in updated_plan), "La actualización del plan falló")

        # Comprobar reflexiones guardadas
        reflections = self.memory.retrieve_recent_reflections(limit=3)
        self.assertGreaterEqual(len(reflections), 1, "No se guardaron reflexiones en memoria")

        # Estado de memoria
        status = self.memory.get_status()
        self.assertIn("reflections_stored", status, "Estado de memoria no contiene reflexiones")
        self.assertGreater(status["reflections_stored"], 0, "No hay reflexiones almacenadas")

if __name__ == '__main__':
    unittest.main()
