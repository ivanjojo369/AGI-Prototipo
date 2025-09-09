import unittest
from planner.task_planner import TaskPlanner

class TestTaskPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = TaskPlanner()

    def test_create_plan(self):
        plan = self.planner.create_plan(["Tarea A", "Tarea B"])
        self.assertEqual(len(plan), 2)

    def test_update_plan(self):
        plan = self.planner.create_plan(["Tarea A"])
        updated = self.planner.update_plan(plan, "Resultado")
        self.assertTrue(any("result" in step for step in updated))

    def test_prioritize_tasks(self):
        plan = [
            {"task": "Tarea A", "priority": 1},
            {"task": "Tarea B", "priority": 5}
        ]
        prioritized = self.planner.prioritize_tasks(plan)
        self.assertEqual(prioritized[0]["task"], "Tarea B")

if __name__ == '__main__':
    unittest.main()
