import unittest
from agents.meta_agent import MetaAgent

class TestMetaAgent(unittest.TestCase):
    def setUp(self):
        self.meta_agent = MetaAgent()

    def test_analyze_task(self):
        result = self.meta_agent.analyze_task("Tarea de prueba")
        self.assertIn("Resultado", result)

    def test_adjust_strategy(self):
        try:
            self.meta_agent.adjust_strategy()
        except Exception:
            self.fail("adjust_strategy lanzó una excepción")

    def test_execute_reflection(self):
        try:
            self.meta_agent.execute_reflection("Reflexión de prueba")
        except Exception:
            self.fail("execute_reflection lanzó una excepción")

if __name__ == '__main__':
    unittest.main()
