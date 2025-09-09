import unittest
from memory.unified_memory import UnifiedMemory

class TestUnifiedMemory(unittest.TestCase):
    def setUp(self):
        self.memory = UnifiedMemory(data_dir="tests/temp_memory")

    def test_add_and_retrieve_interaction(self):
        self.memory.add_interaction("usuario", "Hola AGI")
        context = self.memory.get_context(limit=1)
        self.assertEqual(context[0][1], "Hola AGI")

    def test_store_and_retrieve_event(self):
        self.memory.store_event("prueba", "Evento de prueba")
        events = self.memory.get_recent_events(limit=1)
        self.assertEqual(events[0]["contenido"], "Evento de prueba")

    def test_vector_search(self):
        self.memory.add_to_vector_memory("Recuerdo importante")
        results = self.memory.search_vector_memory("importante")
        self.assertTrue(len(results) > 0)

    def test_reflections(self):
        self.memory.store_reflection("Reflexión de prueba")
        reflections = self.memory.retrieve_recent_reflections(limit=1)
        self.assertEqual(reflections[0]["contenido"], "Reflexión de prueba")

if __name__ == '__main__':
    unittest.main()
