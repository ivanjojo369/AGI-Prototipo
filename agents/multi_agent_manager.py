import asyncio
import psutil
import GPUtil
import numpy as np
from adapters.agi_agent import AGIAgent
from memory.unified_memory import UnifiedMemory
from datetime import datetime

class MultiAgentManager:
    def __init__(self, num_agents=4, memory=None):
        self.num_agents = num_agents
        self.agents = [AGIAgent(name=f"Agente_{i}") for i in range(num_agents)]
        self.memory = memory if memory else UnifiedMemory()
        self.gpu_count = len(GPUtil.getGPUs())
        self.cpu_count = psutil.cpu_count(logical=True)

        print(f"🚀 Multi-Agente inicializado con {self.num_agents} agentes")
        print(f"🧠 Memorias cargadas: {self.memory.get_memory_count()}")

    async def process_messages(self, messages):
        """Procesa mensajes distribuyéndolos entre los agentes y balanceando GPU/CPU."""
        tasks = []
        for i, message in enumerate(messages):
            agent = self.agents[i % self.num_agents]
            tasks.append(self._handle_with_balance(agent, message))
        results = await asyncio.gather(*tasks)
        return results

    async def _handle_with_balance(self, agent, message):
        """Selecciona GPU o CPU según carga y procesa el mensaje."""
        gpu_id = self._select_least_used_gpu()
        cpu_load = psutil.cpu_percent(interval=0.1)
        gpu_load = self._get_gpu_load(gpu_id)

        if gpu_load < 70:
            device = f"GPU {gpu_id}" if self.gpu_count > 0 else "CPU"
        else:
            device = "CPU"

        result = await agent.handle_message(f"[{device}] {message}")
        self._store_result_in_memory(result)
        return result

    def _store_result_in_memory(self, result):
        """Genera un embedding simple y almacena el resultado en la memoria unificada."""
        embedding = np.random.rand(self.memory.embedding_size).astype(np.float32)
        self.memory.add_memory(result, embedding)

    def _select_least_used_gpu(self):
        """Devuelve el ID de la GPU menos utilizada."""
        if self.gpu_count == 0:
            return None
        gpus = GPUtil.getGPUs()
        min_gpu = min(gpus, key=lambda gpu: gpu.load)
        return min_gpu.id

    def _get_gpu_load(self, gpu_id):
        """Obtiene la carga de la GPU especificada."""
        if gpu_id is None:
            return 100  # forzar CPU
        gpu = next((g for g in GPUtil.getGPUs() if g.id == gpu_id), None)
        return gpu.load * 100 if gpu else 100

    async def stress_test(self, num_messages=5000):
        """Ejecuta un test de estrés multi-agente para validar rendimiento."""
        print(f"🔥 Iniciando stress test multi-agente con {self.num_agents} agentes y {num_messages} mensajes...")
        messages = [f"Mensaje {i}" for i in range(num_messages)]
        start_time = datetime.now()
        results = await self.process_messages(messages)
        duration = (datetime.now() - start_time).total_seconds()
        print(f"✅ Stress test completado en {duration:.2f}s - Total mensajes procesados: {len(results)}")
        return results
