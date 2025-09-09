import asyncio
import os
import json
from datetime import datetime
from agents.multi_agent_manager import MultiAgentManager


async def run_stress_test(num_messages: int = 5000, num_agents: int = 4):
    print(f"🚀 Iniciando stress test multi-agente con {num_agents} agentes y {num_messages} mensajes...")
    manager = MultiAgentManager(num_agents=num_agents)

    # Generar mensajes de prueba
    messages = [f"Mensaje de prueba {i}" for i in range(num_messages)]

    # Ejecutar procesamiento paralelo
    await manager.process_messages(messages)

    # Guardar reporte
    os.makedirs("logs/benchmarks/multi_agent", exist_ok=True)
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_agents": num_agents,
        "num_messages": num_messages,
        "status": "completed"
    }
    report_path = "logs/benchmarks/multi_agent/stress_multi_agent_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"📄 Reporte final guardado en {report_path}")


if __name__ == "__main__":
    asyncio.run(run_stress_test())
