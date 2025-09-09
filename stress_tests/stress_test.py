import asyncio
from multi_agent_manager import MultiAgentManager

async def run_stress_test():
    manager = MultiAgentManager(
        model_path="models/openchat-3.5-1210.Q4_K_M.gguf", 
        num_agents=5
    )
    await manager.run_agents()

if __name__ == "__main__":
    asyncio.run(run_stress_test())
