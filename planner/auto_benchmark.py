import subprocess
import datetime
import json
import os

LOG_DIR = "logs/benchmarks/auto/"
os.makedirs(LOG_DIR, exist_ok=True)

def run_benchmark():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(f"🚀 Ejecutando benchmark automático... [{timestamp}]")

    # Ejecutar benchmark de AGI
    result = subprocess.run(["python", "benchmark_agi.py"], capture_output=True, text=True)

    output = {
        "timestamp": timestamp,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }

    filename = os.path.join(LOG_DIR, f"benchmark_{timestamp}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"✅ Benchmark completado. Resultados guardados en: {filename}")

if __name__ == "__main__":
    run_benchmark()
