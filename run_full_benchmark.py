import os
import json
import subprocess
import datetime
import winsound
from plyer import notification

# Paths
BENCHMARK_GPU = "benchmarks/gpu_monitor.py"
GPU_PLOT = "planner/gpu_latency_plot.py"
STRESS_TEST = "stress_tests/stress_memory_test.py"
OUTPUT_JSON = "logs/benchmarks/final_report.json"
OUTPUT_PNG = "logs/plots/final_benchmark.png"

def run_command(command):
    """Ejecuta un comando de Python y retorna la salida."""
    print(f"▶ Ejecutando: {command}")
    result = subprocess.run(
        ["python", command],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    return result.stdout

def main():
    print("🚀 Iniciando ejecución unificada...\n")

    # 1️⃣ Benchmark GPU
    gpu_output = run_command(BENCHMARK_GPU)

    # 2️⃣ Generar gráfico VRAM
    plot_output = run_command(GPU_PLOT)

    # 3️⃣ Stress Test multi-agente
    stress_output = run_command(STRESS_TEST)

    # 4️⃣ Guardar reporte JSON
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = {
        "timestamp": timestamp,
        "gpu_benchmark": gpu_output,
        "gpu_plot": plot_output,
        "stress_test": stress_output
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"✅ Reporte guardado en: {OUTPUT_JSON}")

    # 5️⃣ Copiar gráfico generado a final_benchmark.png (si existe)
    source_plot = "logs/plots/gpu_vram_usage.png"
    if os.path.exists(source_plot):
        import shutil
        shutil.copy(source_plot, OUTPUT_PNG)
        print(f"✅ Gráfico final guardado en: {OUTPUT_PNG}")

    # 6️⃣ Alerta sonora y notificación
    winsound.Beep(1000, 700)  # Sonido (1 segundo)
    notification.notify(
        title="✅ Benchmark Completado",
        message="La ejecución unificada ha finalizado correctamente.",
        timeout=10
    )

    print("\n🔔 Benchmark completo. Revisa el reporte y gráfico final.")

if __name__ == "__main__":
    main()
