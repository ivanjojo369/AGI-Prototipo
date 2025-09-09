import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

GPU_LOG_DIR = "logs/benchmarks/gpu"
PLOT_DIR = "logs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def load_gpu_benchmarks():
    benchmarks = []
    for file in os.listdir(GPU_LOG_DIR):
        if file.endswith(".json"):
            with open(os.path.join(GPU_LOG_DIR, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                benchmarks.extend(data)
    return sorted(benchmarks, key=lambda x: x["timestamp"])

def plot_gpu_latency(benchmarks):
    timestamps = [datetime.fromisoformat(b["timestamp"]) for b in benchmarks]
    latencies = [b["latency_s"] for b in benchmarks]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, latencies, label="Latencia de Inferencia (s)", marker='o', color="green")
    plt.xlabel("Tiempo")
    plt.ylabel("Latencia (segundos)")
    plt.title("Evolución de Latencia en GPU")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "gpu_latency.png"))
    plt.close()

def plot_gpu_vram(benchmarks):
    timestamps = [datetime.fromisoformat(b["timestamp"]) for b in benchmarks]
    vram_usage = [b["vram_usage_mb"] for b in benchmarks]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, vram_usage, label="Uso de VRAM (MB)", marker='o', color="blue")
    plt.xlabel("Tiempo")
    plt.ylabel("VRAM (MB)")
    plt.title("Consumo de VRAM durante inferencia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "gpu_vram_usage.png"))
    plt.close()

def generate_gpu_report():
    benchmarks = load_gpu_benchmarks()
    if not benchmarks:
        print("⚠️ No se encontraron benchmarks GPU históricos.")
        return
    plot_gpu_latency(benchmarks)
    plot_gpu_vram(benchmarks)
    print(f"✅ Gráficos GPU generados en {PLOT_DIR}")

if __name__ == "__main__":
    generate_gpu_report()
