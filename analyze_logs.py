import os
import re
import statistics
import matplotlib.pyplot as plt

LOG_DIR = "logs"
PLOTS_DIR = os.path.join(LOG_DIR, "plots")

def parse_log_for_graphs(file_path):
    """Extrae métricas detalladas para gráficos."""
    timestamps = []
    cpu_usage = []
    ram_usage = []
    gpu_usage = []
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            cpu_match = re.search(r"CPU:\s+(\d+(?:\.\d+)?)%", line)
            ram_match = re.search(r"RAM:\s+(\d+(?:\.\d+)?)%", line)
            gpu_match = re.search(r"Uso:\s+(\d+)%", line)

            if cpu_match and ram_match:
                timestamps.append(i)
                cpu_usage.append(float(cpu_match.group(1)))
                ram_usage.append(float(ram_match.group(1)))
                gpu_usage.append(int(gpu_match.group(1)) if gpu_match else 0)

    return timestamps, cpu_usage, ram_usage, gpu_usage

def parse_log_summary(file_path):
    """Extrae métricas resumen (promedios y memoria vectorial)."""
    cpu_usage = []
    ram_usage = []
    gpu_usage = []
    vector_memories = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            cpu_match = re.search(r"CPU:\s+(\d+(?:\.\d+)?)%", line)
            ram_match = re.search(r"RAM:\s+(\d+(?:\.\d+)?)%", line)
            gpu_match = re.search(r"Uso:\s+(\d+)%", line)
            vector_match = re.search(r"'vector_memories':\s*(\d+)", line)

            if cpu_match:
                cpu_usage.append(float(cpu_match.group(1)))
            if ram_match:
                ram_usage.append(float(ram_match.group(1)))
            if gpu_match:
                gpu_usage.append(int(gpu_match.group(1)))
            if vector_match:
                vector_memories = max(vector_memories, int(vector_match.group(1)))

    return {
        "cpu_avg": statistics.mean(cpu_usage) if cpu_usage else 0,
        "ram_avg": statistics.mean(ram_usage) if ram_usage else 0,
        "gpu_avg": statistics.mean(gpu_usage) if gpu_usage else 0,
        "vector_memories": vector_memories
    }

def analyze_and_plot_logs():
    if not os.path.exists(LOG_DIR):
        print("⚠️ Carpeta de logs no encontrada.")
        return

    # Crear carpeta plots si no existe
    os.makedirs(PLOTS_DIR, exist_ok=True)

    results = []
    for file_name in os.listdir(LOG_DIR):
        if file_name.endswith(".log"):
            file_path = os.path.join(LOG_DIR, file_name)
            summary = parse_log_summary(file_path)
            timestamps, cpu, ram, gpu = parse_log_for_graphs(file_path)
            results.append((file_name, summary, timestamps, cpu, ram, gpu))

    if not results:
        print("⚠️ No se encontraron logs para analizar.")
        return

    print("\n📊 Resumen de métricas AGI")
    print("-" * 40)

    for file_name, metrics, timestamps, cpu, ram, gpu in results:
        print(f"📝 {file_name}")
        print(f"   🔹 CPU Promedio: {metrics['cpu_avg']:.2f}%")
        print(f"   🔹 RAM Promedio: {metrics['ram_avg']:.2f}%")
        print(f"   🔹 GPU Promedio: {metrics['gpu_avg']:.2f}%")
        print(f"   🔹 Vector Memories: {metrics['vector_memories']}\n")

        # Crear gráficos y guardarlos
        if timestamps:
            plt.figure(figsize=(10, 5))
            plt.plot(timestamps, cpu, label="CPU (%)")
            plt.plot(timestamps, ram, label="RAM (%)")
            plt.plot(timestamps, gpu, label="GPU (%)")
            plt.title(f"Métricas de {file_name}")
            plt.xlabel("Tiempo (intervalos)")
            plt.ylabel("Uso (%)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Guardar gráfico
            plot_file = os.path.join(PLOTS_DIR, f"{file_name.replace('.log', '')}.png")
            plt.savefig(plot_file)
            plt.close()

            print(f"   🖼️ Gráfico guardado en: {plot_file}")

if __name__ == "__main__":
    analyze_and_plot_logs()
