import os
import json
import pandas as pd
import matplotlib.pyplot as plt

AUTO_BENCH_DIR = "logs/benchmarks/auto/"
PLOTS_DIR = "logs/plots/"
os.makedirs(PLOTS_DIR, exist_ok=True)

def generate_graph():
    data = []
    for file in os.listdir(AUTO_BENCH_DIR):
        if file.endswith(".json"):
            with open(os.path.join(AUTO_BENCH_DIR, file), "r", encoding="utf-8") as f:
                content = json.load(f)
                # Extraer latencia si está presente en stdout
                lines = content["stdout"].splitlines()
                latency = next((line for line in lines if "Tiempo de inferencia" in line), None)
                if latency:
                    value = float(latency.split(":")[-1].replace("ms", "").strip())
                    data.append({"timestamp": content["timestamp"], "latency": value})

    if not data:
        print("⚠️ No se encontraron datos de benchmark para graficar.")
        return

    df = pd.DataFrame(data)
    df.sort_values("timestamp", inplace=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["latency"], marker="o")
    plt.xticks(rotation=45)
    plt.title("Evolución de la latencia de inferencia")
    plt.xlabel("Fecha")
    plt.ylabel("Latencia (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "inference_latency.png"))
    plt.close()
    print("✅ Gráfico generado en logs/plots/inference_latency.png")

if __name__ == "__main__":
    generate_graph()
