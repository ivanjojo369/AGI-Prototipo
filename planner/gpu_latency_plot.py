import pandas as pd
import matplotlib.pyplot as plt
import os
import json

LOG_DIR = "logs/benchmarks/gpu/"
PLOTS_DIR = "logs/plots/"
os.makedirs(PLOTS_DIR, exist_ok=True)

def generate_graph():
    data = []
    print("📈 Generando gráfico de uso de VRAM...")

    for file in os.listdir(LOG_DIR):
        if file.endswith(".json"):
            with open(os.path.join(LOG_DIR, file)) as f:
                content = json.load(f)
                data.append({
                    "timestamp": content["timestamp"],
                    "used": content["used"]
                })

    if not data:
        print("⚠️ No se encontraron datos para graficar.")
        return

    df = pd.DataFrame(data)
    df.sort_values("timestamp", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["used"], marker="o", label="VRAM usada (MB)")
    plt.title("Evolución del uso de VRAM")
    plt.xlabel("Fecha")
    plt.ylabel("VRAM usada (MB)")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gpu_vram_usage.png"))
    plt.close()

    print(f"✅ Gráfico generado en: {os.path.join(PLOTS_DIR, 'gpu_vram_usage.png')}")

if __name__ == "__main__":
    generate_graph()
