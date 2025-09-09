import os
import json
import re
from tqdm import tqdm

def repair_json_large(file_path, output_path):
    print("🔧 Reparando archivo JSON grande y corrupto...")

    file_size = os.path.getsize(file_path)
    repaired_lines = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="📄 Procesando") as pbar:
            for line in f:
                # Reparaciones básicas por línea
                line = re.sub(r'(\}\s*\{)', '},{', line)
                repaired_lines.append(line)
                pbar.update(len(line))

    repaired_data = "".join(repaired_lines).strip()

    # Asegurar formato de lista
    if not repaired_data.startswith("["):
        repaired_data = "[" + repaired_data
    if not repaired_data.endswith("]"):
        repaired_data = repaired_data + "]"

    try:
        data = json.loads(repaired_data)
        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2, ensure_ascii=False)
        print(f"✅ Archivo reparado y guardado en {output_path}")
        print(f"📦 Total recuerdos recuperados: {len(data)}")
    except json.JSONDecodeError as e:
        print(f"❌ No se pudo reparar automáticamente: {e}")

if __name__ == "__main__":
    input_file = os.path.join("data", "memory", "vector_meta.json")
    output_file = os.path.join("data", "memory", "vector_meta_reparado.json")
    repair_json_large(input_file, output_file)
