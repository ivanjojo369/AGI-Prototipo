import os
import json
import re
from tqdm import tqdm

def repair_json_partial(file_path, output_path):
    print("🔧 Reparando archivo JSON parcialmente...")

    valid_objects = []
    buffer = ""
    errors = 0
    total_objects = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    
    # Intentar dividir por llaves (asumiendo que el archivo tiene objetos concatenados)
    chunks = re.split(r'(?<=\})\s*(?=\{)', raw)
    
    with tqdm(total=len(chunks), desc="🔍 Procesando objetos") as pbar:
        for chunk in chunks:
            total_objects += 1
            try:
                obj = json.loads(chunk)
                valid_objects.append(obj)
            except json.JSONDecodeError:
                errors += 1
            pbar.update(1)

    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(valid_objects, out, indent=2, ensure_ascii=False)

    print(f"✅ Reparación completada. Guardado en {output_path}")
    print(f"📦 Objetos válidos recuperados: {len(valid_objects)} / {total_objects}")
    print(f"⚠️ Objetos corruptos eliminados: {errors}")

if __name__ == "__main__":
    input_file = os.path.join("data", "memory", "vector_meta.json")
    output_file = os.path.join("data", "memory", "vector_meta_reparado_parcial.json")
    repair_json_partial(input_file, output_file)
