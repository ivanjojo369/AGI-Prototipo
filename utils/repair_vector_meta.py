import json
import os

def repair_large_json(input_path, output_path):
    print("🔧 Reparando archivo JSON grande y corrupto...")
    fixed_data = []
    
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        buffer = ""
        for line in f:
            buffer += line.strip()
            try:
                obj = json.loads(buffer)
                fixed_data.append(obj)
                buffer = ""
            except json.JSONDecodeError:
                continue  # Todavía no es un JSON válido, seguimos leyendo
        
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(fixed_data, out, indent=2, ensure_ascii=False)

    print(f"✅ Archivo reparado y guardado en {output_path}")
    print(f"📦 Total recuerdos recuperados: {len(fixed_data)}")

if __name__ == "__main__":
    input_file = os.path.join("data", "memory", "vector_meta.json")
    output_file = os.path.join("data", "memory", "vector_meta_fixed.json")
    repair_large_json(input_file, output_file)
