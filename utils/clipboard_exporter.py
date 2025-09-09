import os
import json
import time
import pyperclip
from datetime import datetime
from pyperclip import PyperclipWindowsException

EXPORT_DIR = "clipboard_exports"

def ensure_export_dir():
    """Crea la carpeta de exportación si no existe."""
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)

def save_to_json(content: str):
    """Guarda el texto copiado en un archivo JSON único."""
    ensure_export_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(EXPORT_DIR, f"export_{timestamp}.json")
    
    data = {
        "timestamp": timestamp,
        "content": content
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"✅ Archivo generado: {filename}")

def main():
    print("🚀 Clipboard Exporter activo. Copia texto y se guardará automáticamente...")
    
    # Espera inicial para evitar errores al arrancar Windows
    time.sleep(10)
    
    last_text = ""
    
    while True:
        try:
            text = pyperclip.paste()
            if text and text != last_text:
                save_to_json(text)
                last_text = text
        except PyperclipWindowsException:
            print("⚠️ Portapapeles ocupado. Reintentando...")
            time.sleep(1)
            continue
        
        time.sleep(1)

if __name__ == "__main__":
    main()
