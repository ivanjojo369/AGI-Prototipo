from utils.logger import Logger
import os
import sys
from datetime import datetime

LOG_FILE = "sessions/insights.log"
EXPORT_DIR = "sessions/exports"

def asegurar_log():
    if not os.path.exists(LOG_FILE):
        Logger.info("⚠️ No hay insights registrados aún.")
        sys.exit()

def listar_insights():
    """Muestra todos los insights con número de línea."""
    asegurar_log()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lineas = f.readlines()
    for i, linea in enumerate(lineas, start=1):
        Logger.info(f"{i:04d}: {linea.strip()}")

def buscar_insight(palabra):
    """Busca insights por palabra clave."""
    asegurar_log()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        contenido = f.read()
    resultados = []
    for bloque in contenido.split("=" * 60):
        if palabra.lower() in bloque.lower():
            resultados.append(bloque.strip())
    if resultados:
        for i, res in enumerate(resultados, start=1):
            Logger.info(f"\n🔎 Resultado {i}:\n{res}\n")
    else:
        Logger.info(f"❌ No se encontraron insights que contengan '{palabra}'.")

def mostrar_por_fecha(fecha):
    """Muestra insights por fecha (formato: YYYY-MM-DD)."""
    asegurar_log()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        contenido = f.read()
    resultados = []
    for bloque in contenido.split("=" * 60):
        if fecha in bloque:
            resultados.append(bloque.strip())
    if resultados:
        for i, res in enumerate(resultados, start=1):
            Logger.info(f"\n📅 Insight {i}:\n{res}\n")
    else:
        Logger.info(f"❌ No se encontraron insights del {fecha}.")

def limpiar_log():
    """Limpia el archivo de insights."""
    asegurar_log()
    confirm = input("⚠️ ¿Seguro que deseas limpiar todos los insights? (s/n): ").lower()
    if confirm == "s":
        open(LOG_FILE, "w", encoding="utf-8").close()
        Logger.info("✅ Archivo de insights limpiado.")
    else:
        Logger.info("❌ Acción cancelada.")

def exportar_insight(line_number):
    """Exporta un insight específico a un archivo .txt."""
    asegurar_log()
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lineas = f.readlines()

    if line_number < 1 or line_number > len(lineas):
        Logger.info("❌ Número de línea inválido.")
        return

    # Obtener bloque completo del insight
    start = line_number - 1
    while start > 0 and not lineas[start].startswith("=" * 60):
        start -= 1

    end = line_number - 1
    while end < len(lineas) and not lineas[end].startswith("=" * 60):
        end += 1

    bloque = "".join(lineas[start:end])
    archivo = os.path.join(EXPORT_DIR, f"insight_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    with open(archivo, "w", encoding="utf-8") as f:
        f.write(bloque)

    Logger.info(f"✅ Insight exportado a: {archivo}")

def mostrar_ayuda():
    print("""
📖 Uso de manage.py:

python manage.py listar           → Lista todos los insights
python manage.py buscar <palabra> → Busca insights por palabra clave
python manage.py fecha <YYYY-MM-DD> → Muestra insights por fecha específica
python manage.py limpiar          → Limpia todos los insights
python manage.py exportar <número de línea> → Exporta insight a archivo .txt
""")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        mostrar_ayuda()
        sys.exit()

    comando = sys.argv[1].lower()

    if comando == "listar":
        listar_insights()
    elif comando == "buscar" and len(sys.argv) > 2:
        buscar_insight(sys.argv[2])
    elif comando == "fecha" and len(sys.argv) > 2:
        mostrar_por_fecha(sys.argv[2])
    elif comando == "limpiar":
        limpiar_log()
    elif comando == "exportar" and len(sys.argv) > 2:
        try:
            exportar_insight(int(sys.argv[2]))
        except ValueError:
            Logger.info("❌ Debes indicar un número de línea válido.")
    else:
        mostrar_ayuda()
