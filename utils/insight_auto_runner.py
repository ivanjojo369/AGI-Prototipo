import os
import sys

# Asegura que la ruta del proyecto esté en sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import corregido (relativo)
from .logger import Logger

def main():
    print("🚀 Iniciando sistema automático de insights...")

    logger = Logger("AGI Insight Logger")
    logger.log("Sistema de insights iniciado correctamente.")

    # Ejemplo de lógica de insights
    logger.log("Analizando datos del sistema AGI...")
    # Aquí puedes añadir las funciones de análisis necesarias

if __name__ == "__main__":
    main()
