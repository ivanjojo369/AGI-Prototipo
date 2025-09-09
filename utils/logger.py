import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, name: str = "AGI Logger", log_dir: str = "logs"):
        """
        Inicializa el sistema de logging.
        :param name: Nombre del logger (para identificar la fuente de logs).
        :param log_dir: Carpeta donde se guardarán los archivos de log.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Crear carpeta de logs si no existe
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Nombre del archivo con timestamp
        log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

        # Handler para archivo
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formato unificado
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Evitar duplicados
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def log(self, message: str, level: str = "info"):
        """
        Registra un mensaje en el log.
        :param message: Texto del log
        :param level: Nivel del log (info, warning, error, debug)
        """
        level = level.lower()
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)
        else:
            self.logger.info(message)
