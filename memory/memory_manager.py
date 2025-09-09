import logging
import chromadb

class MemoryManager:
    def __init__(self):
        self.logger = logging.getLogger("MemoryManager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.logger.info("Inicializando MemoryManager con Chroma local...")

        self.client = chromadb.Client()

    def store_memory(self, key: str, value: str):
        """
        Almacena un recuerdo en la base de datos local.
        """
        self.logger.info(f"Guardando memoria [{key}] = {value}")
        collection = self.client.get_or_create_collection("memories")
        collection.add(documents=[value], ids=[key])

    def get_memory(self, key: str):
        """
        Recupera un recuerdo desde la base de datos local.
        """
        self.logger.info(f"Recuperando memoria [{key}]")
        collection = self.client.get_or_create_collection("memories")
        result = collection.get(ids=[key])
        return result.get("documents", [""])[0]
