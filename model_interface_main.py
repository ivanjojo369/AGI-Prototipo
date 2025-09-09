from abc import ABC, abstractmethod

class ModelInterface(ABC):
    @abstractmethod
    def generar_respuesta(self, prompt: str) -> str:
        pass
