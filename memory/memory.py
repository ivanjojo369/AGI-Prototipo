# memory.py

import os
import json
import faiss
import numpy as np

# Simulación de base de datos (si FAISS no está activo)
SIMULATED_MEMORY = [
    {"contenido": "Hola, te llamas Alejandro."},
    {"contenido": "Te gusta aprender sobre AGI."},
    {"contenido": "Tu última interacción fue una pregunta sobre reflexión."}
]

# Ruta opcional para cargar embeddings FAISS (si estuviera habilitado)
FAISS_INDEX_PATH = "chroma_db/index.faiss"
VECTORS_PATH = "chroma_db/vectors.npy"
METADATA_PATH = "chroma_db/metadata.json"


def search_memory(query, top_k=3, use_faiss=False):
    """
    Busca recuerdos relacionados con el query.
    Por defecto, usa una memoria simulada.
    Si use_faiss=True y existen los archivos necesarios, usa FAISS.
    """
    if use_faiss and os.path.exists(FAISS_INDEX_PATH):
        return search_faiss_memory(query, top_k)
    else:
        return simulated_memory_search(query, top_k)


def simulated_memory_search(query, top_k=3):
    """
    Retorna recuerdos simulados (mock) relevantes.
    """
    return SIMULATED_MEMORY[:top_k]


def search_faiss_memory(query, top_k=3):
    """
    Busca en un índice FAISS si está disponible.
    """
    # Cargar índice
    index = faiss.read_index(FAISS_INDEX_PATH)

    # Cargar vectores y metadatos
    vectors = np.load(VECTORS_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Convertir query a vector (esto es solo un ejemplo)
    query_vector = np.random.rand(1, vectors.shape[1]).astype('float32')  # reemplazar por tu encoder real

    # Buscar
    D, I = index.search(query_vector, top_k)

    resultados = []
    for i in I[0]:
        if 0 <= i < len(metadata):
            resultados.append(metadata[i])
    return resultados


# Debug: imprimir funciones exportadas
if __name__ == "__main__":
    import inspect
    print("Funciones disponibles en memory.py:")
    for name, obj in inspect.getmembers(__import__('memory')):
        if inspect.isfunction(obj):
            print("-", name)
