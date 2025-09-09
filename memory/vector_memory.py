import chromadb
from sentence_transformers import SentenceTransformer

class VectorMemoryManager:
    def __init__(self, host="localhost", port=8765, collection_name="memory"):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def add_memory(self, id, text, metadata=None):
        embedding = self.embedder.encode(text).tolist()
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[id],
            metadatas=[metadata] if metadata else None
        )

    def query_memory(self, query_text, n_results=3):
        query_embedding = self.embedder.encode(query_text).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    def clear_memory(self):
        self.collection.delete(where={})

    def list_memory_ids(self):
        return self.collection.peek()["ids"]

    def persist(self):
        pass  # HttpClient no requiere persistencia local
