from __future__ import annotations
import json, os, threading
from typing import Dict, Any, List, Optional

class SimpleVectorStore:
    """
    Almacena documentos en un JSON sencillo.
    Formato por doc: {"id": str, "text": str, "tags": List[str], "meta": Dict[str, Any]}
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        self._docs: Dict[str, Dict[str, Any]] = {}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._load()

    # ------- persistencia -------
    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "docs" in data and isinstance(data["docs"], dict):
                    self._docs = data["docs"]
            except Exception:
                # Archivo corrupto: renómbralo y empieza limpio
                base, ext = os.path.splitext(self.path)
                os.replace(self.path, f"{base}.corrupt{ext}")

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"docs": self._docs}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    # ------- API -------
    @property
    def count(self) -> int:
        return len(self._docs)

    def upsert(self, payload: Dict[str, Any]) -> str:
        """
        payload = {"id": str, "text": str, "tags": List[str], "meta": Dict[str, Any]}
        """
        doc_id = str(payload.get("id") or "")
        if not doc_id:
            raise ValueError("payload['id'] requerido")
        text = str(payload.get("text") or "")
        tags = payload.get("tags") or []
        meta = payload.get("meta") or {}
        with self._lock:
            self._docs[doc_id] = {"id": doc_id, "text": text, "tags": tags, "meta": meta}
            self._save()
        return doc_id

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._docs.get(str(doc_id))

    def items(self) -> List[Dict[str, Any]]:
        return list(self._docs.values())

    def delete(self, doc_id: str) -> None:
        with self._lock:
            self._docs.pop(str(doc_id), None)
            self._save()
