# -*- coding: utf-8 -*-
import os, json, math, re
from pathlib import Path
from typing import Dict, List, Tuple

ALLOWED_EXTS = {".txt", ".md", ".mdx", ".log"}

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9]+", text.lower())

def _tf(words: List[str]) -> Dict[str, float]:
    tf = {}
    for w in words:
        tf[w] = tf.get(w, 0.0) + 1.0
    n = float(len(words)) or 1.0
    for k in tf:
        tf[k] /= n
    return tf

def ingest_folder(folder: str, out_index: str) -> Dict:
    """Crea un índice TF simple (sin embeddings) para RAG mínimo."""
    folder = str(folder)
    docs = []
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() in ALLOWED_EXTS and p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
            tokens = _tokenize(text)
            docs.append({"path": str(p), "text": text, "tf": _tf(tokens)})

    # idf
    df = {}
    for d in docs:
        seen = set(d["tf"].keys())
        for w in seen:
            df[w] = df.get(w, 0) + 1
    N = len(docs) or 1
    idf = {w: math.log((N + 1) / (c + 1)) + 1.0 for w, c in df.items()}

    idx = {"docs": docs, "idf": idf, "N": N}
    Path(out_index).parent.mkdir(parents=True, exist_ok=True)
    with open(out_index, "w", encoding="utf-8") as fh:
        json.dump(idx, fh)
    return {"ok": True, "docs": len(docs), "index": out_index}
