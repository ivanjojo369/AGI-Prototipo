# -*- coding: utf-8 -*-
"""
Indexa una carpeta de texto/código a data/semantic_store.json para RAG.
Uso:
  python -m scripts.index_folder --path ./data --ext .txt,.md,.py --chunk 1200 --overlap 200 --mode fresh
"""
from __future__ import annotations
import os, json, argparse
from typing import List, Dict, Any
from root.settings import SEMANTIC_STORE_JSON, EMBED_DIM, RAG_MAX_CHUNK_LEN
from utils.normalize import clean_markdown, squash_spaces, dedupe_chunks
from memory.episodic_memory import embed_text  # mismo embedder que Memoria

TEXT_EXTS_DEFAULT = [".txt", ".md", ".py", ".json", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".csv"]

def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def chunk_text(text: str, chunk: int, overlap: int) -> List[str]:
    t = text if len(text) <= RAG_MAX_CHUNK_LEN else text[:RAG_MAX_CHUNK_LEN]
    if len(t) <= chunk:
        return [t]
    out = []
    i = 0
    step = max(1, chunk - overlap)
    while i < len(t):
        out.append(t[i:i+chunk])
        i += step
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Carpeta a indexar")
    ap.add_argument("--ext", type=str, default=",".join(TEXT_EXTS_DEFAULT), help="Extensiones separadas por coma")
    ap.add_argument("--chunk", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--dedupe", type=float, default=0.92, help="Umbral Jaccard para dedupe (0..1)")
    ap.add_argument("--mode", type=str, default="fresh", choices=["fresh","append"], help="fresh: reescribe; append: añade")
    args = ap.parse_args()

    exts = [e.strip().lower() for e in args.ext.split(",") if e.strip()]
    index: List[Dict[str, Any]] = []

    if os.path.exists(SEMANTIC_STORE_JSON) and args.mode == "append":
        try:
            with open(SEMANTIC_STORE_JSON, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            index = []

    next_id = len(index)
    for root, _, files in os.walk(args.path):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if exts and ext not in exts:
                continue
            fpath = os.path.join(root, fn)
            raw = read_file(fpath)
            if not raw.strip():
                continue
            norm = clean_markdown(raw)
            norm = squash_spaces(norm)
            chunks = chunk_text(norm, args.chunk, args.overlap)

            # dedupe
            keep_ix = dedupe_chunks(chunks, jaccard_threshold=args.dedupe)
            for ci in keep_ix:
                ch = chunks[ci]
                vec = embed_text(ch)
                index.append({
                    "id": f"doc_{next_id}",
                    "text": ch,
                    "meta": {"path": fpath, "chunk_id": ci, "source": "local"},
                    "vec": vec,
                })
                next_id += 1

    with open(SEMANTIC_STORE_JSON, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Indexados {next_id} chunks en {SEMANTIC_STORE_JSON} (modo={args.mode})")

if __name__ == "__main__":
    main()
