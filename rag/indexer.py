# rag/indexer.py
# Indexador RAG -> escribe chunks + embeddings en data/rag_store.json (lista de items).
from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Dict, Iterable

# ---- settings con fallback -------------------------------------------
try:
    from root import settings as S
except Exception:
    S = None

PROJ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = getattr(S, "DATA_DIR", PROJ_DIR / "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = str(
    getattr(
        S,
        "RAG_STORE_JSON",
        os.getenv("RAG_STORE_JSON", str(DATA_DIR / "rag_store.json")),
    )
)

# Defensa contra regresiones: el indexador RAG jamás debe escribir encima
# del almacén de memoria semántica.
SEMANTIC_STORE_PATH = Path(
    str(getattr(S, "SEMANTIC_STORE_JSON", DATA_DIR / "semantic_store.json"))
).resolve()
CHUNK_SIZE     = int(getattr(S, "CHUNK_SIZE",     int(os.getenv("RAG_CHUNK_SIZE", "900"))))
CHUNK_OVERLAP  = int(getattr(S, "CHUNK_OVERLAP",  int(os.getenv("RAG_CHUNK_OVERLAP", "120"))))

# usamos el mismo embedder que la memoria para consistencia
from memory.episodic_memory import embed_text

TEXT_EXTS = {".md", ".txt", ".py", ".json", ".rst"}

def _iter_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    exts = {e.lower().strip() for e in exts}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _chunks(text: str, size: int, overlap: int):
    text = text.replace("\r\n", "\n")
    i = 0
    while i < len(text):
        yield text[i:i+size]
        i += max(1, size - overlap)

def build_index(folder: str, exts: str = ".md,.txt,.py,.json", mode: str = "fresh") -> Dict:
    """
    Construye el índice documental moderno en ``data/rag_store.json``.

    mode:
      - fresh   -> reemplaza el índice con el contenido de ``folder``;
      - append  -> añade/actualiza por id sin duplicar chunks existentes;
      - rebuild -> equivalente a fresh, explícito para scripts antiguos.
    """
    folder_p = Path(folder).resolve()
    if not folder_p.exists():
        return {"ok": False, "error": f"folder-not-found: {folder_p}"}

    out_p = Path(OUT_PATH).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if out_p == SEMANTIC_STORE_PATH:
        return {
            "ok": False,
            "error": "refusing-to-overwrite-semantic-memory-store",
            "semantic_store": str(SEMANTIC_STORE_PATH),
            "rag_store": str(out_p),
        }

    existing: List[Dict] = []
    if mode == "append" and out_p.exists():
        try:
            raw = json.loads(out_p.read_text("utf-8"))
            if isinstance(raw, list):
                existing = [row for row in raw if isinstance(row, dict)]
        except Exception:
            existing = []

    # append usa un mapa por id: reindexar un documento actualiza sus chunks
    # en vez de duplicarlos indefinidamente.
    by_id: Dict[str, Dict] = {
        str(row.get("id")): row
        for row in existing
        if row.get("id")
    }
    added = 0
    updated = 0

    normalized_exts = []
    for item in exts.split(","):
        item = item.strip().lower()
        if not item:
            continue
        normalized_exts.append(item if item.startswith(".") else "." + item)

    for file in _iter_files(folder_p, normalized_exts):
        raw = _read_text(file)
        if not raw.strip():
            continue
        try:
            relative = file.relative_to(folder_p).as_posix()
        except Exception:
            relative = file.as_posix()

        for j, ch in enumerate(_chunks(raw, CHUNK_SIZE, CHUNK_OVERLAP)):
            rid = f"{relative}::chunk{j}"
            item = {
                "id": rid,
                "text": ch,
                "meta": {
                    "path": file.as_posix(),
                    "relative_path": relative,
                    "chunk": j,
                },
                "vec": embed_text(ch),
            }
            if rid in by_id:
                updated += 1
            else:
                added += 1
            by_id[rid] = item

    items = list(by_id.values())
    out_p.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "ok": True,
        "written": str(out_p),
        "store_kind": "rag",
        "count": len(items),
        "added": added,
        "updated": updated,
        "mode": mode,
    }

# --- CLI ---
def main(argv=None):
    import argparse
    p = argparse.ArgumentParser("rag.indexer", description="Construye el índice documental RAG separado de Semantic Memory.")
    p.add_argument("--path", required=True, help="Carpeta a indexar (p. ej. ./docs)")
    p.add_argument("--ext", default=".md,.txt,.py,.json", help="Extensiones separadas por coma")
    p.add_argument("--mode", default="fresh", choices=["fresh","append","rebuild"])
    args = p.parse_args(argv)
    res = build_index(args.path, args.ext, args.mode)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
