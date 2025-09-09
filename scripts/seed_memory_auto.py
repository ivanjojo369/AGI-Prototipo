# scripts/seed_memory_auto.py
import os, sys, inspect
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from memory.semantic_memory import SemanticMemory

DOCS = [
    ("doc://1", "Kant publicó la Crítica de la razón pura en 1781."),
    ("doc://2", "La Ilustración enfatiza la razón."),
    ("doc://3", "El siglo XVIII transformó Europa."),
]

WRITE_NAMES = [
    "write_episode", "write", "add", "upsert",
    "index", "index_text", "index_texts", "add_text", "add_texts",
    "add_document", "add_documents", "ingest", "ingest_texts",
    "insert", "put", "append", "store", "save", "save_document",
]

SEARCH_NAMES = ["search", "retrieve", "query", "knn", "similarity_search"]

def try_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except TypeError as e:
        return None, e
    except Exception as e:
        return None, e

def write_any(sm, id, text, **meta):
    for name in WRITE_NAMES:
        if not hasattr(sm, name):
            continue
        fn = getattr(sm, name)
        if not callable(fn):
            continue

        for call in (
            lambda: fn(id=id, text=text, meta=meta),              # (id, text, meta)
            lambda: fn(text=text, meta=dict(id=id, **meta)),      # (text, meta) con id en meta
            lambda: fn(text),                                     # (text)
            lambda: fn({"id": id, "text": text, "meta": meta}),   # (dict)
            lambda: fn([{"id": id, "text": text, "meta": meta}]), # ([dict])
        ):
            _, err = try_call(call)
            if err is None:
                print(f"[OK] {name} ← {id}")
                return True
    return False

def find_search(sm):
    for name in SEARCH_NAMES:
        if hasattr(sm, name) and callable(getattr(sm, name)):
            return name
    return None

def main():
    sm = SemanticMemory()
    wrote = 0
    for did, txt in DOCS:
        if write_any(sm, did, txt, source="demo"):
            wrote += 1
    print(f"\nIngestados (éxitos): {wrote}/{len(DOCS)}")

    sname = find_search(sm)
    if not sname:
        print("No encontré método de búsqueda (search/retrieve/query).")
        return
    sfn = getattr(sm, sname)
    print(f"Usando búsqueda: {sname}")

    for q in ["¿Cuándo se publicó la Crítica de la razón pura?", "Kant 1781"]:
        for kws in (dict(k=5, include_embeddings=True),
                    dict(top_k=5, include_embeddings=True),
                    dict(k=5), dict(top_k=5),
                    {}):
            res, err = try_call(sfn, q, **kws)
            if err is None:
                items = res or []
                def _gid(it):
                    if isinstance(it, dict):
                        return it.get("doc_id") or it.get("id") or it.get("doc")
                    return getattr(it, "doc_id", getattr(it, "id", None))
                rows = [(_gid(it),
                         (it.get("score") if isinstance(it, dict) else getattr(it, "score", None)))
                        for it in items]
                print(f"\n{q} -> {rows}")
                break

if __name__ == "__main__":
    main()
