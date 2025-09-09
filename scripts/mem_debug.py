# scripts/mem_debug.py
# -*- coding: utf-8 -*-

import os, sys, inspect
from typing import Any, Dict, Optional, Tuple

def _die(msg: str, code: int = 1):
    print(f"[ERR] {msg}")
    sys.exit(code)

try:
    # Ajusta si tu módulo vive en otra ruta
    from memory.semantic_memory import SemanticMemory
except Exception as e:
    _die(f"No pude importar SemanticMemory: {e}")

def _find_method(obj, *names):
    for n in names:
        if hasattr(obj, n):
            fn = getattr(obj, n)
            if callable(fn):
                return n, fn
    return None, None

def _snapshot_paths(sm) -> Dict[str, Any]:
    keys = ("base_dir","root","db_path","sqlite_path","faiss_path",
            "index_path","store_path","path")
    out = {}
    for k in keys:
        if hasattr(sm, k):
            out[k] = getattr(sm, k)
    return out

def _count_docs(sm) -> Optional[int]:
    for name in ("count", "n_docs", "size", "__len__"):
        if hasattr(sm, name):
            try:
                v = getattr(sm, name)
                return v() if callable(v) else int(v)
            except Exception:
                pass
    return None

def _upsert_any(sm, did: str, text: str, meta: Dict[str, Any]) -> Tuple[bool, Optional[Exception]]:
    """
    Intenta múltiples firmas para upsert/(write/add/ingest).
    Soporta métodos que exigen 'payload' (dict o lista de dicts) y variantes posicionales.
    """
    name, fn = _find_method(sm, "upsert", "write_episode", "write", "add", "ingest", "upsert_document")
    if not fn:
        print("[WARN] No encontré método de escritura (upsert/write/add/ingest).")
        return False, None

    print(f"[INFO] Método de escritura: {name}")

    # Detecta si la firma contiene 'payload'
    uses_payload_kw = False
    try:
        sig = inspect.signature(fn)
        uses_payload_kw = "payload" in sig.parameters
        print(f"[INFO] Firma de '{name}': {sig}")
    except Exception:
        sig = None

    # Candidatos de payload (dict) y lista de dicts
    base_dicts = [
        {"id": did, "text": text, "metadata": meta},
        {"doc_id": did, "text": text, "metadata": meta},
        {"id": did, "content": text, "metadata": meta},
        {"id": did, "body": text, "metadata": meta},
        {"id": did, "text": text, "meta": meta},
        {"id": did, "text": text},
    ]
    candidates = []
    for d in base_dicts:
        candidates.append(d)
        candidates.append([d])  # algunos upsert esperan lista

    last_exc: Optional[Exception] = None

    # 1) Si parece exigir 'payload', prueba con payload kw y también posicional
    for cand in candidates:
        try:
            if uses_payload_kw:
                fn(payload=cand)
            else:
                fn(cand)  # por si el primer parámetro es 'payload' posicional
            return True, None
        except Exception as e:
            last_exc = e

    # 2) Intentos posicionales clásicos
    positional_tries = [
        (did, text, None, meta),
        (text,),                 # p.ej. write(text)
        (text, did),             # p.ej. add(text, id)
        (did, text),             # p.ej. upsert(id, text)
    ]
    for args in positional_tries:
        try:
            fn(*args)
            return True, None
        except Exception as e:
            last_exc = e

    return False, last_exc

def _search_any(sm, q: str, k: int = 5):
    name, fn = _find_method(sm, "search", "retrieve", "query")
    if not fn:
        print("[ERR] SemanticMemory no expone search/retrieve/query.")
        return

    # Detecta parámetro top_k/k
    params = {}
    try:
        sig = inspect.signature(fn)
        if "top_k" in sig.parameters: params["top_k"] = k
        elif "k" in sig.parameters:  params["k"] = k
        print(f"[INFO] Firma de '{name}': {sig}")
    except Exception:
        pass

    # Ejecuta búsqueda
    try:
        items = fn(q, **params) if params else fn(q)
    except TypeError:
        try: items = fn(q, k=k)
        except Exception as e:
            print(f"[ERR] Falló la búsqueda: {e}")
            return
    except Exception as e:
        print(f"[ERR] Falló la búsqueda: {e}")
        return

    print("Q:", q)

    # --- Nuevo: mostrar crudos para inspección (primeros 3)
    for idx, it in enumerate((items or [])[:3]):
        if isinstance(it, dict):
            print(f"[RAW {idx}] keys:", list(it.keys()))
        else:
            print(f"[RAW {idx}] type:", type(it), "value:", it)

    def pick(d: dict, *keys, default=None):
        for k in keys:
            if k in d: return d[k]
        # Busca en anidados comunes
        for nest in ("document","doc","payload","item","data"):
            v = d.get(nest)
            if isinstance(v, dict):
                for k in keys:
                    if k in v: return v[k]
        return default

    top = []
    for it in items or []:
        if isinstance(it, dict):
            # id / doc_id
            did = pick(it, "doc_id", "id", "document_id", "uuid", default=None)

            # score
            sc  = pick(it, "score", "similarity", "cos", default=None)
            if sc is None:
                # Si viene como distance (menor mejor), conviértelo a score
                dist = pick(it, "distance", default=None)
                if dist is not None:
                    try:
                        sc = 1.0 - float(dist)          # ejemplo simple
                    except Exception:
                        sc = 0.0
            try:
                sc = round(float(sc or 0.0), 3)
            except Exception:
                sc = 0.0
        else:
            # objetos/tuplas: intenta atributos
            did = getattr(it, "doc_id", None) or getattr(it, "id", None)
            try:
                sc = round(float(getattr(it, "score", 0.0)), 3)
            except Exception:
                sc = 0.0

        top.append((did, sc))

    print("Top:", top)

def _ensure_demo(sm):
    demo = [
        ("doc://1", "Kant publicó la Crítica de la razón pura en 1781.", {"source":"mem_debug"}),
        ("doc://2", "La Ilustración enfatiza la razón.", {"source":"mem_debug"}),
        ("doc://3", "El siglo XVIII transformó Europa.", {"source":"mem_debug"}),
    ]
    ok, tot = 0, len(demo)
    for did, text, meta in demo:
        success, err = _upsert_any(sm, did, text, meta)
        if success:
            ok += 1
            print(f"[OK] upsert ← {did}")
        else:
            print(f"[WARN] No pude insertar {did}: {err}")
    if ok:
        print(f"[INFO] Ingestados (éxitos): {ok}/{tot}")

def main():
    if os.getcwd() not in (os.environ.get("PYTHONPATH") or ""):
        sys.path.insert(0, os.getcwd())

    try:
        sm = SemanticMemory()
    except Exception as e:
        _die(f"No pude instanciar SemanticMemory(): {e}")

    paths = _snapshot_paths(sm)
    print("[INFO] Rutas detectadas en SemanticMemory:")
    for k, v in paths.items():
        print(f"  - {k}: {v}")

    n = _count_docs(sm)
    print(f"[INFO] Docs en memoria: {n if n is not None else 'desconocido'}")

    if not n or n == 0:
        _ensure_demo(sm)

    print("\n[TEST] Búsquedas:")
    _search_any(sm, "¿Cuándo se publicó la Crítica de la razón pura?", 5)
    _search_any(sm, "Lenguaje de programación creado por Guido van Rossum", 5)

if __name__ == "__main__":
    main()
