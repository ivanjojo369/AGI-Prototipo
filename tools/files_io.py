from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
from datetime import datetime

import pandas as pd

try:
    import pdfplumber  # opcional
except Exception:  # pragma: no cover
    pdfplumber = None

# Límites prudentes para evitar cargas enormes
MAX_TEXT_CHARS = 50_000
MAX_ROWS_PREVIEW = 200


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _head_df(df: pd.DataFrame, n: int = 5) -> Dict[str, Any]:
    n = max(1, min(int(n or 5), MAX_ROWS_PREVIEW))
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(map(str, df.columns)),
        "head": df.head(n).to_dict(orient="records"),
        "preview_rows": n,
        "truncated": df.shape[0] > n
    }


def _txt_preview(p: Path, n_chars: int = MAX_TEXT_CHARS) -> str:
    txt = p.read_text(encoding="utf-8", errors="replace")
    return txt[: n_chars]


def _is_dir(p: Path) -> bool:
    try:
        return p.is_dir()
    except Exception:
        return False


def _dir_listing(p: Path) -> List[Dict[str, Any]]:
    items = []
    for child in sorted(p.iterdir()):
        try:
            stat = child.stat()
            items.append({
                "name": child.name,
                "path": str(child),
                "kind": "dir" if child.is_dir() else "file",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
            })
        except Exception:
            items.append({"name": child.name, "path": str(child), "kind": "unknown"})
    return items


def _pdf_preview(p: Path, n_pages: int = 10, n_chars: int = MAX_TEXT_CHARS) -> Tuple[str, int]:
    if not pdfplumber:
        return ("[files_io] pdfplumber no instalado", 0)
    text_parts: List[str] = []
    pages_read = 0
    with pdfplumber.open(p) as pdf:
        for page in pdf.pages[: max(1, n_pages)]:
            pages_read += 1
            text_parts.append(page.extract_text() or "")
            if sum(len(t) for t in text_parts) >= n_chars:
                break
    text = "\n".join(text_parts)[: n_chars]
    return (text, pages_read)


def run(path: str, action: str = "preview", n: int = 5, **kw) -> Dict[str, Any]:
    """
    Lee/inspecciona archivos y directorios de forma segura.
    Acciones:
      - "preview" (default): resumen rápido según tipo
      - "head": cabecera de CSV/XLSX (n filas)
      - "tail": últimas n líneas (TXT/MD) o n filas (CSV/XLSX)
      - "read": lee texto completo (con tope de caracteres)
      - "info": metadatos del archivo
      - "list": lista un directorio
    """
    result: Dict[str, Any] = {"ok": 0, "action": action or "preview", "path": path, "ts": _now_iso()}

    p = Path(path).expanduser().resolve()

    if not p.exists():
        result["error"] = "archivo o directorio no encontrado"
        return result

    # Soporte de listado de directorio
    if _is_dir(p):
        if action not in {"list", "preview"}:
            result["error"] = "acción no válida para directorios (usa 'list' o 'preview')"
            return result
        items = _dir_listing(p)
        result.update({"ok": 1, "kind": "dir", "items": items, "count": len(items)})
        return result

    suf = p.suffix.lower()
    action = (action or "preview").lower()
    n = int(n or 5)

    # INFO común
    try:
        stat = p.stat()
        meta = {
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "suffix": suf
        }
    except Exception:
        meta = {"suffix": suf}

    # CSV
    if suf in {".csv"}:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            return {**result, "error": f"no se pudo leer CSV: {e}"}
        payload = _head_df(df, n)
        if action == "tail":
            payload = {
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "columns": list(map(str, df.columns)),
                "tail": df.tail(max(1, min(n, MAX_ROWS_PREVIEW))).to_dict(orient="records"),
                "preview_rows": n
            }
        return {**result, "ok": 1, "kind": "csv", "meta": meta, **payload}

    # XLSX / XLS
    if suf in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(p)
        except Exception as e:
            return {**result, "error": f"no se pudo leer Excel: {e}"}
        payload = _head_df(df, n)
        if action == "tail":
            payload = {
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "columns": list(map(str, df.columns)),
                "tail": df.tail(max(1, min(n, MAX_ROWS_PREVIEW))).to_dict(orient="records"),
                "preview_rows": n
            }
        return {**result, "ok": 1, "kind": "xlsx", "meta": meta, **payload}

    # Markdown / Texto plano
    if suf in {".md", ".txt"}:
        txt = _txt_preview(p, MAX_TEXT_CHARS if action in {"preview", "read"} else 10_000)
        if action == "tail":
            lines = txt.splitlines()
            txt = "\n".join(lines[-max(1, n):])
        return {**result, "ok": 1, "kind": "text", "meta": meta, "text": txt}

    # JSON
    if suf in {".json"}:
        try:
            raw = _txt_preview(p, MAX_TEXT_CHARS)
            data = json.loads(raw)
            summary = {"type": type(data).__name__}
            if isinstance(data, list):
                summary["length"] = len(data)
            elif isinstance(data, dict):
                summary["keys"] = list(data.keys())[:50]
            payload = {"summary": summary}
            if action in {"preview", "head"}:
                payload["raw_preview"] = raw[:10_000]
            elif action == "read":
                payload["raw"] = raw
            return {**result, "ok": 1, "kind": "json", "meta": meta, **payload}
        except Exception as e:
            return {**result, "error": f"JSON inválido o demasiado grande: {e}"}

    # PDF
    if suf == ".pdf":
        text, pages_read = _pdf_preview(p, n_pages=max(1, min(n, 10)))
        return {**result, "ok": 1, "kind": "pdf", "meta": {**meta, "pages_read": pages_read}, "text": text}

    return {**result, "error": f"formato no soportado: {suf}"}
