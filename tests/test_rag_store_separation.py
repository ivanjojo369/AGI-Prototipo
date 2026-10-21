from __future__ import annotations

import json
import tempfile
from pathlib import Path

from root import settings
from rag import indexer, retriever


def test_settings_separate_semantic_and_rag_paths():
    semantic = Path(settings.SEMANTIC_STORE_JSON).resolve()
    rag = Path(settings.RAG_STORE_JSON).resolve()

    assert semantic != rag
    assert semantic.name == "semantic_store.json"
    assert rag.name == "rag_store.json"

    print("=" * 100)
    print("TEST 1 — Semantic Memory y RAG usan archivos distintos")


def test_indexer_refuses_semantic_store_collision():
    old_out = indexer.OUT_PATH
    old_semantic = indexer.SEMANTIC_STORE_PATH

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        docs = root / "docs"
        docs.mkdir()
        (docs / "sample.txt").write_text(
            "Documento de prueba para el índice RAG.",
            encoding="utf-8",
        )

        semantic = root / "semantic_store.json"
        semantic.write_text('{"docs":{"fact:test":{"text":"NO TOCAR"}}}', encoding="utf-8")

        try:
            indexer.OUT_PATH = str(semantic)
            indexer.SEMANTIC_STORE_PATH = semantic.resolve()

            result = indexer.build_index(str(docs), ".txt", "fresh")

            assert result["ok"] is False
            assert result["error"] == "refusing-to-overwrite-semantic-memory-store"
            preserved = json.loads(semantic.read_text(encoding="utf-8"))
            assert preserved["docs"]["fact:test"]["text"] == "NO TOCAR"
        finally:
            indexer.OUT_PATH = old_out
            indexer.SEMANTIC_STORE_PATH = old_semantic

    print("=" * 100)
    print("TEST 2 — rag.indexer se niega a sobrescribir semantic_store.json")


def test_indexer_writes_rag_list_format():
    old_out = indexer.OUT_PATH
    old_semantic = indexer.SEMANTIC_STORE_PATH

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        docs = root / "docs"
        docs.mkdir()
        (docs / "physics.txt").write_text(
            "El horizonte de sucesos de Schwarzschild está en r = 2GM/c^2.",
            encoding="utf-8",
        )

        rag_store = root / "rag_store.json"
        semantic_store = root / "semantic_store.json"

        try:
            indexer.OUT_PATH = str(rag_store)
            indexer.SEMANTIC_STORE_PATH = semantic_store.resolve()

            result = indexer.build_index(str(docs), ".txt", "fresh")

            assert result["ok"] is True
            assert result["store_kind"] == "rag"
            assert Path(result["written"]).resolve() == rag_store.resolve()

            data = json.loads(rag_store.read_text(encoding="utf-8"))
            assert isinstance(data, list)
            assert data
            assert data[0]["text"]
            assert "meta" in data[0]
            assert "vec" in data[0]
            assert not semantic_store.exists()
        finally:
            indexer.OUT_PATH = old_out
            indexer.SEMANTIC_STORE_PATH = old_semantic

    print("=" * 100)
    print("TEST 3 — rag.indexer escribe el formato moderno en rag_store.json")


def test_retriever_reads_only_rag_store():
    old_store = retriever.RAG_STORE_JSON

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rag_store = root / "rag_store.json"

        text = "El horizonte de sucesos no es la singularidad."
        rag_store.write_text(
            json.dumps(
                [
                    {
                        "id": "doc-1",
                        "text": text,
                        "meta": {"path": "physics.txt", "chunk": 0},
                        "vec": retriever.embed_text(text),
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        try:
            retriever.RAG_STORE_JSON = str(rag_store)
            rows = retriever._load_index()

            assert len(rows) == 1
            assert rows[0]["id"] == "doc-1"
            assert rows[0]["text"] == text
            assert rows[0]["meta"]["path"] == "physics.txt"
        finally:
            retriever.RAG_STORE_JSON = old_store

    print("=" * 100)
    print("TEST 4 — rag.retriever carga rag_store.json")


def main():
    test_settings_separate_semantic_and_rag_paths()
    test_indexer_refuses_semantic_store_collision()
    test_indexer_writes_rag_list_format()
    test_retriever_reads_only_rag_store()

    print("=" * 100)
    print("RAG STORE SEPARATION 1.0: 4/4 pruebas superadas.")


if __name__ == "__main__":
    main()
