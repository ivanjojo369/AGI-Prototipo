# tests/test_memory_metrics.py
# Métricas extendidas sobre unified_memory: Recall@k, MRR@k y nDCG@k
# Mantiene compatibilidad con used_fact y retrieval_hit@k.

import csv
from pathlib import Path

from memory.metrics import (
    MemoryTrace, RetrievalItem,
    compute_memory_metrics, compute_and_log_memory_metrics
)

def test_compute_and_log_extended_metrics(tmp_path):
    # Construir trace con 3 documentos, el relevante va primero
    trace = MemoryTrace(goal="¿Cuándo se publicó la Crítica de la razón pura?")
    items = [
        RetrievalItem(doc_id="doc://1", score=0.9, text="Kant publicó la Crítica en 1781.", emb=None),
        RetrievalItem(doc_id="doc://2", score=0.7, text="La Ilustración enfatiza la razón.", emb=None),
        RetrievalItem(doc_id="doc://3", score=0.6, text="El siglo XVIII transformó Europa.", emb=None),
    ]
    trace.retrieved = items

    # Respuesta que utiliza el hecho de 1781 (debería marcar used_fact True)
    answer = "La 'Crítica de la razón pura' de Kant se publicó en 1781."

    # Métricas en memoria (sin escribir)
    metrics = compute_memory_metrics(
        goal=trace.goal,
        answer=answer,
        trace=trace,
        k=2,
        gold_doc_ids=["doc://1"],  # ground truth presente en top-2
        run_id="testrun123",
    )

    # Compatibilidad: aún puede existir retrieval_hit@k (bool)
    assert "retrieval_hit@k" in metrics
    assert metrics["retrieval_hit@k"] in (True, False, None)

    # Nuevas métricas: recall@k, mrr@k, ndcg@k
    assert "recall@k" in metrics, "Falta recall@k en compute_memory_metrics"
    assert "mrr@k" in metrics, "Falta mrr@k en compute_memory_metrics"
    assert "ndcg@k" in metrics, "Falta ndcg@k en compute_memory_metrics"

    # Con un solo relevante en top-1 dentro de k=2:
    #   recall@k = 1.0, mrr@k = 1/1 = 1.0, ndcg@k = 1.0
    assert abs(metrics["recall@k"] - 1.0) < 1e-6
    assert abs(metrics["mrr@k"] - 1.0) < 1e-6
    assert abs(metrics["ndcg@k"] - 1.0) < 1e-6

    # used_fact debe estar presente y ser True para esta respuesta
    assert "used_fact" in metrics
    assert metrics["used_fact"] in (True, False)
    # Es esperado True porque la respuesta contiene "1781"
    assert metrics["used_fact"] is True

    # Log a CSV (con columnas extendidas)
    out_dir = tmp_path / "logs" / "benchmarks"
    out = compute_and_log_memory_metrics(
        run_id="testrun123",
        goal=trace.goal,
        answer=answer,
        trace=trace,
        k=2,
        gold_doc_ids=["doc://1"],
        out_dir=str(out_dir),
    )

    # Debe haberse creado un CSV del día
    csvs = list(out_dir.glob("memory_metrics_*.csv"))
    assert csvs, "No se generó el CSV de métricas"

    # El CSV debe tener las columnas nuevas
    with open(csvs[0], "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        for needed in ["recall@k", "mrr@k", "ndcg@k", "used_fact", "retrieval_hit@k"]:
            assert needed in fieldnames, f"Falta columna {needed} en el CSV"
        rows = list(reader)
        assert rows, "El CSV está vacío"
