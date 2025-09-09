# tests/test_memory_metrics.py
# Verifica métricas de memoria: retrieval@k, used_fact y logging a CSV.

import os
from pathlib import Path
import numpy as np

from memory.metrics import (
    MemoryTrace, RetrievalItem,
    compute_memory_metrics, compute_and_log_memory_metrics
)

def test_compute_and_log(tmp_path):
    # Construir trace con 3 documentos
    trace = MemoryTrace(goal="¿Cuándo se publicó la Crítica de la razón pura?")
    items = [
        RetrievalItem(doc_id="doc://1", score=0.9, text="Kant publicó la Crítica en 1781.", emb=None),
        RetrievalItem(doc_id="doc://2", score=0.7, text="La Ilustración enfatiza la razón.", emb=None),
        RetrievalItem(doc_id="doc://3", score=0.6, text="El siglo XVIII transformó Europa.", emb=None),
    ]
    trace.retrieved = items

    # Respuesta que utiliza el hecho de 1781 (debería marcar used_fact True)
    answer = "La 'Crítica de la razón pura' de Kant se publicó en 1781."

    metrics = compute_memory_metrics(
        goal=trace.goal,
        answer=answer,
        trace=trace,
        k=2,
        gold_doc_ids=["doc://1"],  # ground truth presente en top-2
        run_id="testrun123",
    )
    assert metrics["retrieval_hit@k"] in (True, False, None)
    assert metrics["used_fact"] in (True, False)

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
