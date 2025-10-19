# -*- coding: utf-8 -*-
# rag/router.py — Router limpio para el RAG mínimo (TF-IDF)

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from .rag_service import rag_query, RAG_DEFAULT_DIR

router = APIRouter(prefix="/v1/rag", tags=["rag"])

class RagRequest(BaseModel):
    query: str = Field("", description="Consulta; si vacío, devuelve top-docs.")
    k: int = Field(5, ge=1, le=50)
    corpus_dir: str = Field(RAG_DEFAULT_DIR)
    min_score: float = 0.0
    max_chars: int = 1200

class RagHit(BaseModel):
    score: float
    doc: str
    snippet: str

@router.post("/query", response_model=List[RagHit])
def do_rag(req: RagRequest):
    try:
        return rag_query(
            query=req.query,
            corpus_dir=req.corpus_dir,
            k=req.k,
            min_score=req.min_score,
            max_chars=req.max_chars,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {type(e).__name__}: {e}")
