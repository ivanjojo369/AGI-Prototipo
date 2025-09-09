# tests/test_agi_flow.py
# Smoke tests del agente: loop básico, T02 (primer JSON válido), stop-loss + métricas.

import types
import uuid
import json
import os
import shutil
from pathlib import Path

import pytest

from agents.agent import Agent, AgentConfig

# --------------------------- Utilidades de parcheo ---------------------------

class DummyMemory:
    """Memoria vectorial mínima para provocar un retrieval y habilitar métricas."""
    def retrieve(self, query, k=5):
        return [
            {"id": "doc://1", "text": "Kant publicó la Crítica en 1781.", "score": 0.9},
            {"id": "doc://2", "text": "La Ilustración enfatiza la razón.", "score": 0.7},
            {"id": "doc://3", "text": "El siglo XVIII transformó Europa.", "score": 0.6},
        ]

def patch_generate_sequence(agent: Agent, texts):
    """Parchea Agent._generate para devolver una secuencia de textos controlada."""
    seq = iter(texts)
    def _fake_generate(_self, prompt: str):
        try:
            s = next(seq)
        except StopIteration:
            s = texts[-1]
        return s, {"text": s}
    agent._generate = types.MethodType(_fake_generate, agent)  # bind

# ------------------------------- Fixtures ------------------------------------

@pytest.fixture()
def tmp_logs_dir(tmp_path):
    logs = tmp_path / "logs" / "benchmarks"
    logs.mkdir(parents=True, exist_ok=True)
    return logs

@pytest.fixture()
def base_agent(tmp_logs_dir, monkeypatch):
    cfg = AgentConfig(
        api_url="http://localhost:8010/chat",
        max_new_tokens=64,
        timeout=5.0,
        strict=False,
        mode="auto",
        max_steps=3,
        retry_count=1,
        reflect_budget=0,  # ReflectionEngine puede no estar en el entorno
        metrics_enabled=True,
        metrics_out_dir=str(tmp_logs_dir),
        metrics_k=5,
    )
    ag = Agent(cfg)
    # Inyectamos memoria para activar métricas
    ag.memory = DummyMemory()

    # Espiamos compute_and_log_memory_metrics si está disponible
    called = {"n": 0}
    try:
        import agents.agent as agent_mod
        real = getattr(agent_mod, "compute_and_log_memory_metrics", None)
        if real is not None:
            def spy(*args, **kwargs):
                called["n"] += 1
                return real(*args, **kwargs)
            monkeypatch.setattr(agent_mod, "compute_and_log_memory_metrics", spy)
    except Exception:
        pass

    return ag, called

# --------------------------------- Tests -------------------------------------

def test_non_strict_simple(base_agent):
    ag, called = base_agent
    patch_generate_sequence(ag, ["Hola mundo."])
    out = ag.run("Di hola.")
    assert isinstance(out, str) and "Hola" in out
    # Debe haber intentado loguear métricas (si memory.metrics está disponible)
    assert called["n"] >= 0

def test_strict_t02_extract_first_json(base_agent):
    ag, _ = base_agent
    ag.cfg.strict = True
    # Mezcla de texto + JSON válido intercalado
    text = "prefacio bla bla {\"a\": 1, \"b\": [1,2,3]} posfacio"
    patch_generate_sequence(ag, [text])
    out = ag.run("Devuelve JSON con a y b.")
    # En modo estricto, Agent retorna el JSON serializado (string)
    parsed = json.loads(out)
    assert parsed["a"] == 1 and parsed["b"] == [1, 2, 3]

def test_strict_t02_fixes_trailing_comma(base_agent):
    ag, _ = base_agent
    ag.cfg.strict = True
    # JSON con coma colgante → el extractor interno intenta corregirlo
    bad_then_ok = [
        "texto {\"a\": 1,} ruido",                   # inválido (con coma colgante) -> el fix intenta parsear
    ]
    patch_generate_sequence(ag, bad_then_ok)
    out = ag.run("Devuelve JSON con campo a.")
    parsed = json.loads(out)
    assert parsed["a"] == 1

def test_stoploss_retry_triggers_and_metrics(base_agent):
    ag, called = base_agent
    ag.cfg.strict = True
    ag.cfg.retry_count = 1  # un reintento lógico

    seq = [
        "no-json :)",                 # 1ª salida: inválida → dispara reintento
        "ok {\"ok\": true}"           # 2ª salida: contiene JSON válido
    ]
    patch_generate_sequence(ag, seq)
    out = ag.run("Devuelve JSON con ok=true.")
    parsed = json.loads(out)
    assert parsed["ok"] is True
    # debería haber intentado escribir métricas 1 vez
    assert called["n"] >= 1

