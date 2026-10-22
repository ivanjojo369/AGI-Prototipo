$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host '=== QUIPU CORE 4.0.0 — FINAL RELEASE CHECK ==='

$compile = @(
  'quipu_loop.py',
  'quipu_http.py',
  'root/settings.py',
  'rag/pipeline.py',
  'rag/indexer.py',
  'rag/retriever.py',
  'planner/planner.py',
  'planner/htn_simple.py',
  'memory/unified_memory.py',
  'memory/embeddings_local.py'
)
python -m py_compile @compile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$tests = @(
  'tests/test_scientific_engine.py',
  'tests/test_scientific_engine_integration.py',
  'tests/test_core_release_candidate.py',
  'tests/test_rag_store_separation.py',
  'tests/test_skill_orchestrator.py',
  'tests/test_cognitive_control_integration.py',
  'tests/test_relevance_gate.py',
  'tests/test_metacognitive_gate.py',
  'tests/test_output_guard.py',
  'tests/test_self_correction.py',
  'tests/test_working_world_state.py',
  'tests/test_conversation_state.py',
  'tests/test_long_range_retriever.py',
  'tests/test_factual_verifier_regression.py',
  'tests/test_factual_support_validation.py',
  'tests/test_factual_judge_reliability.py',
  'tests/test_factual_verifier_integration.py'
)
python -m pytest -q @tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ''
Write-Host 'QUIPU CORE 4.0.0: RELEASE GATE SUPERADO.'
