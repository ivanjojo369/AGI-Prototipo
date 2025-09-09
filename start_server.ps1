param(
  [string]$ModelPath   = 'C:\RUTA\AL\MODELO\openchat-3.5.Q5_K_S.gguf',
  [int]$N_CTX          = 8192,
  [int]$N_THREADS      = 8,
  [int]$N_BATCH        = 256,
  [int]$N_GPU_LAYERS   = 0,
  [string]$ApiKey      = 'CUKJMXoqkHYS2Zapxfl0tD85wyPnueOLE4sQARNr'
)

$env:MODEL_PATH   = $ModelPath
$env:N_CTX        = "$N_CTX"
$env:N_THREADS    = "$N_THREADS"
$env:N_BATCH      = "$N_BATCH"
$env:N_GPU_LAYERS = "$N_GPU_LAYERS"
$env:REQUIRE_API_KEY = '1'
$env:DEMO_KEY        = $ApiKey

uvicorn llama_server:app --host 127.0.0.1 --port 8010 --reload
