param(
  [string]$ApiUrl       = "http://127.0.0.1:8010/chat",
  [string]$ApiKey       = $env:DEMO_KEY,
  [int]   $MaxNewTokens = 110,
  [int]   $MaxSteps     = 5,
  [string]$TasksFile    = "$PSScriptRoot\benchmarks\golden_prompts.json"
)

# --- Mostrar config
Write-Host "Tokens $MaxNewTokens  -MaxSteps $MaxSteps"
Write-Host "[agent_battery] RepoRoot: $PSScriptRoot"
Write-Host "[agent_battery] API: $ApiUrl"

# --- API key al entorno (si viene por parámetro)
if ($ApiKey) {
  $env:DEMO_KEY = $ApiKey
  Write-Host "[agent_battery] Using X-API-Key from env: True"
} else {
  Write-Host "[agent_battery] Using X-API-Key from env: " -NoNewline
  Write-Host ([bool]$env:DEMO_KEY)
}

# --- Rutas
$repoRoot = $PSScriptRoot
$benchPy  = Join-Path $repoRoot "benchmarks\benchmark_agent.py"
if (-not (Test-Path $benchPy)) { throw "No se encontró $benchPy" }

if (-not (Test-Path $TasksFile)) {
  $TasksFile = Join-Path $repoRoot "benchmarks\golden_prompts.json"
  if (-not (Test-Path $TasksFile)) { throw "No se encontró $TasksFile" }
}

$outDir   = Join-Path $repoRoot "logs\benchmarks"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# --- Ejecutar benchmark_agent.py con flags nuevos
$pyArgs = @(
  $benchPy,
  "--api-url",        $ApiUrl,
  "--api-key",        $env:DEMO_KEY,
  "--tasks-file",     $TasksFile,
  "--max-new-tokens", $MaxNewTokens,
  "--max-steps",      $MaxSteps
)

Write-Host "[Benchmark] Tareas: $(Get-Content $TasksFile -Raw | Measure-Object -Line).Lines  |  API: $ApiUrl  |  Out: $outDir"

# Asegura Unicode limpio en consola
$env:PYTHONIOENCODING = "utf-8"

# Lanza Python
& python @pyArgs
if ($LASTEXITCODE -ne 0) { Write-Warning "benchmark_agent.py devolvió código $LASTEXITCODE" }

# --- Mostrar último results.csv (resumen)
$lastRun = Get-ChildItem (Join-Path $outDir "agent_run_*") -Directory |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($lastRun) {
  $resCsv = Join-Path $lastRun.FullName "results.csv"
  $infCsv = Join-Path $lastRun.FullName "infer_metrics.csv"

  if (Test-Path $resCsv) {
  Write-Host "`n[Benchmark] PASS summary:`n"
  Import-Csv $resCsv |
    Select-Object task_id, goal, pass, status, lat_ms |
    Format-Table -AutoSize
}

  Write-Host "`n[Benchmark] CSVs:"
  Write-Host " - $resCsv"
  Write-Host " - $infCsv"
}
