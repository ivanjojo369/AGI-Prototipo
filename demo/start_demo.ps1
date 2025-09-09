# --- start_demo.ps1 ---
param(
  [string]$EnvName = "llama-env-py310",
  [string]$Host    = "127.0.0.1",
  [int]$ApiPort    = 8010,
  [int]$WebPort    = 8009
)

$ErrorActionPreference = "Stop"

# Ir a la raíz del repo (carpeta del script -> ..)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Resolver módulo FastAPI automáticamente
$Module = ""
if (Test-Path ".\llama_server.py") { $Module = "llama_server:app" }
elseif (Test-Path ".\server.py")   { $Module = "server:app" }
else { Write-Error "No se encontró llama_server.py ni server.py en $pwd"; exit 1 }

# Preparar comando con conda (fallback a ruta típica si 'conda' no está en PATH)
$prefix = ""
if (Get-Command conda -ErrorAction SilentlyContinue) {
  $prefix = "conda activate $EnvName; "
} else {
  $act = Join-Path $env:USERPROFILE "miniconda3\Scripts\activate"
  if (-not (Test-Path $act)) { Write-Warning "No se encontró conda. Continuo sin activar env." }
  else { $prefix = "& '$act'; conda activate $EnvName; " }
}

# Cerrar procesos que ya estén usando los puertos (suave)
$ports = @($ApiPort,$WebPort)
foreach ($p in $ports) {
  $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
  $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($pid in $pids) { try { Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue } catch {} }
}

# Lanzar API (nueva ventana)
$apiCmd = "${prefix}uvicorn $Module --host $Host --port $ApiPort --reload"
Start-Process powershell -ArgumentList '-NoExit','-Command', $apiCmd | Out-Null
Start-Sleep -Seconds 2

# Lanzar servidor estático (nueva ventana)
$webCmd = "${prefix}cd demo; python -m http.server $WebPort --bind $Host"
Start-Process powershell -ArgumentList '-NoExit','-Command', $webCmd | Out-Null
Start-Sleep -Seconds 1

# Abrir la demo en el navegador
$demoUrl = "http://$Host`:$WebPort/presentation.html"
Start-Process $demoUrl

Write-Host "Demo iniciada:"
Write-Host "  API:  http://$Host`:$ApiPort/"
Write-Host "  UI:   $demoUrl"
