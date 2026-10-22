# Quipu / llama.cpp - lanzador local seguro (RC 1.1)
# Archivo de destino: AGI-Prototipo\scripts\run_llama_server.ps1
# Requiere llama-cpp-python con soporte de servidor en el entorno AegisForge_agent\.venv.
[CmdletBinding()]
param(
    [string]$ModelPath = ".\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    [ValidateRange(1, 65535)][int]$Port = 8123,
    [string]$ApiKey = "",
    [ValidateRange(256, 262144)][int]$NCtx = 8192,
    [ValidateRange(-1, 4096)][int]$NGpuLayers = 35,
    [ValidateSet("chatml")][string]$ChatFormat = "chatml"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).ProviderPath

# La ruta del GGUF se interpreta respecto a la raiz del proyecto, no del CWD.
if ([System.IO.Path]::IsPathRooted($ModelPath)) {
    $ModelCandidate = $ModelPath
} else {
    $ModelCandidate = Join-Path $ProjectRoot $ModelPath
}
if (-not (Test-Path -LiteralPath $ModelCandidate -PathType Leaf)) {
    throw "No se encontro el modelo GGUF: $ModelCandidate"
}
$ResolvedModel = (Resolve-Path -LiteralPath $ModelCandidate).ProviderPath

# Usar siempre el Python del entorno que Quipu emplea.
$PythonExe = Join-Path $ProjectRoot "AegisForge_agent\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "No se encontro Python del entorno virtual: $PythonExe"
}

# NO terminar procesos automaticamente: el puerto puede pertenecer a otro servicio.
$Listener = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($Listener.Count -gt 0) {
    $Owners = ($Listener | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
    throw "El puerto $Port ya esta ocupado (PID: $Owners). Deten el servidor adecuado manualmente antes de iniciar otro."
}

# Se aceptan credenciales por parametro o variable de entorno; nunca incrustadas.
# QuipuLoop utiliza LLAMA_API_KEY para autenticar sus peticiones.
if ([string]::IsNullOrWhiteSpace($ApiKey)) {
    $ApiKey = $env:LLAMA_API_KEY
}
if ([string]::IsNullOrWhiteSpace($ApiKey)) {
    Write-Warning "LLAMA_API_KEY no esta definida. El servicio local 127.0.0.1 se iniciara sin autenticacion."
}

$ServerArgs = @(
    "-m", "llama_cpp.server",
    "--model", $ResolvedModel,
    "--chat_format", $ChatFormat,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--n_ctx", "$NCtx",
    "--n_gpu_layers", "$NGpuLayers",
    "--embedding", "true"
)
if (-not [string]::IsNullOrWhiteSpace($ApiKey)) {
    $ServerArgs += @("--api_key", $ApiKey)
}

Write-Host "Iniciando llama.cpp para Quipu en http://127.0.0.1:$Port/v1"
Write-Host "Modelo: $ResolvedModel"
Write-Host "Chat format: $ChatFormat | Contexto: $NCtx | Capas GPU: $NGpuLayers"
Write-Host "La ventana debe permanecer abierta. Para detenerlo, presiona Ctrl+C."

& $PythonExe @ServerArgs
if ($LASTEXITCODE -ne 0) {
    throw "llama_cpp.server termino con codigo $LASTEXITCODE. Revisa los mensajes anteriores."
}
