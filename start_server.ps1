# Quipu - punto de entrada para iniciar el LLM local Qwen mediante llama.cpp.
# Este archivo reemplaza el antiguo arranque de OpenChat/puerto 8010.
# No incluye claves de API, rutas absolutas del equipo ni procesos legacy.

[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8123,

    [ValidateRange(256, 262144)]
    [int]$NCtx = 8192,

    [ValidateRange(-1, 4096)]
    [int]$NGpuLayers = 35,

    [string]$ModelPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Launcher = Join-Path $ProjectRoot "scripts\run_llama_server.ps1"

if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
    throw "No se encontro el lanzador de llama.cpp: $Launcher"
}

# Las credenciales, si se usan, se leen por el lanzador desde LLAMA_API_KEY.
# No guardes claves en este archivo ni las pases como argumentos de consola.
$LaunchParams = @{
    Port       = $Port
    NCtx       = $NCtx
    NGpuLayers = $NGpuLayers
}

# Si no se indica una ruta, el lanzador usa su modelo Qwen GGUF predeterminado.
if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
    $LaunchParams.ModelPath = $ModelPath
}

Write-Host "Quipu: iniciando Qwen mediante scripts\run_llama_server.ps1..."
Write-Host "Servidor local: http://127.0.0.1:$Port/v1"
Write-Host "Deja esta ventana abierta. Para detener el servidor: Ctrl+C."

& $Launcher @LaunchParams
