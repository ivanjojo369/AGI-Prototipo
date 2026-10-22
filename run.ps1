Param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("bench", "start", "stop", "status")]
    [string]$Command,

    [int]$Port = 8080
)

# Wrapper PowerShell para controlar GREEN (bench + servidor HTTP).
# Uso:
#   .\run.ps1 -Command bench
#   .\run.ps1 -Command start -Port 8080
#   .\run.ps1 -Command stop
#   .\run.ps1 -Command status

$ErrorActionPreference = "Stop"

$python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$runCtrl = "run_ctrl.py"

switch ($Command) {
    "bench" {
        & $python $runCtrl "bench"
    }
    "start" {
        & $python $runCtrl "start" "--port" $Port
    }
    "stop" {
        & $python $runCtrl "stop"
    }
    "status" {
        & $python $runCtrl "status"
    }
}
