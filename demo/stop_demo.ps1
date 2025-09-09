# --- stop_demo.ps1 ---
param(
  [int[]]$Ports = @(8010, 8009)
)

$ErrorActionPreference = "SilentlyContinue"

foreach ($p in $Ports) {
  $conns = Get-NetTCPConnection -LocalPort $p -State Listen
  $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($pid in $pids) {
    try {
      $proc = Get-Process -Id $pid
      Write-Host "Cerrando PID $pid ($($proc.ProcessName)) en puerto $p..."
      Stop-Process -Id $pid -Force
    } catch {}
  }
}
Write-Host "Listo. Puertos liberados."
