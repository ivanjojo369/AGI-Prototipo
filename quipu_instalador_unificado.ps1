#requires -Version 5.1
<#
 Quipu - Administrador unificado de instalaciones historicas
 Consolida: install_core_4_0, install_factual_rc, install_compat_rc,
 install_functional_rc, install_functional_rc2, install_session_rc3,
 install_factual_1_2 e install_scientific_engine.

 IMPORTANTE:
 - No contiene los archivos Python de los paquetes historicos. Para instalar
   una etapa es necesario indicar su carpeta de contenido descomprimido.
 - No se instalan etapas anteriores encima de un proyecto actualizado.
 - No se crean ni copian archivos de test salvo con -IncludeTests.
 - Por defecto (Audit) NO modifica nada.

 Ejemplos:
 .\quipu_instalador_unificado.ps1
 .\quipu_instalador_unificado.ps1 -Action Tests
 .\quipu_instalador_unificado.ps1 -Action Dependencies
 .\quipu_instalador_unificado.ps1 -Action Install -Stage Factual12 `
     -PackageRoot 'C:\Ruta\paquete_descomprimido' -DryRun
 .\quipu_instalador_unificado.ps1 -Action Install -Stage Factual12 `
     -PackageRoot 'C:\Ruta\paquete_descomprimido'
#>
[CmdletBinding()]
param(
    [ValidateSet('Audit','Install','Dependencies','Tests')]
    [string]$Action = 'Audit',
    [ValidateSet('None','Core40','FactualRC11','CompatRC111','FunctionalRC1',
                 'FunctionalRC2','SessionRC3','Factual12')]
    [string]$Stage = 'None',
    [string]$ProjectRoot = 'C:\Users\PC\Documents\AGI-Prototipo',
    [string]$PackageRoot,
    [switch]$DryRun,
    [switch]$IncludeTests,
    [switch]$AllowUnverifiedBase
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function New-Entry {
    param(
        [string]$Path,
        [string]$ExpectedBefore = '',
        [string]$ExpectedAfter = '',
        [bool]$TestFile = $false
    )
    return [pscustomobject]@{
        Path = $Path.Replace('/', '\')
        Before = $ExpectedBefore.ToLowerInvariant()
        After = $ExpectedAfter.ToLowerInvariant()
        TestFile = $TestFile
    }
}

function Get-StageEntries {
    param([string]$Name)
    switch ($Name) {
        'Core40' {
            # La instalacion original de Core 4.0 no verificaba versiones base.
            # Ahora exige consentimiento explicito si pisa archivos existentes.
            $paths = @(
                'quipu_loop.py','quipu_http.py','root/settings.py',
                'rag/pipeline.py','rag/indexer.py','planner/planner.py',
                'planner/htn_simple.py','memory/unified_memory.py',
                'memory/embeddings_local.py','scientific_engine/engine.py',
                'scientific_engine/science_router.py',
                'scientific_engine/tool_registry.py',
                'scientific_engine/physics/symbolic_math.py',
                'scientific_engine/physics/units.py',
                'tests/test_scientific_engine.py',
                'tests/test_scientific_engine_integration.py',
                'tests/test_core_release_candidate.py',
                'FINAL_RELEASE_CHECK.ps1','CORE_RELEASE_NOTES.md',
                'requirements_core_4_0.txt'
            )
            foreach ($path in $paths) {
                New-Entry $path -TestFile ($path -like 'tests/*')
            }
        }
        'FactualRC11' {
            foreach ($path in @(
                'verifier/support_validator.py','verifier/factual_verifier.py',
                'rag/retriever.py','quipu_loop.py',
                'tests/test_factual_rc_1_1.py'
            )) {
                New-Entry $path -TestFile ($path -like 'tests/*')
            }
        }
        'CompatRC111' {
            New-Entry 'quipu_loop.py' '4ab3e6a04fd925547e221d7b66ad68d4ac594785b1ecd6e07846c56e50818472'
            New-Entry 'verifier/support_validator.py' '958c2274c31cfeb1308439b0e236ecd4a19d8ba69795fdf205afdbdc7b9a97bc'
            New-Entry 'tests/test_quipu_factual_reliability.py' '59afaf250625b21a50ac3cf17d1cd6786563ea8b4a30a38a08862d5b4578fb00' '' $true
        }
        'FunctionalRC1' {
            New-Entry 'quipu_loop.py' 'c80217b29e69ed0c2571d8bf682d6cff9b2abfcd35402b7c03a156ba80e455bd'
            New-Entry 'memory/unified_memory.py' 'f9b682fae7ab5680fef1d428b1a0348c7063695e0926c526cb4176675f384d4c'
            New-Entry 'scientific_engine/engine.py' 'b5baa1fe0e759a42d8f5c9bdbb09621a6a07b8e6e9eaec5c282fd61b7872c33d'
            New-Entry 'tests/test_quipu_functional_reliability.py' '' '' $true
        }
        'FunctionalRC2' {
            New-Entry 'quipu_loop.py' 'f74d00c7590dd021e78cc69a39158a4d79d4eedb0027e292cfaa8646bf579e77'
            New-Entry 'tests/test_cognitive_control_integration.py' 'c9e4de2272c2556f65934f576f40d15d94f0f4c7c4d10e81a26ded4635fe7e50' '' $true
            New-Entry 'tests/test_quipu_session_isolation.py' '' '' $true
        }
        'SessionRC3' {
            New-Entry 'quipu_loop.py' 'ebe741ceb06eb731a445eb57497725b132af557835a1536dd312de4e26825ec7'
            New-Entry 'tests/test_quipu_session_exact_memory.py' '' '' $true
        }
        'Factual12' {
            New-Entry 'quipu_loop.py' `
                'EBCC98F08754E305E2EDDC291B9138E1F27684C9F857D92D7DB1F90C27D3951A' `
                '6B324A498903E3893B61C6798EE905E1060CADC9DF759C9F71ACF6AF40F3C124'
            New-Entry 'verifier/factual_verifier.py' `
                'E49153920CCB1E9EA06DA596F4BE2EA50CC0608F25EB88190BE50BBF31E3E478' `
                '937B99909E1BCA7053CB37C469F9FE0DC8178A0F3B2EBA87686D567D8DAD2AD7'
            New-Entry 'tests/test_quipu_factual_reliability_1_2.py' '' '' $true
        }
    }
}

function Get-Hash {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "No existe la carpeta del proyecto: $ProjectRoot"
}
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$python = Join-Path $ProjectRoot 'AegisForge_agent\.venv\Scripts\python.exe'

switch ($Action) {
    'Audit' {
        Write-Host "Quipu | Auditoria sin cambios" -ForegroundColor Cyan
        Write-Host "Proyecto: $ProjectRoot"
        foreach ($relative in @(
            'quipu_loop.py', 'quipu_http.py',
            'verifier/factual_verifier.py','verifier/support_validator.py',
            'rag/retriever.py','scientific_engine/engine.py',
            'FINAL_RELEASE_CHECK.ps1','requirements_scientific_engine.txt'
        )) {
            $path = Join-Path $ProjectRoot $relative
            $exists = Test-Path -LiteralPath $path -PathType Leaf
            Write-Host ("{0,-48} {1}" -f $relative, $(if ($exists) {'OK'} else {'FALTA'}))
        }
        Write-Host ("Entorno virtual: " + $(if (Test-Path -LiteralPath $python -PathType Leaf) {'OK'} else {'FALTA'}))
        Write-Host 'No se han modificado archivos.'
        break
    }

    'Dependencies' {
        if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
            throw "No existe el Python del entorno virtual: $python"
        }
        $requirements = Join-Path $ProjectRoot 'requirements_scientific_engine.txt'
        if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) {
            throw "No existe $requirements"
        }
        Write-Host 'Dependencias de Scientific Engine:' -ForegroundColor Cyan
        Write-Host "  $requirements"
        if ($DryRun) {
            Write-Host 'Simulacion: no se instalaron paquetes.'
            break
        }
        & $python -m pip install -r $requirements
        if ($LASTEXITCODE -ne 0) { throw "pip fallo con codigo $LASTEXITCODE" }
        Write-Host 'Dependencias instaladas. No se ha actualizado pip automaticamente.'
        break
    }

    'Tests' {
        if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
            throw "No existe el Python del entorno virtual: $python"
        }
        $tests = @(@(
            'tests\test_quipu_factual_reliability.py',
            'tests\test_quipu_factual_reliability_1_2.py',
            'tests\test_quipu_functional_reliability.py',
            'tests\test_quipu_session_exact_memory.py',
            'tests\test_quipu_session_isolation.py',
            'tests\test_cognitive_control_integration.py'
        ) | Where-Object { Test-Path -LiteralPath (Join-Path $ProjectRoot $_) -PathType Leaf })
        if ($DryRun) {
            Write-Host 'Pruebas existentes que se ejecutarian:'
            $tests | ForEach-Object { Write-Host "  $_" }
            Write-Host '  FINAL_RELEASE_CHECK.ps1'
            break
        }
        Push-Location $ProjectRoot
        try {
            if (@($tests).Count -gt 0) {
                & $python -m pytest -q @tests
                if ($LASTEXITCODE -ne 0) { throw "Pruebas de pytest fallaron: $LASTEXITCODE" }
            }
            $gate = Join-Path $ProjectRoot 'FINAL_RELEASE_CHECK.ps1'
            if (-not (Test-Path -LiteralPath $gate -PathType Leaf)) {
                throw "No existe Release Gate: $gate"
            }
            # El gate historico puede ejecutar 'exit'; aislarlo en un proceso hijo.
            # Garantizar que sus comandos 'python' usen AegisForge_agent\.venv.
            $oldPath = $env:PATH
            try {
                $env:PATH = (Split-Path -Parent $python) + ';' + $oldPath
                & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $gate
                if ($LASTEXITCODE -ne 0) {
                    throw "Release Gate fallido con codigo $LASTEXITCODE"
                }
            } finally {
                $env:PATH = $oldPath
            }
        } finally {
            Pop-Location
        }
        Write-Host 'Pruebas finalizadas sin crear archivos de test.'
        break
    }

    'Install' {
        if ($Stage -eq 'None') {
            throw 'Selecciona -Stage (Core40, FactualRC11, CompatRC111, FunctionalRC1, FunctionalRC2, SessionRC3 o Factual12).'
        }
        if ([string]::IsNullOrWhiteSpace($PackageRoot)) {
            throw 'Indica -PackageRoot con la carpeta EXTRAIDA del paquete historico de esa etapa.'
        }
        if (-not (Test-Path -LiteralPath $PackageRoot -PathType Container)) {
            throw "No existe la carpeta del paquete: $PackageRoot"
        }
        $package = (Resolve-Path -LiteralPath $PackageRoot).Path
        if ($package.TrimEnd('\') -ieq $ProjectRoot.TrimEnd('\')) {
            throw 'PackageRoot debe ser una carpeta distinta del proyecto para no copiar archivos sobre si mismos.'
        }
        $entries = @(Get-StageEntries -Name $Stage | Where-Object { -not $_.TestFile -or $IncludeTests })
        $changes = @()
        foreach ($entry in $entries) {
            $target = Join-Path $ProjectRoot $entry.Path
            $source = Join-Path $package $entry.Path
            if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
                throw "Falta contenido del paquete: $source. No se modifico nada."
            }
            if ([IO.Path]::GetFullPath($target) -ieq [IO.Path]::GetFullPath($source)) {
                throw "Origen y destino son el mismo archivo: $target"
            }
            $sourceHash = Get-Hash $source
            if ($entry.After -and $sourceHash -ne $entry.After) {
                throw "El contenido del paquete no coincide con el SHA256 esperado: $($entry.Path)"
            }
            if (Test-Path -LiteralPath $target -PathType Leaf) {
                $currentHash = Get-Hash $target
                if ($currentHash -eq $sourceHash) {
                    Write-Host "Ya actualizado: $($entry.Path)"
                    continue
                }
                if ($entry.Before) {
                    if ($currentHash -ne $entry.Before) {
                        throw "Version local incompatible: $($entry.Path) SHA256=$currentHash. No se modifico nada."
                    }
                } elseif (-not $AllowUnverifiedBase) {
                    throw "El paquete $Stage no tiene hash base para $($entry.Path). Usa -AllowUnverifiedBase solamente tras verificar los cambios. No se modifico nada."
                }
                $existed = $true
            } else {
                if ($entry.Before) {
                    throw "Falta archivo de base requerido: $($entry.Path). No se modifico nada."
                }
                $existed = $false
            }
            $changes += [pscustomobject]@{
                Relative = $entry.Path
                Source = $source
                Target = $target
                SourceHash = $sourceHash
                Existed = $existed
            }
        }
        if ($changes.Count -eq 0) {
            Write-Host 'Todos los archivos seleccionados ya coinciden con el paquete. No se hizo nada.'
            break
        }
        Write-Host "Etapa: $Stage"
        Write-Host 'Cambios previstos:'
        $changes | ForEach-Object { Write-Host "  $($_.Relative)" }
        if ($DryRun) {
            Write-Host 'Simulacion terminada: no se modificaron archivos.'
            break
        }
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
        $backup = Join-Path $ProjectRoot ('.quipu_backups\unificado_' + $Stage + '_' + $stamp)
        $written = @()
        try {
            # Respaldo de TODOS los destinos existentes antes de sustituir ninguno.
            foreach ($item in $changes) {
                if ($item.Existed) {
                    $save = Join-Path $backup $item.Relative
                    New-Item -ItemType Directory -Path (Split-Path -Parent $save) -Force | Out-Null
                    Copy-Item -LiteralPath $item.Target -Destination $save -ErrorAction Stop
                    if ((Get-Hash $item.Target) -ne (Get-Hash $save)) {
                        throw "Fallo al comprobar el respaldo de $($item.Relative)"
                    }
                }
            }
            foreach ($item in $changes) {
                New-Item -ItemType Directory -Path (Split-Path -Parent $item.Target) -Force | Out-Null
                # Registramos antes de escribir para restaurar incluso ante copia parcial.
                $written += $item
                Copy-Item -LiteralPath $item.Source -Destination $item.Target -Force -ErrorAction Stop
                if ((Get-Hash $item.Target) -ne $item.SourceHash) {
                    throw "Hash de destino inesperado: $($item.Relative)"
                }
                Write-Host "Actualizado: $($item.Relative)"
            }
        } catch {
            $originalError = $_
            Write-Warning 'La instalacion fallo. Restaurando los archivos afectados...'
            foreach ($item in @($written)) {
                try {
                    if ($item.Existed) {
                        $save = Join-Path $backup $item.Relative
                        Copy-Item -LiteralPath $save -Destination $item.Target -Force -ErrorAction Stop
                    } elseif (Test-Path -LiteralPath $item.Target -PathType Leaf) {
                        Remove-Item -LiteralPath $item.Target -Force -ErrorAction Stop
                    }
                } catch {
                    Write-Warning "Restauracion manual necesaria: $($item.Target)"
                }
            }
            throw "Instalacion fallida: $originalError"
        }
        Write-Host "Etapa $Stage aplicada correctamente." -ForegroundColor Green
        Write-Host "Respaldo local: $backup"
        Write-Host 'Ejecuta -Action Tests antes de considerar aprobada esta etapa.'
        break
    }
}
