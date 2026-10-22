# Auditoria estatica de knowledge/science antes de publicarlo.
# Solo lee archivos: no modifica el proyecto ni hace git add/commit/push.
[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
$problems = [System.Collections.Generic.List[string]]::new()
$warnings = [System.Collections.Generic.List[string]]::new()

function Fail([string]$Message) { $script:problems.Add($Message) }
function Warn([string]$Message) { $script:warnings.Add($Message) }

try {
    $root = (Resolve-Path -LiteralPath $ProjectRoot).ProviderPath
    $science = Join-Path $root 'knowledge\science'
    $manifestPath = Join-Path $science 'sources_manifest.json'

    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "No se encontro: $manifestPath"
    }

    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $manifest.corpus_id -or -not $manifest.documents) {
        Fail 'El manifiesto requiere corpus_id y documents.'
    }
    if ($manifest.do_not_mix_stores -ne $true) {
        Warn 'El manifiesto no indica do_not_mix_stores=true.'
    }

    $expected = @{}
    foreach ($doc in @($manifest.documents)) {
        $relative = [string]$doc.path
        $relative = $relative.Replace('\', '/')
        if ($relative -notmatch '^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+\.md$') {
            Fail "Ruta de documento no permitida en el manifiesto: $relative"
            continue
        }
        $key = $relative.ToLowerInvariant()
        if ($expected.ContainsKey($key)) {
            Fail "Documento duplicado en el manifiesto: $relative"
            continue
        }
        $expected[$key] = $true

        $path = Join-Path $science ($relative.Replace('/', [IO.Path]::DirectorySeparatorChar))
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            Fail "Falta documento declarado: $relative"
            continue
        }
        $file = Get-Item -LiteralPath $path
        if ($file.Length -eq 0) { Fail "Documento vacio: $relative" }
        if ($file.Length -gt 1MB) { Warn "Documento mayor a 1 MiB: $relative" }
        if ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            Fail "Enlace simbolico o junction no permitido: $relative"
        }
        foreach ($field in @('domain', 'title', 'source_org', 'source_type')) {
            if ([string]::IsNullOrWhiteSpace([string]$doc.$field)) {
                Fail "Falta $field en: $relative"
            }
        }
        if (@($doc.source_urls).Count -eq 0) {
            Fail "No hay URLs de fuentes para: $relative"
        }
        foreach ($url in @($doc.source_urls)) {
            if ([string]$url -notmatch '^https://[^\s]+$') {
                Fail "URL de fuente no HTTPS en: $relative"
            }
        }
    }

    $files = @(Get-ChildItem -LiteralPath $science -File -Recurse -Force)
    $secretChecks = @(
        '(?i)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        '(?i)\b(?:sk-[a-z0-9_-]{16,}|gh[pousr]_[a-z0-9]{20,}|github_pat_[a-z0-9_]{20,})\b',
        '(?i)\b(?:api[_-]?key|client[_-]?secret|password|access[_-]?token)\s*[:=]\s*\S{12,}'
    )
    foreach ($file in $files) {
        $rel = $file.FullName.Substring($science.Length).TrimStart('\', '/').Replace('\','/')
        if ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            Fail "Enlace simbolico o junction: $rel"
            continue
        }
        if ($rel -ne 'sources_manifest.json' -and -not $expected.ContainsKey($rel.ToLowerInvariant())) {
            Fail "Archivo no declarado en sources_manifest.json: $rel"
        }
        if ($file.Extension -notin @('.md', '.json')) {
            Fail "Extension inesperada: $rel"
            continue
        }
        $content = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
        foreach ($rule in $secretChecks) {
            if ($content -match $rule) {
                Fail "Posible secreto: $rel (revisar localmente; no imprimir contenido)"
                break
            }
        }
        if ($file.Extension -eq '.json') {
            try { $null = $content | ConvertFrom-Json }
            catch { Fail "JSON invalido: $rel" }
        }
    }

    Write-Host "Documentos declarados: $($expected.Count) | Archivos inspeccionados: $($files.Count)"
    foreach ($item in $warnings) { Write-Warning $item }
    foreach ($item in $problems) { Write-Host "ERROR: $item" -ForegroundColor Red }
    Write-Warning 'La auditoria NO comprueba licencias, copyright, exactitud cientifica ni permisos de redistribucion. Revisa manualmente cada documento y sus fuentes antes de publicar.'
    if ($problems.Count -gt 0) {
        Write-Host "NO APTO para preparar: $($problems.Count) problema(s)." -ForegroundColor Red
        exit 1
    }
    Write-Host 'Auditoria tecnica superada. Publicacion sujeta a revision manual de derechos y contenido.' -ForegroundColor Green
    exit 0
}
catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
