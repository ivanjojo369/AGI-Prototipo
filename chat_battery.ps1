param(
  [string]$ApiUrl = 'http://127.0.0.1:8010',
  [string]$ApiKey = $env:DEMO_KEY
)

$ErrorActionPreference = 'Stop'

# -------------- utils --------------
function EnsureDir { param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) { New-Item -ItemType Directory -Path $Path | Out-Null }
}
function NowStr { (Get-Date).ToString('yyyyMMdd_HHmmss') }
function WriteText { param([string]$Path,[string]$Text) Set-Content -Path $Path -Encoding UTF8 -Value $Text }
function ToJsonPS { param($Obj,[int]$Depth=12)
  if ($PSVersionTable.PSVersion.Major -ge 7) { $Obj | ConvertTo-Json -Depth $Depth -Compress }
  else { $Obj | ConvertTo-Json -Depth $Depth }
}

# -------------- logs --------------
$rootLog = Join-Path -Path 'logs' -ChildPath ("chat_battery_{0}" -f (NowStr))
EnsureDir $rootLog

# -------------- tolerant JSON parser --------------
function ExtractBraced { param([string]$txt,[char]$open,[char]$close)
  $depth = 0; $start = -1
  for ($i=0; $i -lt $txt.Length; $i++) {
    $c = $txt[$i]
    if ($c -eq $open) { if ($depth -eq 0) { $start = $i } ; $depth++ }
    elseif ($c -eq $close) {
      if ($depth -gt 0) {
        $depth--
        if ($depth -eq 0 -and $start -ge 0) { return $txt.Substring($start, $i-$start+1) }
      }
    }
  }
  return $null
}

function ParseTopJson { param([string]$s)
  if ([string]::IsNullOrWhiteSpace($s)) { throw 'Empty' }
  # strip fences and trim
  $s = $s -replace '^\s*```(?:json)?\s*','' -replace '\s*```\s*$',''
  $s = $s.Trim()
  # normalize quotes
  $s = $s -replace '[\u201C\u201D]','"' -replace "[\u2018\u2019]","'"

  function TryJson([string]$t) { try { ,($t | ConvertFrom-Json -ErrorAction Stop) } catch { $null } }

  $o = TryJson $s
  if ($o) { return $o }

  $objTxt = ExtractBraced $s '{' '}'
  if ($objTxt) {
    $objTxt = $objTxt -replace ',(\s*[}\]])','$1'
    $o = TryJson $objTxt
    if ($o) { return $o }
  }

  $arrTxt = ExtractBraced $s '[' ']'
  if ($arrTxt) {
    $arrTxt = $arrTxt -replace ',(\s*[}\]])','$1'
    $o = TryJson $arrTxt
    if ($o) { return $o }
  }
  throw 'No top-level JSON or primitive invalid'
}

function ParsePlannerJson { param([string]$outer)
  $o = ParseTopJson $outer
  $subs = @()
  if ($o -is [hashtable] -and $o.ContainsKey('subgoals')) { $subs = $o['subgoals'] }
  elseif ($o -is [System.Array]) { $subs = $o }
  else { throw "planner: missing 'subgoals'" }

  $clean = @()
  foreach ($g in $subs) {
    $d = $null; $w = $null
    if ($g -is [hashtable]) {
      if ($g.ContainsKey('desc'))      { $d = [string]$g['desc'] }
      if ($g.ContainsKey('done_when')) { $w = [string]$g['done_when'] }
    } elseif ($g -is [string]) {
      $d = $g; $w = ''
    }
    if ($null -ne $d) {
      if ($null -eq $w) { $w = '' }
      $clean += [pscustomobject]@{ desc = $d; done_when = $w }
    }
  }
  if ($clean.Count -eq 0) { throw 'planner: empty subgoals' }
  return $clean
}

# -------------- HTTP helpers --------------
$Headers = @{ 'Content-Type' = 'application/json' }
if ($ApiKey) { $Headers['X-API-Key'] = $ApiKey }

function PostJson { param([string]$Path,[hashtable]$Body)
  $uri = ($ApiUrl.TrimEnd('/')) + $Path
  $json = ToJsonPS $Body
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $status = 0; $resp = $null
  try {
    $resp = Invoke-RestMethod -Method Post -Uri $uri -Headers $Headers -Body $json -ContentType 'application/json'
    $status = 200
  } catch {
    $status = try { [int]$_.Exception.Response.StatusCode.value__ } catch { 0 }
    $resp = $null
  }
  $sw.Stop()
  [pscustomobject]@{ status=$status; ms=$sw.ElapsedMilliseconds; body=$resp; sent=$json }
}

function GetHealth {
  $uri = ($ApiUrl.TrimEnd('/')) + '/'
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $status = 0; $resp = $null
  try { $resp = Invoke-RestMethod -Method Get -Uri $uri -Headers $Headers; $status = 200 }
  catch { $status = try { [int]$_.Exception.Response.StatusCode.value__ } catch { 0 } }
  $sw.Stop()
  [pscustomobject]@{ status=$status; ms=$sw.ElapsedMilliseconds; body=$resp }
}

# -------------- request builders --------------
# builders
$DEFAULT_STOP = @('<|end_of_turn|>','</s>')

function NewChatBody {
  param([string]$Sys,[string]$User,[hashtable]$Params)

  $msgs = @()
  if ($Sys)  { $msgs += @{ role = 'system'; content = $Sys } }
  if ($User) { $msgs += @{ role = 'user'  ; content = $User } }

  # PowerShell 5.1 friendly: si $Params es null, usar @{}
  if ($null -eq $Params) { $Params = @{} }

  return @{
    messages = $msgs
    params   = $Params
  }
}

# -------------- runner --------------
$rows = New-Object System.Collections.Generic.List[object]
function AddRow { param($id,$name,$ms,$status,[string]$note)
  $rows.Add([pscustomobject]@{ id=$id; name=$name; ms=$ms; status=$status; note=$note })
  $label = if ($status -eq 200) { 'ok' } else { 'http_error' }
  Write-Host ("[{0}] {1} -> {2} ms (status {3}) {4}" -f $id,$name,$ms,$status,$label)
}

# [1] latency
$h = GetHealth
AddRow 1 'latency' $h.ms $h.status ''

# [2] planner
$plSys = 'Planner HTN. Return ONLY valid JSON root {"subgoals":[{"desc":"...","done_when":"..."}]}. No backticks, no extra text. If unsure, return {"subgoals":[]}. Use ASCII quotes.'
$plUser = 'Implement FAISS in semantic memory, evaluate retrieval@k and document setup.'
$plParams = @{ max_new_tokens=160; temperature=0.2; top_p=0.8; stop=$DEFAULT_STOP; stream=$false }
$plBody   = NewChatBody -Sys $plSys -User $plUser -Params $plParams
$p = PostJson -Path '/chat' -Body $plBody
$plNote = ''
if ($p.status -eq 200) {
  try {
    $txt = if ($p.body -and $p.body.text) { [string]$p.body.text } else { ToJsonPS $p.body }
    $goals = ParsePlannerJson $txt
    WriteText (Join-Path $rootLog 't2_planner_subgoals.json') (ToJsonPS @{ subgoals=$goals })
  } catch { $plNote = 'parse_error: ' + $_.Exception.Message }
}
AddRow 2 'planner_htn' $p.ms $p.status $plNote

# [3] checklist
$ckSys  = 'Technical auditor. Return concise numbered checklist with concrete commands. <= 6 lines.'
$ckUser = 'Checklist to fix 500 in /chat and lower total_ms on uvicorn (Windows, llama.cpp optional).'
$ck = PostJson -Path '/chat' -Body (NewChatBody -Sys $ckSys -User $ckUser -Params @{max_new_tokens=120;temperature=0.35;top_p=0.9;stop=$DEFAULT_STOP;stream=$false})
AddRow 3 'checklist' $ck.ms $ck.status ''

# [4] golden
$gd = PostJson -Path '/chat' -Body (NewChatBody -Sys '' -User 'ping' -Params @{max_new_tokens=40;temperature=0.5;stop=$DEFAULT_STOP;stream=$false})
AddRow 4 'golden' $gd.ms $gd.status ''

# [5] json_cmds
$jsSys  = 'Return ONLY valid JSON root {"cmds":[{"name":"...","args":{"k":"v"}}]}. No extra text.'
$jsUser = 'Give three example commands to prepare FAISS in a Python project.'
$js = PostJson -Path '/chat' -Body (NewChatBody -Sys $jsSys -User $jsUser -Params @{max_new_tokens=120;temperature=0.3;stop=$DEFAULT_STOP;stream=$false})
AddRow 5 'json_cmds' $js.ms $js.status ''

# -------------- persist --------------
WriteText (Join-Path $rootLog 't1_health.json')          (ToJsonPS $h)
WriteText (Join-Path $rootLog 't2_planner_req.json')     (ToJsonPS $plBody)
WriteText (Join-Path $rootLog 't2_planner_raw.json')     (ToJsonPS $p)
WriteText (Join-Path $rootLog 't3_checklist.json')       (ToJsonPS $ck)
WriteText (Join-Path $rootLog 't4_golden.json')          (ToJsonPS $gd)
WriteText (Join-Path $rootLog 't5_json_cmds.json')       (ToJsonPS $js)

$summary = Join-Path $rootLog 'summary.csv'
$rows | Export-Csv -Path $summary -NoTypeInformation -Encoding UTF8

Write-Host ''
Write-Host ("Resumen: {0}" -f $summary) -ForegroundColor Cyan
Write-Host 'For TTFT/tokens/s: see server logs/infer_metrics.csv.'
