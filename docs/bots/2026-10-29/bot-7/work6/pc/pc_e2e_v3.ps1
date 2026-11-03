# bot-2, 2026-11-01. Сквозной прогон v3 против v2 на ПК (GTX 1650, 4 ГБ, Docker/WSL2).
# Ничего не посылает на лидерборд, в Kaggle, в git и в S3.
#
# По умолчанию пути на C: (F: не трогать). Диск обязателен только если задан -RequireDrive.
# PowerShell 5.1 и 7: без Tee-Object -Append, без ??, без LinkType, Join-Path из двух частей.
#
# Действительный прогон (код выхода 0, строка «ИТОГ: PASS») — все пункты:
#   * в логе v3 есть «sp_dfine n_det=N total» и N > 0;
#   * ни в одном логе нет SP_DFINE_UNAVAILABLE;
#   * SHA256 result.geojson v3 и v2 различаются;
#   * официальная метрика посчитана, и у markup sp_or число попаданий (tp) > 0,
#     если D-FINE вообще запускался.
# Иначе код выхода 1 и «ИТОГ: FAIL» со списком причин.
#
# Примеры:
#   powershell -ExecutionPolicy Bypass -File pc_e2e_v3.ps1 -Mode docker `
#     -ZipV3 C:\Sloi\pc_jobs\v3e2e\kit\submission_v3_e2e.zip `
#     -Regions C:\Sloi\pc_jobs\v3e2e\regions -Out C:\Sloi\pc_jobs\v3e2e\out_fix
#   pwsh -NoProfile -File pc_e2e_v3.ps1 -Mode mock -Py python3 -Metric materials/compute_metrics.py
param(
  [ValidateSet('docker', 'venv', 'mock')][string]$Mode = 'docker',
  [string]$Image = 'pantum/n7q:20261024',
  [string]$Py = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe',
  [string]$TfDir = 'C:\Sloi\pc_jobs\v3e2e\tf521',
  [switch]$SetupTf,
  [string]$Regions = 'C:\Sloi\pc_jobs\v3e2e\regions',
  [string]$Bot7 = 'C:\Sloi\pc_jobs\v3e2e\kit',
  [string]$ZipV2 = 'C:\Sloi\pc_jobs\v3e2e\kit\submission_bot7cand_v2_eb268d6.zip',
  [string]$ZipV3 = 'C:\Sloi\pc_jobs\v3e2e\kit\submission_v3_e2e.zip',
  [string]$Out = 'C:\Sloi\pc_jobs\v3e2e\out_fix',
  [string]$RequireDrive = 'C',
  [string]$ContainerMemory = '24g',
  [string]$ShmSize = '8g',
  [string]$Metric = '',
  [string]$Gt = '',
  [int]$MockNDet = 3,
  [switch]$MockUnavailable,
  [switch]$MockIdentical,
  [int]$MockSporMatches = 1
)
$ErrorActionPreference = 'Stop'
$script:Fails = New-Object System.Collections.Generic.List[string]
$script:TimeFile = $null

function Add-Fail([string]$Message) {
  [void]$script:Fails.Add($Message)
  Write-Host "FAIL: $Message"
}
function Write-Log([string]$Message) {
  Write-Host $Message
  if ($script:TimeFile) { Add-Content -Path $script:TimeFile -Value $Message -Encoding UTF8 }
}
function Test-WindowsOS { return ($env:OS -eq 'Windows_NT') }
function Join-Child([string]$Parent, [string]$Name) { return (Join-Path -Path $Parent -ChildPath $Name) }

if ($Mode -eq 'mock' -and -not $PSBoundParameters.ContainsKey('Out')) {
  $Out = Join-Child ([System.IO.Path]::GetTempPath()) 'pc_e2e_v3_mock'
  $RequireDrive = ''
}
if ($Mode -eq 'mock') { $RequireDrive = '' }

New-Item -ItemType Directory -Force -Path $Out | Out-Null
$script:TimeFile = Join-Child $Out 'time.txt'
Set-Content -Path $script:TimeFile -Value '' -Encoding UTF8
Write-Log ("start {0} mode={1}" -f (Get-Date -Format o), $Mode)

if ($RequireDrive) {
  $prefix = ($RequireDrive.TrimEnd(':') + ':\').ToUpper()
  foreach ($p in @($Py, $TfDir, $Regions, $Bot7, $ZipV2, $ZipV3, $Out)) {
    if (-not $p.ToUpper().StartsWith($prefix)) { throw "путь не на диске ${prefix}: $p" }
  }
}

function Test-NoReparse([string]$Root) {
  if (-not (Test-WindowsOS)) { return }
  if (-not (Test-Path $Root)) { return }
  Get-ChildItem -Path $Root -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $attr = $_.Attributes
    if (($attr -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
      throw "регион $($_.Name) — ссылка (junction/symlink), скопируйте каталог, не ссылку"
    }
  }
}

function Get-VramMiB {
  if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $null }
  $raw = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
  if ($LASTEXITCODE -ne 0 -or -not $raw) { return $null }
  $best = $null
  foreach ($row in @($raw)) {
    $n = 0
    $cell = ($row.ToString().Split(',')[0]).Trim()
    if ([int]::TryParse($cell, [ref]$n)) {
      if ($null -eq $best -or $n -gt $best) { $best = $n }
    }
  }
  return $best
}

function Get-RamMiB([int]$ProcId, [string]$DockerName) {
  $best = 0.0
  $any = $false
  if ($DockerName) {
    $mem = & docker stats --no-stream --format '{{.MemUsage}}' $DockerName 2>$null
    if ($LASTEXITCODE -eq 0 -and $mem) {
      $part = ($mem.ToString().Split('/')[0]).Trim()
      if ($part -match '^([\d\.]+)\s*([KMGT]iB|[KMGT]B)$') {
        $n = [double]$Matches[1]
        switch ($Matches[2]) {
          'KiB' { $n = $n / 1024 }
          'MiB' { }
          'GiB' { $n = $n * 1024 }
          'TiB' { $n = $n * 1024 * 1024 }
          'KB' { $n = $n / 1000 }
          'MB' { }
          'GB' { $n = $n * 1000 }
          'TB' { $n = $n * 1000 * 1000 }
        }
        $best = $n
        $any = $true
      }
    }
  }
  if ($ProcId -gt 0) {
    try {
      $ws = (Get-Process -Id $ProcId -ErrorAction Stop).WorkingSet64
      $mib = [double]$ws / 1MB
      if ($mib -gt $best) { $best = $mib }
      $any = $true
      if (Test-WindowsOS) {
        $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, WorkingSetSize
        $ids = @($ProcId)
        $i = 0
        while ($i -lt $ids.Count) {
          $cur = $ids[$i]
          foreach ($row in @($all)) {
            if ([int]$row.ParentProcessId -eq $cur) { $ids += [int]$row.ProcessId }
          }
          $i++
        }
        $sum = 0.0
        foreach ($row in @($all)) {
          if ($ids -contains [int]$row.ProcessId) { $sum += [double]$row.WorkingSetSize }
        }
        $tree = $sum / 1MB
        if ($tree -gt $best) { $best = $tree }
      }
    } catch { }
  }
  if (-not $any) { return $null }
  return [int][Math]::Round($best)
}

function Write-CudaWrapper([string]$Path) {
  @'
import runpy, sys
code = 0
try:
    sys.argv = ["inference/main.py", sys.argv[1], sys.argv[2]]
    runpy.run_path("inference/main.py", run_name="__main__")
except SystemExit as exc:
    code = int(exc.code or 0)
finally:
    try:
        import torch
        if torch.cuda.is_available():
            print("TORCH_CUDA_MAX_ALLOC_MIB=%.1f" % (torch.cuda.max_memory_allocated() / 1048576.0))
        else:
            print("TORCH_CUDA_MAX_ALLOC_MIB=na")
    except Exception as exc:
        print("TORCH_CUDA_MAX_ALLOC_ERR=%s" % type(exc).__name__)
raise SystemExit(code)
'@ | Set-Content -Path $Path -Encoding UTF8
}

function Invoke-OneRun([string]$Tag, [string]$Work, [string]$ResultDir) {
  New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null
  $result = Join-Child $ResultDir 'result.geojson'
  if (Test-Path $result) { Remove-Item -Force $result }
  $log = Join-Child $Out ("main_{0}.log" -f $Tag)
  $err = Join-Child $Out ("main_{0}.err" -f $Tag)
  $wrap = Join-Child $Out 'cuda_peak.py'
  if (-not (Test-Path $wrap)) { Write-CudaWrapper $wrap }
  $name = "expds-e2e-$Tag"
  $peakV = $null
  $peakR = $null
  $t0 = Get-Date
  if ($Mode -eq 'docker') {
    & docker rm -f $name 2>$null | Out-Null
    $args = @(
      'run', '--name', $name, '--rm', '--gpus', 'all', '--network', 'none',
      '--read-only', '--user', '12345:12345', '--memory', $ContainerMemory,
      '--pids-limit', '8192', '--shm-size', $ShmSize,
      '-v', "${Work}:/app:ro", '-v', "${Regions}:/data/input:ro",
      '-v', "${ResultDir}:/data/output", '-v', "${Out}:/probe:ro",
      '-w', '/app', $Image, 'python', '/probe/cuda_peak.py',
      '/data/input', '/data/output/result.geojson'
    )
    $proc = Start-Process -FilePath 'docker' -ArgumentList $args -PassThru -NoNewWindow `
      -RedirectStandardOutput $log -RedirectStandardError $err
  } else {
    if ($Tag -eq 'v3' -and $TfDir -and (Test-Path $TfDir)) { $env:PYTHONPATH = $TfDir }
    $proc = Start-Process -FilePath $Py -ArgumentList @($wrap, $Regions, $result) -WorkingDirectory $Work `
      -PassThru -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError $err
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
  }
  while (-not $proc.HasExited) {
    $v = Get-VramMiB
    if ($null -ne $v -and ($null -eq $peakV -or $v -gt $peakV)) { $peakV = $v }
    $dockerName = ''
    if ($Mode -eq 'docker') { $dockerName = $name }
    $r = Get-RamMiB -ProcId $proc.Id -DockerName $dockerName
    if ($null -ne $r -and ($null -eq $peakR -or $r -gt $peakR)) { $peakR = $r }
    Start-Sleep -Milliseconds 1000
  }
  $proc.WaitForExit()
  $v = Get-VramMiB
  if ($null -ne $v -and ($null -eq $peakV -or $v -gt $peakV)) { $peakV = $v }
  $wall = [int]((Get-Date) - $t0).TotalSeconds
  $torchLine = ''
  if (Test-Path $log) {
    $hit = Select-String -Path $log -Pattern 'TORCH_CUDA_MAX_ALLOC_MIB=(\S+)' | Select-Object -Last 1
    if ($hit) { $torchLine = $hit.Matches[0].Groups[1].Value }
  }
  $vramText = 'n/a'
  if ($null -ne $peakV) { $vramText = "$peakV MiB" }
  $ramText = 'n/a'
  if ($null -ne $peakR) { $ramText = "$peakR MiB" }
  $torchText = 'n/a'
  if ($torchLine) { $torchText = "$torchLine MiB" }
  Write-Log ("{0}: wall {1} s, peak RAM {2}, peak VRAM {3}, torch.cuda.max_memory_allocated {4}, exit {5}" -f `
      $Tag, $wall, $ramText, $vramText, $torchText, $proc.ExitCode)
  if ($proc.ExitCode -ne 0) { Add-Fail "$Tag inference exit $($proc.ExitCode)" }
  if ($Mode -ne 'mock' -and $null -eq $peakV) { Add-Fail "$Tag нет замера VRAM (nvidia-smi)" }
  if ($Mode -ne 'mock' -and $null -eq $peakR) { Add-Fail "$Tag нет замера RAM" }
}

function New-Square([double]$X, [double]$Y) {
  $x2 = $X + 10
  $y2 = $Y + 10
  return "[[[${X},${Y}],[${x2},${Y}],[${x2},${y2}],[${X},${y2}],[${X},${Y}]]]"
}

function Write-Fc([string]$Path, [string]$Coords, [string]$Markup) {
  $json = '{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Polygon","coordinates":' + $Coords + '},"properties":{"region_name":"101","class_name":"kurgany_povrezhdennye","markup_type":"' + $Markup + '","confidence":0.9}}]}'
  Set-Content -Path $Path -Value $json -Encoding UTF8
}

if ($Mode -eq 'mock') {
  $nd = $MockNDet
  $lines = @(
    "INFO expds.pipeline: sp_dfine n_det=$nd region=101 complete=True, прошло 1 с",
    "INFO expds.pipeline: sp_dfine n_det=$nd total, участков полностью 1, моделей 1"
  )
  if ($MockUnavailable) { $lines += 'ERROR expds.sp_dfine: SP_DFINE_UNAVAILABLE: модели не загрузились' }
  $log3 = Join-Child $Out 'main_v3.log'
  $log2 = Join-Child $Out 'main_v2.log'
  Set-Content -Path $log3 -Value ($lines -join "`n") -Encoding UTF8
  Set-Content -Path $log2 -Value 'INFO v2 done' -Encoding UTF8
  Set-Content -Path (Join-Child $Out 'main_v3.err') -Value '' -Encoding UTF8
  Set-Content -Path (Join-Child $Out 'main_v2.err') -Value '' -Encoding UTF8
  $r2 = Join-Child $Out 'res_v2'
  $r3 = Join-Child $Out 'res_v3'
  New-Item -ItemType Directory -Force -Path $r2, $r3 | Out-Null
  $gtCoords = New-Square 4100000 7500000
  if ($MockSporMatches -le 0) { $predCoords = New-Square 4300000 7600000 } else { $predCoords = $gtCoords }
  Write-Fc (Join-Child $r2 'result.geojson') (New-Square 4100100 7500100) 'sp_or'
  if ($MockIdentical) {
    Copy-Item (Join-Child $r2 'result.geojson') (Join-Child $r3 'result.geojson') -Force
  } else {
    Write-Fc (Join-Child $r3 'result.geojson') $predCoords 'sp_or'
  }
  Write-Fc (Join-Child $Out 'gt.geojson') $gtCoords 'sp_or'
  $Gt = Join-Child $Out 'gt.geojson'
  $selfRam = Get-RamMiB -ProcId $PID -DockerName ''
  $selfV = Get-VramMiB
  $ramText = 'n/a'
  if ($null -ne $selfRam) { $ramText = "$selfRam MiB" }
  $vramText = 'n/a'
  if ($null -ne $selfV) { $vramText = "$selfV MiB" }
  Write-Log "mock: wall 0 s, peak RAM $ramText, peak VRAM $vramText, torch.cuda.max_memory_allocated n/a"
} else {
  if (-not (Test-Path $Regions)) { throw "нет регионов: $Regions" }
  Test-NoReparse $Regions
  foreach ($f in @($ZipV2, $ZipV3)) { if (-not (Test-Path $f)) { throw "нет архива: $f" } }
  $sumFile = Join-Child $Bot7 'SHA256SUMS'
  if (Test-Path $sumFile) {
    $sums = @{}
    Get-Content -Path $sumFile | ForEach-Object {
      if (-not $_) { return }
      $parts = $_.Split(@(' ', "`t"), [StringSplitOptions]::RemoveEmptyEntries)
      if ($parts.Count -ge 2) { $sums[$parts[1].Trim().TrimStart('*')] = $parts[0].ToLower() }
    }
    foreach ($f in @($ZipV2, $ZipV3)) {
      $leaf = Split-Path -Leaf $f
      if (-not $sums.ContainsKey($leaf)) { throw "нет в SHA256SUMS: $leaf" }
      $got = (Get-FileHash -Algorithm SHA256 $f).Hash.ToLower()
      if ($got -ne $sums[$leaf]) { throw "SHA256 не совпал: $leaf" }
      Write-Log "sha ok $leaf"
    }
  }
  foreach ($pair in @(@('v2', $ZipV2), @('v3', $ZipV3))) {
    $tag = $pair[0]
    $zip = $pair[1]
    $dest = Join-Child $Out ("zip_{0}" -f $tag)
    $mainPy = Join-Child $dest 'inference'
    $mainPy = Join-Child $mainPy 'main.py'
    if (-not (Test-Path $mainPy)) {
      New-Item -ItemType Directory -Force -Path $dest | Out-Null
      & tar -xf $zip -C $dest
      if ($LASTEXITCODE -ne 0) { throw "tar не распаковал $zip" }
    }
  }
  $cfg = Join-Child (Join-Child $Out 'zip_v3') 'configs'
  $cfg = Join-Child $cfg 'default.toml'
  $enabled = $false
  if (Test-Path $cfg) {
    $pyChk = @'
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
text = open(sys.argv[1], encoding="utf-8").read()
cfg = tomllib.loads(text)
print("SP_DFINE_ENABLED=" + ("1" if (cfg.get("sp_dfine") or {}).get("enabled") else "0"))
'@
    $chk = Join-Child $Out 'chk_cfg.py'
    Set-Content -Path $chk -Value $pyChk -Encoding UTF8
    $outChk = & $Py $chk $cfg 2>&1 | Out-String
    if ($outChk -match 'SP_DFINE_ENABLED=1') { $enabled = $true }
  }
  if (-not $enabled) { throw "в zip v3 [sp_dfine] enabled не true — D-FINE не запустится" }
  if ($Mode -eq 'venv' -and $SetupTf) {
    & $Py -m pip install --no-deps --target $TfDir transformers==5.21.0 tokenizers==0.23.3 regex==2026.9.29 typer==0.27.3 rich==15.0.0 markdown-it-py==4.2.0 mdurl==0.1.2 pygments==2.21.0 shellingham==1.5.4 annotated-doc==0.0.5
    if ($LASTEXITCODE -ne 0) { throw "pip --target не удался" }
  }
  Invoke-OneRun 'v2' (Join-Child $Out 'zip_v2') (Join-Child $Out 'res_v2')
  Invoke-OneRun 'v3' (Join-Child $Out 'zip_v3') (Join-Child $Out 'res_v3')
  if (-not $Gt) {
    $tools = Join-Child $Out 'tools'
    $tar = Join-Child $Bot7 'tools_v3e2e.tar'
    if ((Test-Path $tar) -and -not (Test-Path (Join-Child $tools 'e2e_score.py'))) {
      & tar -xf $tar -C $Out
    }
    $repo = Join-Child $tools 'repo'
    if (Test-Path (Join-Child $repo 'expds')) {
      $Gt = Join-Child $Out 'gt.geojson'
      Push-Location $repo
      try { & $Py -m expds.gt --root $Regions --out $Gt } finally { Pop-Location }
    }
  }
}

# --- проверки логов и хешей ---
$logFiles = @()
foreach ($name in @('main_v2.log', 'main_v2.err', 'main_v3.log', 'main_v3.err', 'dfine_probe.txt', 'time.txt')) {
  $p = Join-Child $Out $name
  if (Test-Path $p) { $logFiles += $p }
}
if ($logFiles.Count -eq 0) { Add-Fail 'нет логов прогона' }
$unavail = $null
if ($logFiles.Count -gt 0) {
  $unavail = Select-String -Path $logFiles -Pattern 'SP_DFINE_UNAVAILABLE' -SimpleMatch -CaseSensitive -ErrorAction SilentlyContinue
}
if ($unavail) { Add-Fail ("SP_DFINE_UNAVAILABLE: " + $unavail[0].Line.Trim()) }
$ndet = $null
$v3log = Join-Child $Out 'main_v3.log'
if (Test-Path $v3log) {
  $tot = Select-String -Path $v3log -Pattern 'sp_dfine n_det=(\d+) total' | Select-Object -Last 1
  if (-not $tot) { Add-Fail 'нет строки sp_dfine n_det=… total' }
  else {
    $ndet = [int]$tot.Matches[0].Groups[1].Value
    Write-Log "n_det total = $ndet"
    if ($ndet -eq 0) { Add-Fail 'sp_dfine n_det total = 0' }
  }
} else {
  Add-Fail 'нет main_v3.log'
}
$rv2 = Join-Child (Join-Child $Out 'res_v2') 'result.geojson'
$rv3 = Join-Child (Join-Child $Out 'res_v3') 'result.geojson'
if (-not (Test-Path $rv2) -or -not (Test-Path $rv3)) {
  Add-Fail 'нет result.geojson v2 или v3'
} else {
  $h2 = (Get-FileHash -Algorithm SHA256 $rv2).Hash
  $h3 = (Get-FileHash -Algorithm SHA256 $rv3).Hash
  Write-Log "sha256 v2=$h2"
  Write-Log "sha256 v3=$h3"
  if ($h2 -eq $h3) { Add-Fail 'result v3 идентичен v2' }
}

# --- F1: весь файл и срез markup sp_or ---
if (-not $Metric) {
  $cands = @(
    (Join-Child $Out 'tools\compute_metrics.py'),
    (Join-Path -Path $PSScriptRoot -ChildPath '..\..\..\..\..\..\materials\compute_metrics.py')
  )
  foreach ($c in $cands) {
    if (Test-Path $c) { $Metric = (Resolve-Path $c).Path; break }
  }
}
if (-not $Gt -or -not (Test-Path $Gt)) { Add-Fail 'нет эталона GeoJSON (-Gt или expds.gt)' }
elseif (-not $Metric -or -not (Test-Path $Metric)) { Add-Fail 'нет compute_metrics.py (-Metric)' }
elseif ((Test-Path $rv3)) {
  $filt = Join-Child $Out 'filter_spor.py'
  @'
import json, subprocess, sys
from pathlib import Path
py, metric, pred, gt, out_dir = sys.argv[1:6]
out = Path(out_dir)

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def dump(obj, p):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

def run(p_pred, p_gt, dest):
    r = subprocess.run([py, metric, "--predictions", p_pred, "--ground-truth", p_gt, "--output-json", dest],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2000:] + r.stdout[-2000:])
        raise SystemExit(r.returncode)
    return json.loads(Path(dest).read_text(encoding="utf-8"))

pred_j = load(pred)
gt_j = load(gt)
all_j = run(pred, gt, str(out / "score_all.json"))
print("F1_ALL %.6f" % float(all_j.get("final_score") or 0))
spor_mark = {"sp_or", "sp_focal", "sp_focal2", "sp_arch"}
def keep(fc, marks):
    feats = [f for f in fc.get("features") or []
             if str((f.get("properties") or {}).get("markup_type") or "") in marks]
    return {"type": "FeatureCollection", "features": feats}
p_pred = out / "pred_spor.geojson"
p_gt = out / "gt_spor.geojson"
dump(keep(pred_j, spor_mark | {"sp_dfine"}), p_pred)
gt_feats = gt_j.get("features") or []
if any((f.get("properties") or {}).get("markup_type") for f in gt_feats):
    dump(keep(gt_j, spor_mark), p_gt)
else:
    dump(gt_j, p_gt)
spor = run(str(p_pred), str(p_gt), str(out / "score_spor.json"))
tp = sum(int((row or {}).get("tp") or 0) for row in (spor.get("per_class") or {}).values())
print("F1_SP_OR %.6f" % float(spor.get("final_score") or 0))
print("SP_OR_TP %d" % tp)
for cls, row in (spor.get("per_class") or {}).items():
    if int(row.get("num_ground_truth") or 0) or int(row.get("num_predictions") or 0):
        print("CLASS %s f1=%.4f tp=%s fp=%s fn=%s" % (cls, float(row.get("f1") or 0), row.get("tp"), row.get("fp"), row.get("fn")))
'@ | Set-Content -Path $filt -Encoding UTF8
  $scoreOut = & $Py $filt $Py $Metric $rv3 $Gt $Out 2>&1
  $scoreOut | ForEach-Object { Write-Log ("{0}" -f $_) }
  $tpLine = $scoreOut | Where-Object { "$_" -like 'SP_OR_TP *' } | Select-Object -Last 1
  if (-not $tpLine) { Add-Fail 'метрика sp_or не посчиталась' }
  else {
    $tp = [int]($tpLine.ToString().Split(' ')[1])
    $dfineRan = ($null -ne $ndet -and $ndet -gt 0 -and -not $unavail)
    if ($dfineRan -and $tp -le 0) { Add-Fail 'D-FINE запускался, а попаданий sp_or (tp) = 0' }
  }
}

Write-Log '----------'
if ($script:Fails.Count -eq 0) {
  Write-Log 'ИТОГ: PASS'
  exit 0
}
Write-Log ("ИТОГ: FAIL ({0})" -f $script:Fails.Count)
foreach ($item in $script:Fails) { Write-Log (" - {0}" -f $item) }
exit 1
