# Stage v2 (bot-1 29.10 10:25 MSK): inventory + live-ref check + Move-Item (never delete). Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File f_expds3_trash.ps1 [-DoMove]
# Fixes vs v1: process check no longer matches this script's own name (v1 always aborted);
#   ref search covers all of C:\Sloi (excl TRASH/.venv/.git/site-packages) incl. analysis/*.py;
#   live = any script/config (.py .ps1 .cmd .bat .sh .yml .yaml .toml .cfg .ini .json) except .json under notes/docs/reports/results;
#   hard must-keep guard; top-level F:\expds3 inventory; moved.log reset per run.
param([switch]$DoMove)
$ErrorActionPreference = 'Continue'
$E = 'F:\expds3'; $W = 'F:\expds3\work'; $T = 'F:\expds3\TRASH'
$out = 'C:\Sloi\pc_jobs\expds3_trash_out'; New-Item -ItemType Directory -Force $out | Out-Null
function SizeOf($p){ if(Test-Path -LiteralPath $p -PathType Leaf){(Get-Item -LiteralPath $p).Length}else{[double](Get-ChildItem -LiteralPath $p -Recurse -File -Force -EA SilentlyContinue | Measure-Object Length -Sum).Sum} }
$free0 = (Get-PSDrive F).Free
$mustKeep = @('spor_focal_oof','li_plus_oof','li_hard2_oof','stack_base','stack_verify_sp','stack_arch','stack_verify_sp_arch','mock_val','cache','gt_train_v2','fusion','mosaic_li',
  'spor_gor_oof','li_gor_mask_oof','geo_cur','ae_fort_mask_oof','li_fort_mask_oof','gor_phys','sp_arch_oof','or_arch_oof','surf_arch_sp','surf_arch_or',
  'li_fort_mask','ae_fort_mask','surf_arch','li_gor_mask','spor_gor_wide','TRASH')
function Inv($dir){ Get-ChildItem -LiteralPath $dir -Force | ForEach-Object {
  $s = SizeOf $_.FullName
  $n = if($_.PSIsContainer){@(Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Force -EA SilentlyContinue).Count}else{1}
  [pscustomobject]@{Name=$_.Name;Type=$(if($_.PSIsContainer){'dir'}else{'file'});GB=[math]::Round($s/1GB,3);Bytes=[int64]$s;Files=$n;Modified=$_.LastWriteTime.ToString('MM-dd HH:mm')}
} | Sort-Object Bytes -Descending }
Inv $E | Export-Csv "$out\inventory_top.csv" -NoTypeInformation -Encoding UTF8
Inv $W | Export-Csv "$out\inventory.csv" -NoTypeInformation -Encoding UTF8
$cand = @('li_opn5_oof','li_hard3_oof','li_hard3_nocp_oof','li_cur3_oof','li_hard_oof','li_all_oof','li_tin_oof','li_ema_oof','li_dev_oof','li_focal_oof','li_thin','li_gor_wide','li_gor_center',
 'spor_hardneg','spor_r1_oof','spor_all_oof','spor_center_oof','spor_large',
 'spor_focal_r05','spor_focal_r34','spor_focal_photo','spor_focal_scale','spor_focal_hrrep','spor_focal_wide',
 'ae_focal_oof','ae_focal_r1','sp_arch_mask','maggr_finds',
 'stack_rerank_first.geojson','stack_rerank_kurgans.geojson','stack_arch_gate.geojson',
 'sub_unzip','archive_test','wheels','dist_test','zipdiff','__pycache__','bench_in','bench_mix')
$cand += @(Get-ChildItem -LiteralPath $W -Force | Where-Object { $_.Name -match '^(smoke_|probe_|mock_out)' } | ForEach-Object { $_.Name })
$cand = $cand | Select-Object -Unique | Where-Object { Test-Path -LiteralPath (Join-Path $W $_) }
$cand = $cand | Where-Object { $mustKeep -notcontains ([IO.Path]::GetFileNameWithoutExtension($_)) -and $mustKeep -notcontains $_ }
$files = Get-ChildItem 'C:\Sloi' -Recurse -File -Force -EA SilentlyContinue -Include *.py,*.ps1,*.cmd,*.bat,*.sh,*.json,*.yml,*.yaml,*.toml,*.cfg,*.ini,*.txt,*.md |
  Where-Object { $_.Length -lt 3MB -and $_.FullName -notmatch '\\(\.venv|venv|\.git|TRASH|__pycache__|node_modules|site-packages|expds3_trash_out)\\' }
"scanned_files=$(@($files).Count)" | Out-File "$out\scan.txt"
$rows = foreach($c in $cand){
  $base = if($c -match '\.geojson$'){[IO.Path]::GetFileNameWithoutExtension($c)}else{$c}
  $pat = '(?<![A-Za-z0-9_])' + [regex]::Escape($base) + '(?![A-Za-z0-9_])'
  $hits = @($files | Select-String -Pattern $pat -EA SilentlyContinue)
  $live = @($hits | Where-Object { $_.Path -notmatch '\.(md|txt)$' -and -not ($_.Path -match '\.json$' -and $_.Path -match '\\(notes|docs|reports?|results?)\\') })
  [pscustomobject]@{Name=$c;GB=[math]::Round((SizeOf (Join-Path $W $c))/1GB,3);AllHits=$hits.Count;LiveHits=$live.Count;
    LiveWhere=(($live|ForEach-Object{"$($_.Path):$($_.LineNumber)"})|Select-Object -First 6) -join ' ; ';
    NoteWhere=(($hits|Where-Object{$live -notcontains $_}|ForEach-Object{"$($_.Path):$($_.LineNumber)"})|Select-Object -First 3) -join ' ; '}
}
$rows | Export-Csv "$out\refcheck.csv" -NoTypeInformation -Encoding UTF8
$procs = @(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match 'F:[\\/]+expds3' } | Select-Object ProcessId,Name,CommandLine)
$procs | Export-Csv "$out\procs.csv" -NoTypeInformation -Encoding UTF8
"procs=$($procs.Count)" | Out-File -Append "$out\scan.txt"
if($DoMove){
  "run $(Get-Date -Format o)" | Out-File "$out\moved.log"
  if($procs.Count -gt 0){ "ABORT: processes reference F:\expds3" | Out-File -Append "$out\moved.log"; exit 2 }
  New-Item -ItemType Directory -Force $T | Out-Null
  foreach($r in $rows){
    if($r.LiveHits -eq 0 -and $r.AllHits -ge 0){
      $src = Join-Path $W $r.Name; $dst = Join-Path $T $r.Name
      if(Test-Path -LiteralPath $dst){ "SKIP exists $dst" | Out-File "$out\moved.log" -Append; continue }
      try { Move-Item -LiteralPath $src -Destination $dst -EA Stop; "MOVED $($r.Name) $($r.GB) GB" | Out-File "$out\moved.log" -Append }
      catch { "FAIL $($r.Name): $_" | Out-File "$out\moved.log" -Append }
    } else { "KEPT(live ref) $($r.Name): $($r.LiveWhere)" | Out-File "$out\moved.log" -Append }
  }
}
"free_before_GB=$([math]::Round($free0/1GB,2)) free_after_GB=$([math]::Round((Get-PSDrive F).Free/1GB,2))" | Out-File "$out\free.txt"
