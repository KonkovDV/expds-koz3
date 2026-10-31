# Stage: inventory + live-ref check + Move-Item (never delete). Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File f_expds3_trash.ps1 [-DoMove]
param([switch]$DoMove)
$ErrorActionPreference = 'Continue'
$W = 'F:\expds3\work'; $T = 'F:\expds3\TRASH'
$out = "$env:USERPROFILE\expds3_trash_out"; New-Item -ItemType Directory -Force $out | Out-Null
function SizeOf($p){ if(Test-Path $p -PathType Leaf){(Get-Item $p).Length}else{(Get-ChildItem $p -Recurse -File -Force -EA SilentlyContinue | Measure-Object Length -Sum).Sum} }
$free0 = (Get-PSDrive F).Free
# 1. inventory of work top-level
Get-ChildItem $W -Force | ForEach-Object {
  $s = SizeOf $_.FullName
  $n = if($_.PSIsContainer){(Get-ChildItem $_.FullName -Recurse -File -Force -EA SilentlyContinue).Count}else{1}
  [pscustomobject]@{Name=$_.Name;Type=$(if($_.PSIsContainer){'dir'}else{'file'});GB=[math]::Round($s/1GB,3);Files=$n;Modified=$_.LastWriteTime.ToString('MM-dd HH:mm')}
} | Sort-Object GB -Descending | Export-Csv "$out\inventory.csv" -NoTypeInformation -Encoding UTF8
# 2. candidates
$cand = @('li_opn5_oof','li_hard3_oof','li_hard3_nocp_oof','li_cur3_oof','li_hard_oof','li_all_oof','li_tin_oof','li_ema_oof','li_dev_oof','li_focal_oof','li_thin','li_gor_wide','li_gor_center',
 'spor_hardneg','spor_r1_oof','spor_all_oof','spor_center_oof','spor_large',
 'spor_focal_r05','spor_focal_r34','spor_focal_photo','spor_focal_scale','spor_focal_hrrep','spor_focal_wide',
 'ae_focal_oof','ae_focal_r1','sp_arch_mask','maggr_finds',
 'stack_rerank_first.geojson','stack_rerank_kurgans.geojson','stack_arch_gate.geojson',
 'sub_unzip','archive_test','wheels','dist_test','zipdiff','__pycache__','bench_in','bench_mix')
$cand += (Get-ChildItem $W -Force | Where-Object { $_.Name -match '^(smoke_|probe_|mock_out)' }).Name
$cand = $cand | Select-Object -Unique | Where-Object { Test-Path (Join-Path $W $_) }
# 3. live-ref search (exact folder name as token) in live script dirs
$roots = @('C:\Sloi\expds3c','C:\Sloi\pc_jobs','C:\Sloi\work59','C:\Sloi\pkg','C:\Sloi\geo211','C:\Sloi\wt_s4','C:\Sloi\wt_bots') | Where-Object { Test-Path $_ }
$files = Get-ChildItem $roots -Recurse -File -Force -EA SilentlyContinue -Include *.py,*.ps1,*.cmd,*.bat,*.sh,*.json,*.yml,*.yaml,*.toml,*.cfg,*.txt,*.md |
  Where-Object { $_.Length -lt 3MB -and $_.FullName -notmatch '\\(\.venv|\.git|TRASH|__pycache__|node_modules)\\' }
$rows = foreach($c in $cand){
  $base = [IO.Path]::GetFileNameWithoutExtension($c); if($c -notmatch '\.geojson$'){ $base = $c }
  $pat = '(?<![A-Za-z0-9_])' + [regex]::Escape($base) + '(?![A-Za-z0-9_])'
  $hits = $files | Select-String -Pattern $pat -EA SilentlyContinue
  $live = $hits | Where-Object { $_.Path -notmatch '\\(notes|docs|analysis|reports?)\\' -and $_.Path -notmatch '\.(md|txt)$' }
  [pscustomobject]@{Name=$c;GB=[math]::Round((SizeOf (Join-Path $W $c))/1GB,3);AllHits=@($hits).Count;LiveHits=@($live).Count;LiveWhere=(($live|ForEach-Object{"$($_.Path):$($_.LineNumber)"})|Select-Object -First 5) -join ' ; '}
}
$rows | Export-Csv "$out\refcheck.csv" -NoTypeInformation -Encoding UTF8
# process check
$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'expds3' } | Select-Object ProcessId,Name,CommandLine
$procs | Export-Csv "$out\procs.csv" -NoTypeInformation -Encoding UTF8
# 4. move only zero-live-ref candidates
if($DoMove){
  if(@($procs).Count -gt 0){ "ABORT: processes reference expds3" | Out-File "$out\moved.log"; exit 2 }
  New-Item -ItemType Directory -Force $T | Out-Null
  foreach($r in $rows){
    if($r.LiveHits -eq 0){
      $src = Join-Path $W $r.Name; $dst = Join-Path $T $r.Name
      if(Test-Path $dst){ "SKIP exists $dst" | Out-File "$out\moved.log" -Append; continue }
      try { Move-Item -LiteralPath $src -Destination $dst -EA Stop; "MOVED $($r.Name) $($r.GB) GB" | Out-File "$out\moved.log" -Append }
      catch { "FAIL $($r.Name): $_" | Out-File "$out\moved.log" -Append }
    } else { "KEPT(live ref) $($r.Name): $($r.LiveWhere)" | Out-File "$out\moved.log" -Append }
  }
}
"free_before_GB=$([math]::Round($free0/1GB,2)) free_after_GB=$([math]::Round((Get-PSDrive F).Free/1GB,2))" | Out-File "$out\free.txt"
