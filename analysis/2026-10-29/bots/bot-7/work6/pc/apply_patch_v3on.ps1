# bot-7, 29.10.2026 — apply_patch_v3on.ps1: собрать submission_bot7cand_v3on_bc5aa15.zip из старого v3on_36af6b7.zip + патч.
# Только диск C:. Совместим с Windows PowerShell 5.1. Запуск:
#   powershell -ExecutionPolicy Bypass -File C:\Sloi\pc_jobs\v3e2e\kit\apply_patch_v3on.ps1
# Проверяет sha256 старого zip, патча и манифеста; результат проверяется по манифесту (sha256/CRC/размер каждого члена).
param(
  [string]$OldZip   = 'C:\Sloi\pc_jobs\v3e2e\kit\submission_bot7cand_v3on_36af6b7.zip',
  [string]$PatchZip = 'C:\Sloi\pc_jobs\v3e2e\kit\patch_v3on_341a2394.zip',
  [string]$Manifest = '',
  [string]$OutZip   = 'C:\Sloi\pc_jobs\v3e2e\kit\submission_bot7cand_v3on_bc5aa15.zip',
  [string]$SumsFile = 'C:\Sloi\pc_jobs\v3e2e\kit\SHA256SUMS',
  [switch]$Force
)
$ErrorActionPreference = 'Stop'
$OLD_SHA   = '9ec25962ed70ba258c5c4156f443198072c48fe5449e5cc9aa10d8f7e3bd74e9'
$PATCH_SHA = '3503906ddf1e1edbc18db93447603d6a14adcef9055b840e521fb44ee56e672f'
$MAN_SHA   = '77a0dd82ccd7bfa2db9964580d7e344772d2deff8f4b1de708540a73bb288053'
$NEW_SHA   = '341a23941014f41d52afcb47887a64bec9561627ce16b95c31e5b0ab0ff6d4bd'
$NEW_NAME  = 'submission_bot7cand_v3on_bc5aa15.zip'
function Fail($m) { Write-Host "FAIL: $m" -ForegroundColor Red; exit 1 }
function Sha($p) { (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLower() }
if (-not $Manifest) { $Manifest = Join-Path (Split-Path -Parent $PatchZip) 'manifest_v3on_341a2394.txt' }
foreach ($p in @($OldZip, $PatchZip, $Manifest, $OutZip, $SumsFile)) {
  if (-not $p.ToUpper().StartsWith('C:\')) { Fail "путь не на C:\ : $p" }
}
foreach ($p in @($OldZip, $PatchZip, $Manifest)) { if (-not (Test-Path -LiteralPath $p)) { Fail "нет файла: $p" } }
if ((Test-Path -LiteralPath $OutZip) -and -not $Force) { Fail "уже существует: $OutZip (используйте -Force)" }
$free = (Get-PSDrive C).Free
if ($free -lt 1.5GB) { Fail ("мало места на C: {0:N1} ГБ (< 1.5 ГБ)" -f ($free / 1GB)) }
Write-Host 'sha256 входных файлов...'
$s = Sha $OldZip;   if ($s -ne $OLD_SHA)   { Fail "старый zip sha256 $s != $OLD_SHA" }
$s = Sha $PatchZip; if ($s -ne $PATCH_SHA) { Fail "патч sha256 $s != $PATCH_SHA" }
$s = Sha $Manifest; if ($s -ne $MAN_SHA)   { Fail "манифест sha256 $s != $MAN_SHA" }
Write-Host 'OK: старый zip, патч, манифест'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$part = "$OutZip.part"
if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force }
Copy-Item -LiteralPath $OldZip -Destination $part
$pz = [System.IO.Compression.ZipFile]::OpenRead($PatchZip)
$fs = [System.IO.File]::Open($part, 'Open', 'ReadWrite')
$za = New-Object System.IO.Compression.ZipArchive($fs, [System.IO.Compression.ZipArchiveMode]::Update)
$nput = 0; $ndel = 0
try {
  foreach ($e in $pz.Entries) {
    $n = $e.FullName
    if ($n.StartsWith('_b7patch/')) { continue }
    if ($n.StartsWith('/') -or $n.Contains('\') -or $n.Contains('..') -or $n.Contains(':')) { Fail "небезопасное имя в патче: $n" }
    if ($n.EndsWith('/')) { continue }
    $old = @($za.Entries | Where-Object { $_.FullName -eq $n })
    foreach ($o in $old) { $o.Delete() }
    $ne = $za.CreateEntry($n, [System.IO.Compression.CompressionLevel]::Optimal)
    $ne.LastWriteTime = $e.LastWriteTime
    $src = $e.Open(); $dst = $ne.Open(); $src.CopyTo($dst); $dst.Close(); $src.Close()
    $nput++
  }
  $de = $pz.GetEntry('_b7patch/DELETE.txt')
  if ($de) {
    $r = New-Object System.IO.StreamReader($de.Open(), [System.Text.Encoding]::UTF8)
    $txt = $r.ReadToEnd(); $r.Close()
    foreach ($line in ($txt -split "`n")) {
      $line = $line.Trim(); if (-not $line) { continue }
      foreach ($o in @($za.Entries | Where-Object { $_.FullName -eq $line })) { $o.Delete(); $ndel++ }
    }
  }
} finally { $za.Dispose(); $fs.Close(); $pz.Dispose() }
Write-Host "записано членов: $nput, удалено: $ndel"
$got = Sha $part
if ($got -eq $NEW_SHA) {
  Write-Host 'zip побайтно совпадает с эталоном bot-7'
} else {
  Write-Host "sha256 zip $got != эталон (ожидаемо: ZipArchive пишет иначе) — проверка по манифесту..."
  $want = @{}
  foreach ($line in [System.IO.File]::ReadAllLines($Manifest, [System.Text.Encoding]::UTF8)) {
    if (-not $line -or $line.StartsWith('#')) { continue }
    $f = $line.Split("`t", 4); $want[$f[3]] = $f
  }
  $sha = [System.Security.Cryptography.SHA256]::Create()
  $z = [System.IO.Compression.ZipFile]::OpenRead($part)
  $seen = @{}; $bad = 0
  try {
    foreach ($e in $z.Entries) {
      $n = $e.FullName
      if ($seen.ContainsKey($n)) { Write-Host "дубликат: $n"; $bad++; continue }
      $seen[$n] = 1
      if (-not $want.ContainsKey($n)) { Write-Host "лишний член: $n"; $bad++; continue }
      $w = $want[$n]
      if ([int64]$e.Length -ne [int64]$w[2]) { Write-Host "размер: $n $($e.Length) != $($w[2])"; $bad++; continue }
      if ($e.PSObject.Properties.Name -contains 'Crc32') {
        $c = '{0:x8}' -f [uint32]$e.Crc32
        if ($c -ne $w[1]) { Write-Host "crc32: $n $c != $($w[1])"; $bad++; continue }
      }
      # rev2: 0-байтные члены — sha256 пустой строки без чтения потока (PS 5.1 / .NET Framework давал неверный хэш)
      if ([int64]$e.Length -eq 0) { $h = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855' }
      else {
        $st = $e.Open(); $ms = New-Object System.IO.MemoryStream
        try { $st.CopyTo($ms) } finally { $st.Close() }
        $h = ([System.BitConverter]::ToString($sha.ComputeHash($ms.ToArray())) -replace '-', '').ToLower()
        $ms.Dispose()
      }
      if ($h -ne $w[0]) { Write-Host "sha256: $n $h != $($w[0])"; $bad++ }
    }
  } finally { $z.Dispose() }
  foreach ($k in $want.Keys) { if (-not $seen.ContainsKey($k)) { Write-Host "нет члена: $k"; $bad++ } }
  if ($bad -gt 0) { Remove-Item -LiteralPath $part -Force; Fail "проверка по манифесту: ошибок $bad" }
  Write-Host "OK: все $($want.Count) членов совпадают с манифестом (sha256/размер/CRC)"
}
if (Test-Path -LiteralPath $OutZip) { Remove-Item -LiteralPath $OutZip -Force }
Move-Item -LiteralPath $part -Destination $OutZip
$got = Sha $OutZip
$leaf = Split-Path -Leaf $OutZip
$enc = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText("$OutZip.sha256", "$got  $leaf`n", $enc)
$lines = @()
if (Test-Path -LiteralPath $SumsFile) {
  $lines = @([System.IO.File]::ReadAllLines($SumsFile) | Where-Object { $_ -and -not ($_ -match ('\s\*?' + [regex]::Escape($leaf) + '$')) })
}
$lines += "$got  $leaf"
[System.IO.File]::WriteAllText($SumsFile, (($lines -join "`n") + "`n"), $enc)
Write-Host "ГОТОВО: $OutZip"
Write-Host "sha256 $got (записан в $OutZip.sha256 и $SumsFile)"
Write-Host 'Дальше (НОВЫЙ -Out, иначе старая распаковка будет переиспользована):'
Write-Host "  powershell -ExecutionPolicy Bypass -File C:\Sloi\pc_jobs\v3e2e\kit\pc_e2e_v3.ps1 -ZipV3 `"$OutZip`" -Out C:\Sloi\pc_jobs\v3e2e\out3"
