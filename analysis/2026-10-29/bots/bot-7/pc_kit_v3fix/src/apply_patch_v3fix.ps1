# bot-7, 29.10.2026 — apply_patch_v3fix.ps1: собрать submission_bot7cand_v3on_0876090.zip (v3on + фикс географической CRS)
# из лежащего на ПК submission_bot7cand_v3on_bc5aa15.zip + патч. Основа — apply_patch_v3on.ps1 rev2 (09104530…).
# Только диск C:. Совместим с Windows PowerShell 5.1. Запуск:
#   powershell -ExecutionPolicy Bypass -File C:\Sloi\pc_jobs\v3e2e\kit\apply_patch_v3fix.ps1
# Входной zip на ПК собран ZipArchive (apply_patch_v3on.ps1), поэтому его sha256 файла НЕ равен 341a2394… бота 7.
# Он проверяется ПО ЧЛЕНАМ по старому манифесту manifest_v3on_341a2394.txt: имена, число, размеры, CRC (если доступен)
# у всех членов; sha256 — у всех членов (по умолчанию) или, с -QuickInput, только у заменяемых/удаляемых патчем.
# Результат проверяется по новому манифесту (sha256/CRC/размер каждого члена).
param(
  [string]$Kit         = 'C:\Sloi\pc_jobs\v3e2e\kit',
  [string]$OldZip      = '',   # по умолчанию $Kit\submission_bot7cand_v3on_bc5aa15.zip
  [string]$OldManifest = '',   # по умолчанию $Kit\manifest_v3on_341a2394.txt
  [string]$PatchZip    = '',   # по умолчанию $Kit\patch_v3fix_@S8@.zip
  [string]$Manifest    = '',   # по умолчанию $Kit\manifest_v3fix_@S8@.txt
  [string]$OutZip      = '',   # по умолчанию $Kit\submission_bot7cand_v3on_0876090.zip
  [string]$SumsFile    = '',   # по умолчанию $Kit\SHA256SUMS
  [switch]$QuickInput,
  [switch]$Force
)
$ErrorActionPreference = 'Stop'
$OLDMAN_SHA = '@OLDMAN_SHA@'
$PATCH_SHA  = '@PATCH_SHA@'
$MAN_SHA    = '@MAN_SHA@'
$NEW_SHA    = '@NEW_SHA@'
$OLD_REF_SHA = '341a23941014f41d52afcb47887a64bec9561627ce16b95c31e5b0ab0ff6d4bd'   # zip бота 7 (на ПК другой — это норма)
$EMPTY_SHA  = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
function Fail($m) { Write-Host "FAIL: $m" -ForegroundColor Red; exit 1 }
function Sha($p) { (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLower() }
if (-not $OldZip)      { $OldZip      = Join-Path $Kit 'submission_bot7cand_v3on_bc5aa15.zip' }
if (-not $OldManifest) { $OldManifest = Join-Path $Kit 'manifest_v3on_341a2394.txt' }
if (-not $PatchZip)    { $PatchZip    = Join-Path $Kit 'patch_v3fix_@S8@.zip' }
if (-not $Manifest)    { $Manifest    = Join-Path $Kit 'manifest_v3fix_@S8@.txt' }
if (-not $OutZip)      { $OutZip      = Join-Path $Kit 'submission_bot7cand_v3on_0876090.zip' }
if (-not $SumsFile)    { $SumsFile    = Join-Path $Kit 'SHA256SUMS' }
foreach ($p in @($Kit, $OldZip, $OldManifest, $PatchZip, $Manifest, $OutZip, $SumsFile)) {
  if (-not $p.ToUpper().StartsWith('C:\')) { Fail "путь не на C:\ : $p" }
}
foreach ($p in @($OldZip, $OldManifest, $PatchZip, $Manifest)) { if (-not (Test-Path -LiteralPath $p)) { Fail "нет файла: $p" } }
if ((Test-Path -LiteralPath $OutZip) -and -not $Force) { Fail "уже существует: $OutZip (используйте -Force)" }
$free = (Get-PSDrive C).Free
if ($free -lt 1.5GB) { Fail ("мало места на C: {0:N1} ГБ (< 1.5 ГБ)" -f ($free / 1GB)) }
Write-Host 'sha256 патча и манифестов...'
$s = Sha $PatchZip;    if ($s -ne $PATCH_SHA)  { Fail "патч sha256 $s != $PATCH_SHA" }
$s = Sha $Manifest;    if ($s -ne $MAN_SHA)    { Fail "новый манифест sha256 $s != $MAN_SHA" }
$s = Sha $OldManifest; if ($s -ne $OLDMAN_SHA) { Fail "старый манифест sha256 $s != $OLDMAN_SHA" }
Write-Host 'OK: патч, новый и старый манифесты'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Read-Manifest($path) {
  $want = @{}
  foreach ($line in [System.IO.File]::ReadAllLines($path, [System.Text.Encoding]::UTF8)) {
    if (-not $line -or $line.StartsWith('#')) { continue }
    $f = $line.Split("`t", 4); $want[$f[3]] = $f
  }
  return $want
}
# Проверка zip по манифесту. $hashOnly = $null — sha256 у всех членов; иначе только у имён из хэш-таблицы $hashOnly.
function Test-ZipByManifest($zipPath, $want, $hashOnly) {
  $sha = [System.Security.Cryptography.SHA256]::Create()
  $z = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
  $seen = @{}; $bad = 0; $nh = 0
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
      if ($hashOnly -and -not $hashOnly.ContainsKey($n)) { continue }
      # 0-байтные члены — sha256 пустой строки без чтения потока (PS 5.1 / .NET Framework: ComputeHash на потоке давал неверный хэш);
      # остальные — через MemoryStream.ToArray()
      if ([int64]$e.Length -eq 0) { $h = $EMPTY_SHA }
      else {
        $st = $e.Open(); $ms = New-Object System.IO.MemoryStream
        try { $st.CopyTo($ms) } finally { $st.Close() }
        $h = ([System.BitConverter]::ToString($sha.ComputeHash($ms.ToArray())) -replace '-', '').ToLower()
        $ms.Dispose()
      }
      $nh++
      if ($h -ne $w[0]) { Write-Host "sha256: $n $h != $($w[0])"; $bad++ }
    }
  } finally { $z.Dispose(); $sha.Dispose() }
  foreach ($k in $want.Keys) { if (-not $seen.ContainsKey($k)) { Write-Host "нет члена: $k"; $bad++ } }
  return @($bad, $nh)
}

# --- 1. что меняет патч ---
$pz = [System.IO.Compression.ZipFile]::OpenRead($PatchZip)
$touch = @{}; $dels = @()
try {
  foreach ($e in $pz.Entries) {
    $n = $e.FullName
    if ($n.StartsWith('_b7patch/') -or $n.EndsWith('/')) { continue }
    if ($n.StartsWith('/') -or $n.Contains('\') -or $n.Contains('..') -or $n.Contains(':')) { Fail "небезопасное имя в патче: $n" }
    $touch[$n] = 1
  }
  $de = $pz.GetEntry('_b7patch/DELETE.txt')
  if ($de) {
    $r = New-Object System.IO.StreamReader($de.Open(), [System.Text.Encoding]::UTF8)
    $txt = $r.ReadToEnd(); $r.Close()
    foreach ($line in ($txt -split "`n")) { $line = $line.Trim(); if ($line) { $dels += $line; $touch[$line] = 1 } }
  }
} finally { $pz.Dispose() }
Write-Host "патч: заменить/добавить $($touch.Count - $dels.Count), удалить $($dels.Count)"

# --- 2. входной zip по членам (старый манифест) ---
$wantOld = Read-Manifest $OldManifest
if ($QuickInput) { Write-Host "входной zip: имена/число/размеры/CRC всех $($wantOld.Count) членов, sha256 — только затрагиваемых патчем ($($touch.Count))..." ; $r = Test-ZipByManifest $OldZip $wantOld $touch }
else { Write-Host "входной zip: полная проверка всех $($wantOld.Count) членов по старому манифесту (sha256/размер/CRC), ~1–3 мин..."; $r = Test-ZipByManifest $OldZip $wantOld $null }
if ($r[0] -gt 0) { Fail "входной zip не совпадает со старым манифестом: ошибок $($r[0]) ($OldZip)" }
Write-Host "OK: входной zip = v3on_bc5aa15 по членам ($($wantOld.Count) членов, sha256 посчитан у $($r[1]))"

# --- 3. применение патча к копии ---
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
    if ($n.StartsWith('_b7patch/') -or $n.EndsWith('/')) { continue }
    $old = @($za.Entries | Where-Object { $_.FullName -eq $n })
    foreach ($o in $old) { $o.Delete() }
    $ne = $za.CreateEntry($n, [System.IO.Compression.CompressionLevel]::Optimal)
    $ne.LastWriteTime = $e.LastWriteTime
    $src = $e.Open(); $dst = $ne.Open(); $src.CopyTo($dst); $dst.Close(); $src.Close()
    $nput++
  }
  foreach ($line in $dels) {
    foreach ($o in @($za.Entries | Where-Object { $_.FullName -eq $line })) { $o.Delete(); $ndel++ }
  }
} finally { $za.Dispose(); $fs.Close(); $pz.Dispose() }
Write-Host "записано членов: $nput, удалено: $ndel"

# --- 4. результат по новому манифесту (всегда полностью) ---
$got = Sha $part
if ($got -eq $NEW_SHA) { Write-Host 'zip побайтно совпадает с эталоном bot-7' }
else { Write-Host "sha256 zip $got != эталон $NEW_SHA (ожидаемо: ZipArchive пишет иначе) — проверка по манифесту..." }
$want = Read-Manifest $Manifest
$r = Test-ZipByManifest $part $want $null
if ($r[0] -gt 0) { Remove-Item -LiteralPath $part -Force; Fail "проверка по новому манифесту: ошибок $($r[0])" }
Write-Host "OK: все $($want.Count) членов совпадают с новым манифестом (sha256/размер/CRC)"

# --- 5. итог: переименование, sha256 в .sha256 и SHA256SUMS кита (для pc_e2e_v3.ps1) ---
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
Write-Host "  powershell -ExecutionPolicy Bypass -File $Kit\pc_e2e_v3.ps1 -ZipV3 `"$OutZip`" -Out C:\Sloi\pc_jobs\v3e2e\out_v3fix"
