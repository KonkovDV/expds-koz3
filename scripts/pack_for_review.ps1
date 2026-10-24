# Собирает небольшой архив для разбора вне этой машины: материалы организаторов,
# структуру публичной выборки, разметку и сводку растров геофизики.
# Сами растры и LAS не берутся. Ключи и журналы синхронизации S3 не берутся.
#
#   powershell -ExecutionPolicy Bypass -File scripts\pack_for_review.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\pack_for_review.ps1 -WithGeoRasters
#
# -WithGeoRasters добавляет растры MagGr, EtGr и TerFl трейна (несколько сотен МБ).
param(
    [string]$Materials = "C:\Sloi\files\materials",
    [string]$Train = "F:\expds3\train-expds-3",
    [string]$Out = "C:\Sloi\review_pack.zip",
    [switch]$WithGeoRasters
)

$ErrorActionPreference = "Stop"
$stage = Join-Path $env:TEMP ("review_pack_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $stage | Out-Null

function Copy-Tree($src, $dst, [string[]]$patterns) {
    if (-not (Test-Path $src)) { Write-Warning "нет папки $src"; return }
    foreach ($p in $patterns) {
        Get-ChildItem -Path $src -Recurse -File -Filter $p | Where-Object {
            $_.Name -notmatch "(?i)sync|s3cfg|credentials|secret|key"
        } | ForEach-Object {
            $rel = $_.FullName.Substring($src.Length).TrimStart("\")
            $target = Join-Path $dst $rel
            New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
            Copy-Item $_.FullName $target
        }
    }
}

# 1. Материалы организаторов: документы, метрика, бейзлайн, структура публичной выборки.
Copy-Tree $Materials (Join-Path $stage "materials") @("*.pdf", "*.docx", "*.md", "*.txt", "*.py", "*.xlsx", "*.csv", "*.zip")

# 2. Разметка и UTM.json всех участков трейна (GeoJSON, без растров).
Copy-Tree $Train (Join-Path $stage "train_markup") @("*.geojson", "UTM.json")

# 3. Сводка растров геофизики без самих пикселей: каналы, тип, шаг, CRS.
$geoList = Join-Path $stage "geo_rasters.txt"
Get-ChildItem -Path $Train -Recurse -File -Include *.tif, *.tiff |
    Where-Object { $_.Name -match "MagGr|EtGr|TerFl|магнит|электро|термо" } |
    ForEach-Object { "{0}`t{1}" -f $_.FullName.Substring($Train.Length), $_.Length } |
    Set-Content -Encoding UTF8 $geoList

if ($WithGeoRasters) {
    Get-ChildItem -Path $Train -Recurse -File -Include *.tif, *.tiff |
        Where-Object { $_.Name -match "MagGr|EtGr|TerFl" } | ForEach-Object {
            $rel = $_.FullName.Substring($Train.Length).TrimStart("\")
            $target = Join-Path (Join-Path $stage "geo_rasters") $rel
            New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
            Copy-Item $_.FullName $target
        }
}

if (Test-Path $Out) { Remove-Item $Out }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $Out
Remove-Item -Recurse -Force $stage
$mb = [math]::Round((Get-Item $Out).Length / 1MB, 1)
Write-Host "готово: $Out ($mb МБ)"
