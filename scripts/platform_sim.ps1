# Прогон архива в условиях платформы КОЗ №3 — жёстче, чем шаг 0.1 плана v4.
#
# Что моделируется (ТР 7.5.1, прил. 3; консультация №2, сл. 16):
#   * без сети;
#   * корневая ФС только для чтения: писать можно только в папку результата;
#   * код архива — только для чтения, рабочая папка = корень архива;
#   * пользователь не root и без записи в /etc/passwd (HOME = "/");
#   * лимиты: 96 ГБ памяти, 8192 процесса и потока, /dev/shm 16 ГБ;
#   * вход только для чтения, путь результата — как в примере ТР.
#
# Запуск из корня репозитория (PowerShell, Docker Desktop с GPU):
#   powershell -File scripts\platform_sim.ps1 -Image expds-runtime:2026-10-22 `
#       -Data F:\expds3\mock_val -Out F:\expds3\work\platform_sim
# Архив распаковывается во временную папку, как это делает платформа.

param(
    [Parameter(Mandatory = $true)][string]$Image,
    [Parameter(Mandatory = $true)][string]$Data,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Zip = "dist\submission.zip",
    [switch]$NoGpu
)

$ErrorActionPreference = "Stop"
$code = Join-Path $env:TEMP ("expds_sim_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
Expand-Archive -Path $Zip -DestinationPath $code
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $Out "result.geojson")

$gpu = @("--gpus", "all")
if ($NoGpu) { $gpu = @() }
$start = Get-Date
docker run --rm @gpu --network none --read-only --user 12345:12345 `
    --memory 96g --pids-limit 8192 --shm-size 16g `
    -v "${code}:/app:ro" -v "${Data}:/data/input:ro" -v "${Out}:/data/output" -w /app `
    $Image python inference/main.py /data/input /data/output/result.geojson 2>&1 | Tee-Object -FilePath (Join-Path $Out "run.log")
$sec = ((Get-Date) - $start).TotalSeconds
Write-Host ("время контейнера: {0:N0} с" -f $sec)

$py = ".venv\Scripts\python.exe"
& $py -X utf8 -m expds.writer check (Join-Path $Out "result.geojson")
$left = Get-ChildItem -Force $Out | Where-Object { $_.Name -notin @("result.geojson", "run.log") }
if ($left) { Write-Warning ("в папке результата остались файлы: " + ($left.Name -join ", ")) }
Select-String -Path (Join-Path $Out "run.log") -Pattern "среда:|опись входа|ERROR|Traceback|Fatal Python error|нет файлов моделей|без CRS" |
    ForEach-Object { $_.Line }
Remove-Item -Recurse -Force $code
