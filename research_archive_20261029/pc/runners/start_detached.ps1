param([Parameter(Mandatory)]$Script,[Parameter(Mandatory)]$Job)
$Root = "C:\Sloi\pc_jobs\$Job"
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$log = "$Root\launcher.log"
Start-Process -FilePath powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$Script) -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput "$Root\stdout.log" -RedirectStandardError "$Root\stderr.log"
"STARTED $(Get-Date -Format o) script=$Script" | Set-Content -Encoding UTF8 "$Root\STATUS.txt"
Add-Content -Encoding UTF8 $log -Value "STARTED $(Get-Date -Format o) $Script"
