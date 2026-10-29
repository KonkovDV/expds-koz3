$ErrorActionPreference='Continue'
[Console]::OutputEncoding=[Text.Encoding]::UTF8
$Wt='C:\Sloi\wt_bot4_lo_337a27c'
git -C $Wt checkout -- configs/default.toml
$Py='C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
& $Py -c "import tomllib,pathlib; p=pathlib.Path(r'C:\Sloi\wt_bot4_lo_337a27c\configs\default.toml'); b=p.read_bytes(); tomllib.loads(b.decode()); print('toml_ok', b[:4])"
