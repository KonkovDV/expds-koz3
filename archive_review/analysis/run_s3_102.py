"""Докачка 102_Гнёздово_Заольшанская официальным скриптом. Ключи не печатаем."""
import os
import subprocess
from pathlib import Path

py = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe")
script = Path(r"C:\Sloi\files\materials\s3_script\s3_sync_train-expds-3.py")
log = Path(r"F:\expds3\work\s3_102.log")
cmd = [
    str(py), "-X", "utf8", str(script),
    "--mode", "update",
    "--local-dir", r"F:\expds3\train-expds-3",
    "--folder", "102_Гнёздово_Заольшанская",
    "--threads", "2",
    "--log-file", str(log),
]
raise SystemExit(subprocess.call(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"}))
