"""Какие файлы 102 ещё не скачаны: имя и размер, без ключей."""
import importlib.util
from pathlib import Path

src = Path(r"C:\Sloi\files\materials\s3_script\s3_sync_train-expds-3.py")
spec = importlib.util.spec_from_file_location("s3sync", src)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import argparse
ap = argparse.ArgumentParser()
# повторяем только то, что нужно list_remote / build_plan
args = argparse.Namespace(
    folder="102_Гнёздово_Заольшанская",
    local_dir=r"F:\expds3\train-expds-3",
    excludes=["Старая", "Архив"],
    endpoint=mod.DEFAULT_ENDPOINT,
    bucket=mod.DEFAULT_BUCKET,
    access_key=mod.DEFAULT_ACCESS_KEY,
    secret_key=mod.DEFAULT_SECRET_KEY,
    tls_verify=False,
    threads=2,
    mode="update",
    force=False,
    skip_changed=False,
)
client = mod.make_client(args)
objects = mod.list_remote(client, args)
local = Path(args.local_dir)
todo = []
for o in objects:
    dest = local / o.key
    if dest.is_file() and dest.stat().st_size == o.size:
        continue
    todo.append(o)
print(f"missing {len(todo)} of {len(objects)}")
for o in sorted(todo, key=lambda x: -x.size):
    print(f"{o.size:12d}  {o.key}")
