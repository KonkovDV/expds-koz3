# bot 9: подготовка Or-растров для CPU-дрирана из ЛОКАЛЬНЫХ копий (/workspace/data/train-expds-3) кодом src/or_prep_lib.py;
# имя файла — prep_name(S3-ключ, S3-размер), чтобы raster_path() ядра нашёл его в DFINE_PREP_DIR. Локальные копии ≠ S3-оригиналы по байтам.
import os, sys, math, time, threading, numpy as np
from pathlib import Path
G = dict(os=os, math=math, time=time, threading=threading, np=np, Path=Path)
exec(open("/workspace/koz3_shared/bot-9/kaggle/dfine_spor/src/or_prep_lib.py", encoding="utf-8").read(), G)
D = Path("/workspace/koz3_shared/bot-9/or_fix/dryrun_prep/or_prep")
for reg, nm, size in [("056_Фалхан", "Фалхан", 3675203188), ("008_Герите", "Герите", 12355613849)]:
    key = f"{reg}/05_{nm}_Or_ортофото/01_{nm}_Or_ортофото.tif"
    src = Path("/workspace/data/train-expds-3") / key; dst = D / G["prep_name"](key, size)
    if dst.exists(): print("есть", dst.name, flush=True); continue
    print(reg, G["prep_tif"](src, dst), flush=True)
