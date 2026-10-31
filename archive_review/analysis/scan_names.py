"""Файлы разметки, у которых naming.modality_of не находит модальность."""
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
from expds.naming import modality_of, class_of_annotation

root = Path(r"F:\expds3\train-expds-3")
miss_m, miss_c = [], []
n = 0
for p in root.rglob("*.geojson"):
    if "размет" not in str(p.parent):
        continue
    n += 1
    if modality_of(p) is None:
        miss_m.append(p.name)
    if class_of_annotation(p) is None and "structure" not in p.name.lower() and "struct" not in p.name.lower():
        miss_c.append(p.name)
print("files", n)
print("no modality", len(miss_m))
for name in miss_m[:40]:
    print(" M", name)
print("no class", len(miss_c))
for name in miss_c[:40]:
    print(" C", name)
