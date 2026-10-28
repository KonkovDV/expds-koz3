"""Собирает notebooks/01_kurgan_subclass.ipynb и папку пуша kaggle/01_kurgan_subclass/ из ячеек k01/c*.py и configs/01_kurgan_subclass.json."""
import json, shutil
from pathlib import Path
B = Path(__file__).resolve().parents[2]
cfg = json.loads((B / "configs/01_kurgan_subclass.json").read_text(encoding="utf-8"))
cells_src = [p.read_text(encoding="utf-8") for p in sorted((B / "notebooks/k01").glob("c[0-9]_*.py"))]
cells_src[0] = cells_src[0].replace("__CFG__", json.dumps(cfg, ensure_ascii=False, indent=1))
md = ("# KOZ3 · 01 · подкласс курганов (повреждённый / целый), bot-9\n"
      "Вход: `/kaggle/input/koz3-kurgan-subclass-crops/kurgan_subclass_crops.npz` (вывод `scripts/bot9_kurgan_subclass.py crops`).\n"
      "Честная проверка: 10 фолдов по семьям участков (`family_folds`/`site_family`, как `scoreboard --honest 10`).\n"
      "`require_gpu`: без 2×T4 ядро падает в первой ячейке. Выход: `kurgan_subclass_oof.csv`, `kurgan_subclass_auc_by_family.csv`, "
      "`kurgan_subclass_report.json`, `weights/*_all.pt`.")
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": md.splitlines(True)}] + [code(s) for s in cells_src],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = B / "notebooks/01_kurgan_subclass.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
(B / "notebooks/01_kurgan_subclass.py").write_text("\n\n".join(cells_src), encoding="utf-8")
push = B / "kaggle/01_kurgan_subclass"; push.mkdir(parents=True, exist_ok=True)
shutil.copy(out, push / "koz3-kurgan-subclass.ipynb")
meta = {"id": "dmitrijkv/koz3-kurgan-subclass", "title": "koz3-kurgan-subclass", "code_file": "koz3-kurgan-subclass.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["dmitrijkv/koz3-kurgan-subclass-crops", "dmitrijkv/koz3-li-tin-59-t128-bundle"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"}
(push / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
print(out, push)
