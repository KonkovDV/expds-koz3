#!/usr/bin/env python3
"""Раскладывает веса s5 (02_centers_fixed_out.tar.gz → out/) по work/fold{k}/li.pt + done.json,
чтобы scripts/nn_oof.py --resume не обучал заново, а только предсказал свои семьи.
    python stage_s5_for_nn_oof.py --out-dir <распакованный>/out --run li_r18_unet_centers_fix --work F:/expds3/work/oof_c18
Только стандартная библиотека; ничего не удаляет; существующие li.pt не перезаписывает без --force."""
import argparse, json, shutil
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--work", required=True); ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
src, work = Path(a.out_dir), Path(a.work)
summ = {Path(r["result"]["out"]).name: r["result"] for r in json.loads((src / "train_summary.json").read_text(encoding="utf-8"))}
for k in range(a.folds):
    pt = src / f"{a.run}_fold{k}.pt"
    if not pt.exists():
        raise SystemExit(f"нет {pt}")
    fc = work / f"fold{k}"; fc.mkdir(parents=True, exist_ok=True)
    dst = fc / "li.pt"
    if dst.exists() and not a.force:
        raise SystemExit(f"{dst} уже есть (--force для замены)")
    shutil.copy2(pt, dst)
    (fc / "done.json").write_text(json.dumps(summ.get(pt.name, {"out": str(pt)}), ensure_ascii=False), encoding="utf-8")
    print(f"fold{k}: {pt.name} → {dst}")
