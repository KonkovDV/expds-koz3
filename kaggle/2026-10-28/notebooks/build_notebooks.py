"""Собирает три .ipynb из bot9_kaggle_common.py и configs/*.json. Ничего не обучает."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = HERE.parent / "configs"
COMMON = (HERE / "bot9_kaggle_common.py").read_text(encoding="utf-8")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


def nb(cells):
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": False}}}


def embed(name):
    return json.dumps(json.loads((CFG / name).read_text(encoding="utf-8")), ensure_ascii=False, indent=1)


HEAD = """
# bot-9 · {title}
Подготовлено bot-9 28.10.2026, **не запускалось**. Accelerator: **GPU T4 ×2**, интернет не нужен.
Одновременно на аккаунте — только одно GPU-ядро (второе молча получает CPU,
scripts/kaggle_dual_t4_orchestrator.py:637–655). Ячейка 2 падает, если GPU < 2.
Входы: датасет-бандл с `expds_code.tar.gz` **из ветки bot9-kaggle-queue** (ячейка 3 проверяет патч), `hf_hub_cache`, `li_tin_59_t128` (+ `index_fold0..4.csv`).
"""
WRITE_COMMON = "%%writefile /kaggle/working/bot9_kaggle_common.py\n" + COMMON
GUARD = """
import sys, json, os
sys.path.insert(0, "/kaggle/working")
import bot9_kaggle_common as C
CFG = json.loads(r'''{cfg}''')
GPUS = C.gpu_guard(CFG["min_gpus"])   # RuntimeError без CUDA / при < 2 GPU / если тензор не на cuda:i
"""
SETUP = """
from pathlib import Path
BUNDLE = C.find_bundle()
WORK = C.setup_code(BUNDLE)          # RuntimeError, если expds/train.py без патча bot-9
FOLDS = json.loads(r'''{folds}''')
"""
FOLD_CACHES = """
CACHE = C.setup_fold_caches(BUNDLE, FOLDS)   # проверка leave-family-out по index_fold*.csv
Path("/kaggle/working/out").mkdir(exist_ok=True)
"""
SMOKE = """
# 1 эпоха, 512 тайлов, ~1 мин: ловит ошибки до траты квоты. Должен напечатать "device": "cuda:0".
r = C.run_queue({0: [CFG["smoke"]]}, Path("/kaggle/working/logs_smoke"))
C.check_logs_gpu(Path("/kaggle/working/logs_smoke"))
"""
QUEUE = """
cmds = {int(g): v for g, v in CFG["gpu_cmds"].items()}
for g, v in cmds.items():
    assert all("--require-gpu" in c and f"--device cuda:{g}" in c for c in v)
res = C.run_queue(cmds, Path("/kaggle/working/logs"))
C.check_logs_gpu(Path("/kaggle/working/logs"))
print("GPU-секунд по очередям:", {g: sum(r["elapsed_sec"] for r in res if r["gpu"] == g) for g in cmds})
"""
SUMMARY = """
# Итог для ПК: последняя JSON-строка каждого лога (best_*, best_peak_thr, device). OOF-оценка — на ПК, scoreboard --honest 10.
import glob, tarfile
rows = []
for lf in sorted(glob.glob("/kaggle/working/logs/gpu*_step*.log")):
    last = [l for l in open(lf, encoding="utf-8", errors="replace") if l.startswith("{")]
    rows.append({"log": lf, "result": json.loads(last[-1]) if last else None})
json.dump(rows, open("/kaggle/working/out/train_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with tarfile.open("/kaggle/working/{name}_out.tar.gz", "w:gz") as t:
    t.add("/kaggle/working/out", arcname="out"); t.add("/kaggle/working/logs", arcname="logs")
print(len(rows), "логов; архив /kaggle/working/{name}_out.tar.gz")
"""

SUB_TRAIN = r'''
import glob, numpy as np, torch, threading, time
sys.path.insert(0, str(WORK / "scripts"))
import crop_verify as cv
from expds.metric import family_folds
from expds.naming import site_family

npz = sorted(glob.glob(CFG["input_npz_glob"], recursive=True))
assert npz, "нет kurgan_subclass_crops.npz (scripts/bot9_kurgan_subclass.py crops на ПК)"
D = np.load(npz[0], allow_pickle=False)
X, y, region = D["X"], D["y"].astype(np.int64), D["region"].astype(str)
print("окон", len(y), "с меткой", int((y >= 0).sum()), "доля повреждённых", float((y[y >= 0] == 1).mean()))
split = family_folds(sorted(set(region)), site_family, CFG["lfo_folds"])
fold_of = {r: k for k, regs in enumerate(split) for r in regs}
fold = np.array([fold_of[r] for r in region])
for k in range(len(split)):  # честность: ни одна семья не в двух фолдах
    fam_te = {site_family(r) for r in split[k]}
    fam_tr = {site_family(r) for kk, regs in enumerate(split) if kk != k for r in regs}
    assert not (fam_te & fam_tr), f"семья в обучении и проверке фолда {k}: {fam_te & fam_tr}"
w_cls = CFG["loss_weights"]; W1, W0 = float(w_cls["kurgany_povrezhdennye"]), float(w_cls["kurgany_tselye"])

def fit_fold(dev, Xtr, ytr, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = cv.make_model(CFG["encoder"], CFG["pretrained"]).to(dev)
    assert all(p.device == dev for p in m.parameters()), "модель не на GPU"
    opt = torch.optim.AdamW(m.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG["lr"], total_steps=CFG["epochs"] * (len(ytr) // CFG["batch"] + 1))
    scaler = torch.amp.GradScaler("cuda")
    wv = torch.tensor([W0, W1], device=dev) / (W0 + W1) * 2
    for ep in range(CFG["epochs"]):
        m.train(); order = rng.permutation(len(ytr))
        for s in range(0, len(order), CFG["batch"]):
            b = order[s:s + CFG["batch"]]
            if len(b) < 2: continue
            x = cv._tensor(X[b] if Xtr is None else Xtr[b], dev)
            assert x.is_cuda
            x = torch.rot90(x, int(rng.integers(4)), dims=(2, 3))
            if rng.random() < 0.5: x = x.flip(-1)
            t = torch.from_numpy(ytr[b].astype(np.float32)).to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                lo = torch.nn.functional.binary_cross_entropy_with_logits(m(x).float(), t, weight=wv[t.long()])
            opt.zero_grad(set_to_none=True); scaler.scale(lo).backward(); scaler.step(opt); scaler.update(); sched.step()
    return m.eval()

P = {}
def run_gpu(g, seeds):
    dev = torch.device(f"cuda:{g}")
    for seed in seeds:
        p = np.full(len(y), np.nan)
        for k in range(len(split)):
            te = fold == k; tr = (~te) & (y >= 0)
            if len(set(y[tr])) < 2: continue
            t0 = time.time()
            m = fit_fold(dev, X[tr], y[tr], seed * 100 + k)
            p[te] = cv.predict(m, X[te])
            print(f"cuda:{g} seed {seed} fold {k}: train {int(tr.sum())}, test {int(te.sum())}, {time.time()-t0:.1f} s", flush=True)
        P[(g, seed)] = p
th = [threading.Thread(target=run_gpu, args=(int(g), s)) for g, s in CFG["seeds_by_gpu"].items()]
[t.start() for t in th]; [t.join() for t in th]
p_mean = np.nanmean(np.stack(list(P.values())), axis=0)
'''
SUB_EVAL = r'''
import csv
lab = (y >= 0) & np.isfinite(p_mean)
pred = (p_mean >= CFG["apply_thr"]).astype(int)
cand = D["kind"].astype(str) == "cand"
cur = np.array([1 if c == "kurgany_povrezhdennye" else 0 for c in D["cand_class"].astype(str)])
m = lab & cand
rep = {"acc_all_labeled": float((pred[lab] == y[lab]).mean()),
       "cand_labeled": int(m.sum()),
       "cand_wrong_subclass_now": int((cur[m] != y[m]).sum()),
       "fixed_by_clf": int(((cur != y) & (pred == y))[m].sum()),
       "broken_by_clf": int(((cur == y) & (pred != y))[m].sum()),
       "by_fold_acc": {int(k): float((pred[lab & (fold == k)] == y[lab & (fold == k)]).mean()) for k in sorted(set(fold[lab]))}}
print(json.dumps(rep, ensure_ascii=False, indent=1))
Path("/kaggle/working/out").mkdir(exist_ok=True)
json.dump(rep, open("/kaggle/working/out/kurgan_subclass_report.json", "w"), indent=1)
with open("/kaggle/working/kurgan_subclass_oof.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["stack_idx", "region", "kind", "y", "p_povr"])
    for i in range(len(y)):
        w.writerow([int(D["stack_idx"][i]), region[i], D["kind"][i], int(y[i]), "" if not np.isfinite(p_mean[i]) else round(float(p_mean[i]), 6)])
print("→ скачать kurgan_subclass_oof.csv; на ПК: scripts/bot9_kurgan_subclass.py apply … и scoreboard --honest 10")
'''


def build():
    out = {}
    sub = embed("01_kurgan_subclass.json")
    out["01_kurgan_subclass.ipynb"] = nb([
        md(HEAD.format(title="классификатор подкласса кургана (повреждённый / целый), 10 фолдов по семьям")
           + "\nДоп. вход: датасет с `kurgan_subclass_crops.npz` (scripts/bot9_kurgan_subclass.py crops на ПК)."),
        code(WRITE_COMMON), code(GUARD.format(cfg=sub)),
        code(SETUP.format(folds=embed("folds_li_tin_59_t128.json"))),
        code(SUB_TRAIN), code(SUB_EVAL)])
    for n, title in (("02_centers_fixed", "исправленная карта центров лидара (5 фолдов + финал, 2 варианта)"),
                     ("03_seed_ensemble", "семена 1–3 рецепта li_r18_hard2_59 (5 фолдов + финал)")):
        out[n + ".ipynb"] = nb([md(HEAD.format(title=title)), code(WRITE_COMMON),
                                code(GUARD.format(cfg=embed(n + ".json"))),
                                code(SETUP.format(folds=embed("folds_li_tin_59_t128.json"))),
                                code(FOLD_CACHES), code(SMOKE), code(QUEUE), code(SUMMARY.replace("{name}", n))])
    for k, v in out.items():
        (HERE / k).write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
        print("written", HERE / k)


if __name__ == "__main__":
    build()
