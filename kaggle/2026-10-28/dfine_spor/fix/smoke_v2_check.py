# Разбор smoke v2: выполняем собранный smoke-скрипт до оркестрации (те же 8 участков), проверяем банк чипов и таргеты,
# затем (опц.) тренируем фолд f2 тренером бота 5 N итераций на CPU и меряем loss на фиксированном батче до/после.
import sys, os, json, collections, math, time
import numpy as np
H = "/workspace/koz3_shared/bot-9/kaggle/dfine_spor"; SCRIPT = f"{H}/smoke/koz3-dfine-spor-smoke.py"
NTRAIN = int(os.environ.get("CHK_TRAIN_ITERS", "0")); FOLDK = int(os.environ.get("CHK_FOLD", "2"))
src = open(SCRIPT, encoding="utf-8").read(); cut = src[:src.index("REPORT, lock =")]
sys.argv = [SCRIPT, "--allow-cpu"]; g = {"__name__": "__main__", "__file__": SCRIPT}
exec(compile(cut, SCRIPT, "exec"), g)
RECS, INV, FOLD, CFG, CLASSES = g["RECS"], g["INV"], g["FOLD"], g["CFG"], g["CLASSES"]
inv_train = g["inv_train"]; print("CHK folds_run", g["folds_run"], "tile", CFG["tile"], flush=True)
# 1) по участкам: GT-рамок в растре vs объектных чипов; распределение по фолдам
tot = collections.Counter()
for inv in inv_train:
    nb = 0
    for key, size in inv["tifs"]:
        sc = g["Scene"](inv, key, size); gt = g["load_gt"](inv, sc.crs)
        bx = [(sc.to_px(b), c) for b, c in gt]; ins = [b for b, c in bx if b[2] > 0 and b[3] > 0 and b[0] < sc.W and b[1] < sc.H]
        nb += len(ins); W, Hh = sc.W, sc.H; sc.close()
    r = [x for x in RECS if x["region"] == inv["region"]]
    no = sum(x["cls"] >= 0 for x in r); nbg = len(r) - no
    print(f"CHK region {inv['region']} fold {FOLD[inv['region']]} tifs {len(inv['tifs'])} raster {W}x{Hh}px GT_in_raster {nb} obj_chips {no} bg_chips {nbg} "
          f"classes {dict(collections.Counter(CLASSES[x['cls']] for x in r if x['cls'] >= 0))}", flush=True)
    tot[FOLD[inv["region"]]] += len(r)
print("CHK chips per fold", dict(tot), "| train chips for fold k = total - fold k:", {k: len(RECS) - tot[k] for k in g["folds_run"]}, flush=True)
# 2) таргеты из ChipSet (обе ветки аугментаций): формат, диапазоны, пустые
for aug in ("ours", "bot5"):
    CFG["aug"] = aug; ds = iter(g["ChipSet"]([r for r in RECS if r["fold"] != FOLDK], 7)); st = collections.Counter(); bad = []
    for i in range(400):
        x, y = next(ds); b, l = y["boxes"].numpy(), y["class_labels"].numpy()
        st["n"] += 1; st["empty"] += len(b) == 0; st["boxes"] += len(b)
        if x.shape != (3, CFG["tile"], CFG["tile"]) or not (0 <= float(x.min()) and float(x.max()) <= 1): bad.append(("img", tuple(x.shape)))
        if len(b):
            if b.min() < 0 or b.max() > 1: bad.append(("range", float(b.min()), float(b.max())))
            if (b[:, 2] <= 0).any() or (b[:, 3] <= 0).any(): st["zero_wh"] += int(((b[:, 2] <= 0) | (b[:, 3] <= 0)).sum())
            if l.min() < 0 or l.max() > 4: bad.append(("label", int(l.min()), int(l.max())))
            st["small<4px"] += int(((b[:, 2] * CFG["tile"] < 4) | (b[:, 3] * CFG["tile"] < 4)).sum())
            # cxcywh: рамка целиком внутри [0,1]
            if ((b[:, 0] - b[:, 2] / 2) < -1e-4).any() or ((b[:, 0] + b[:, 2] / 2) > 1 + 1e-4).any(): st["outside"] += 1
    print(f"CHK targets fold!={FOLDK} aug={aug}:", dict(st), "bad:", bad[:5], flush=True)
CFG["aug"] = "ours"
if not NTRAIN: sys.exit(0)
# 3) обучение f{FOLDK} тренером бота 5 + loss на фиксированном батче (те же 8 картинок) до/после
import torch
torch.manual_seed(0); tr = [r for r in RECS if r["fold"] != FOLDK]
fx = iter(g["ChipSet"](tr, 999)); FB = [next(fx) for _ in range(8)]
def fixed_loss(m):
    m.train(); out = []
    with torch.no_grad():
        for i in range(0, 8, 2):
            o = m(pixel_values=torch.stack([a for a, _ in FB[i:i + 2]]), labels=[b for _, b in FB[i:i + 2]]); out.append(float(o.loss))
    return round(float(np.mean(out)), 2)
model, info = g["make_model"](seed=100 + FOLDK); l0 = fixed_loss(model); print("CHK fixed-batch loss before", l0, flush=True)
cfg = dict(CFG, iters=NTRAIN, mode="smoke", budget_s=None, stop_ts=None, tag=f"[cpu f{FOLDK}]")
model = g["BOT5"].train(model, g["batches"](tr, 1000 + FOLDK, torch.device("cpu")), cfg, torch.device("cpu"), g["log"])
l1 = fixed_loss(model); rep = model.train_report; h = [x["loss"] for x in rep["hist"]]
q = max(1, len(h) // 4)
print("CHK fixed-batch loss after", l1, "| train loss mean first/last quarter", round(np.mean(h[:q]), 1), round(np.mean(h[-q:]), 1),
      "| finite", all(map(math.isfinite, h)), "| s_it", round(rep["s_it_median"], 2), flush=True)
json.dump({"before": l0, "after": l1, "hist": rep["hist"]}, open(os.environ["DFINE_OUT"] + "/train_trend.json", "w"))
