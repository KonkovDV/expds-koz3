"""bot-7: окна SpOr участка 103, где f0 даёт детекции (score >= 0.3) — опорные окна пробы весов check_submission."""
import json, sys, time
from pathlib import Path
import torch
torch.set_num_threads(2)
sys.path.insert(0, "/workspace/koz3_shared/bot-7/repo")
sys.path.insert(0, "/workspace/koz3_shared/bot-7/work6/vendor_test/lib")
from expds import sp_dfine as D
tif = Path(sys.argv[2])
st = dict(D.DEFAULTS, weights=[sys.argv[1]], batch=1, flip_tta=False)
dev = torch.device("cpu")
models = D.load_models(st, Path("/"), dev)
sc = D.Scene(tif, D.utm_for(tif, None), st["res_m"])
wins = D.windows(sc.W, sc.H, st["tile"], st["overlap"])
print("scene", sc.W, sc.H, "wins", len(wins), flush=True)
found, t0 = [], time.time()
for i, w in enumerate(wins):
    im = sc.read(w[0], w[1], st["tile"])
    if (im.max(2) == 0).mean() > st["skip_empty"]:
        continue
    with torch.no_grad():
        o = models[0](pixel_values=torch.from_numpy(im.transpose(2, 0, 1)[None].astype("float32") / 255))
    p = o.logits.sigmoid().flatten()
    n3 = int((p >= 0.3).sum()); mx = float(p.max())
    print(i, w, n3, round(mx, 3), round(time.time() - t0, 1), flush=True)
    if n3 >= 3:
        found.append({"i": i, "xy": list(w), "n_ge_0.3": n3, "max": round(mx, 4)})
    if len(found) >= 4 or i > 400:
        break
json.dump({"tif": tif.name, "W": sc.W, "H": sc.H, "found": found}, open("windows_103_f0.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(found))
