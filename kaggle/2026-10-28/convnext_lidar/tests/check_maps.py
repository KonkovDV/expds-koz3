# Проверка OOF-карт для реранкера bot-2: состав npz, meta, повтор декодирования кандидатов centerhead.decode по картам.
# python tests/check_maps.py OUT_DIR
import sys, json, glob, numpy as np, collections
sys.path.insert(0, "src/vendor")
from centerhead import cfg_with
from centerhead.decode import decode
import cnx_head as HD
out = sys.argv[1]; rf = json.load(open(f"{out}/region_fold.json", encoding="utf-8"))
print("region_fold.json:", len(rf["region_fold"]), "участков, фолды", dict(sorted(collections.Counter(rf["region_fold"].values()).items())), "jobs", rf["jobs"])
for p in sorted(glob.glob(f"{out}/maps/*.npz")):
    z = np.load(p); m = json.loads(str(z["meta"]))
    print(p.split("/")[-1], {k: (z[k].dtype.name, list(z[k].shape)) for k in z.files if k != "meta"})
    print("  meta:", {k: m[k] for k in ("region", "fold", "family", "stride", "left", "top", "step", "step_head", "crs", "H", "W", "tags", "enc")})
    assert rf["region_fold"][m["region"]] == m["fold"], "фолд в meta ≠ region_fold.json"
    if m["head"] == "centerhead":
        hm = z["hm"].astype(np.float32); cfg = cfg_with(stride=m["stride"], circle_r_m=0.5)
        t = decode(hm, m["left"], m["top"], m["step_head"], wh=z["wh"].astype(np.float32), off=z["off"].astype(np.float32), valid=(z["valid_h"] > 0) & ~(HD.block_mean(z["mask"][0] > 127, m["stride"]) > 0.5), cfg=cfg)   # как ядро: минус клетки городищ
        oof = glob.glob(f"{out}/oof_cnx_{m['tag']}.geojson")[0]
        n_oof = sum(f["properties"]["class_name"].startswith("kurgany") and f["properties"]["region_name"] == m["region"]
                    for f in json.load(open(oof, encoding="utf-8"))["features"])
        print(f"  decode по npz (hm f32, wh/off f16, gor из mask u8): {len(t['X'])} кандидатов; в OOF курганов: {n_oof}")
