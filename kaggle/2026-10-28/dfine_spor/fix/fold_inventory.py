# Инвентарь по фолдам (без скачивания растров): размеры SpOr-растров по заголовку (presigned /vsicurl/), окна инференса
# tile 640/overlap 160 при 0,5 м в UTM, число GT-рамок, участков с разметкой, семей. Ключи не печатаются.
import sys, os, json, math, collections, concurrent.futures as cf
H = "/workspace/koz3_shared/bot-9/kaggle/dfine_spor"; SCRIPT = f"{H}/full/koz3-dfine-spor-full.py"
src = open(SCRIPT, encoding="utf-8").read(); cut = src[:src.index("class Scene:")]
sys.argv = [SCRIPT, "--allow-cpu"]; g = {"__name__": "__main__", "__file__": SCRIPT}
exec(compile(cut, SCRIPT, "exec"), g)
import rasterio
from rasterio.warp import calculate_default_transform
INV, FOLD, FAM, CLI, BUCKET, CFG = g["INV"], g["FOLD"], g["FAM"], g["CLI"], g["BUCKET"], g["CFG"]
T, S = 640, 480
def one(inv):
    out = {"region": inv["region"], "fold": FOLD[inv["region"]], "fam": FAM[inv["region"]], "marked": bool(inv["marks"]), "wins": 0, "mpx": 0, "err": None}
    try:
        out["gt"] = 0
        for key, size in inv["tifs"]:
            url = CLI.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=3600)
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.TIF"):
                with rasterio.open("/vsicurl/" + url) as ds:
                    crs = g["target_crs"](inv, ds)
                    tr, w, h = calculate_default_transform(ds.crs, crs, ds.width, ds.height, *ds.bounds, resolution=0.5)
            xs = len(range(0, max(1, w - 160), S)) or 1; ys = len(range(0, max(1, h - 160), S)) or 1
            out["wins"] += xs * ys; out["mpx"] += w * h / 1e6
            if inv["marks"]: out["gt"] += len(g["load_gt"](inv, crs))
    except Exception as e: out["err"] = repr(e)[:150]
    return out
with cf.ThreadPoolExecutor(8) as ex: R = list(ex.map(one, INV))
json.dump(R, open(sys.argv[0].replace(".py", ".json") if False else os.environ["INV_OUT"], "w"), ensure_ascii=False, indent=0)
agg = collections.defaultdict(lambda: collections.Counter()); fams = collections.defaultdict(set)
for r in R:
    a = agg[r["fold"]]; a["regions"] += 1; a["marked"] += r["marked"]; a["wins"] += r["wins"]; a["gt"] += r.get("gt", 0) or 0; a["err"] += r["err"] is not None
    fams[r["fold"]].add(r["fam"])
for k in sorted(agg): print("FOLD", k, dict(agg[k]), "families", len(fams[k]), flush=True)
print("ERRORS", [(r["region"], r["err"]) for r in R if r["err"]][:5])
