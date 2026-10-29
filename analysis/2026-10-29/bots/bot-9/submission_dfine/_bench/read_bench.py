# Время чтения одного окна 640 px на 0,5 м (WarpedVRT, как dfine_infer.SceneGrid/_ThreadReader), 1 поток GDAL; box — ориентир.
import sys, time, glob, json
sys.path.insert(0, "/workspace/koz3_shared/bot-9/submission_dfine")
import dfine_infer as D
out = {}
for reg in ["183_Сарское", "158_Курганы_29"]:
    d = f"/workspace/data/train-expds-3/{reg}"; tif = sorted(glob.glob(f"{d}/04_*SpOr*/*.tif"))[0]
    g = D.SceneGrid(tif, f"{d}/UTM.json"); rd = D._ThreadReader(g); w = D.windows(g); x, y = w[len(w) // 2]
    t = time.time(); rd.read(x, y, 640); t1 = time.time() - t
    t = time.time(); rd.read(x, y, 640); t2 = time.time() - t
    import rasterio; res = rasterio.open(tif).res
    out[reg] = dict(tif=tif.split("/")[-1], native_res=res, windows=len(w), first_read_s=round(t1, 2), second_read_s=round(t2, 2))
print(json.dumps(out, ensure_ascii=False, indent=1))
