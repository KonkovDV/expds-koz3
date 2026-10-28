# ==== Ядро 1 (CPU, интернет): S3 → обучающие патчи по рецепту MataNerdy + манифест растров для инференса ====
import sys, csv, shutil, concurrent.futures as cf
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
OUT = Path(os.environ.get("MN_OUT", "/kaggle/working"))


def prep_region(cli, region, keys):
    import rasterio
    rast, marks, utm = region_files(keys)
    if utm is None: return [], None, {"region": region, "skip": "нет UTM.json"}
    crs_utm = utm_crs(fetch(cli, *utm))
    mpaths = [(fetch(cli, k, s), mod, cls) for k, s, mod, cls in marks]
    rows, X, M = [], [], []
    info = {"region": region, "utm": crs_utm, "rasters": {}}
    for mod, lst in rast.items():
        mods_marks = [(p, m, c) for p, m, c in mpaths if m == mod]
        info["rasters"][mod] = [{"key": k, "size": s} for k, s in lst]
        if not lst or not mods_marks: continue
        opened = []
        for k, s in lst:
            ds = rasterio.open(fetch(cli, k, s)); opened.append((k, ds, raster_crs(ds, crs_utm)))
        for k, ds, crs in opened:
            info["rasters"][mod][[r["key"] for r in info["rasters"][mod]].index(k)]["crs"] = crs
        by_crs = {}
        for k, ds, crs in opened:
            if crs not in by_crs: by_crs[crs] = load_marks(mods_marks, crs).get(mod, [])
        done = set()
        for k, ds, crs in opened:  # первый растр, с которым пересекается объект (как у конкурента)
            from shapely.geometry import box
            bb = box(*ds.bounds)
            objs = by_crs[crs]
            for i, (g, v) in enumerate(objs):
                if i in done or not g.intersects(bb): continue
                done.add(i)
                r = adaptive_patch(ds, g, objs)
                if r is None: continue
                patch, mask, crop = r
                im, mk = resize_pair(patch, mask)
                X.append(im.astype(np.float16)); M.append(mk)
                rows.append({"region": region, "modality": mod, "markup_type": MOD2MARKUP[mod], "raster": k,
                             "class_name": CLASSES[v], "crop_size": crop})
        for _, ds, _ in opened: ds.close()
    if os.environ.get("MN_KEEP", "0") != "1":
        shutil.rmtree(CACHE / region, ignore_errors=True)
    return rows, (np.stack(X), np.stack(M)) if X else None, info


def main():
    t0 = time.time(); cli = s3_client()
    regions = sorted(FOLDS["family_of"])
    if os.environ.get("MN_REGIONS"): regions = [r for r in regions if r in os.environ["MN_REGIONS"].split(",")]
    fold_of = {r: f["fold"] for f in FOLDS["folds"] for r in f["test_regions"]}
    allrows, XS, MS, manifest = [], [], [], {}
    with cf.ThreadPoolExecutor(3) as ex:  # 3 участка параллельно: скачивание перекрывается с нарезкой
        futs = {ex.submit(lambda r: prep_region(cli, r, list_region(cli, r)), r): r for r in regions}
        for fu in cf.as_completed(futs):
            r = futs[fu]
            try:
                rows, xm, info = fu.result()
            except Exception as e:
                log.error("участок %s: %s", r, e); manifest[r] = {"error": str(e)[:300]}; continue
            for row in rows: row.update(family=FOLDS["family_of"][r], fold=fold_of[r])
            allrows += rows; manifest[r] = info
            if xm is not None: XS.append(xm[0]); MS.append(xm[1])
            log.info("%s: %d патчей, всего %d, %.0f с", r, len(rows), len(allrows), time.time() - t0)
    X = np.concatenate(XS) if XS else np.zeros((0, 256, 256), np.float16)
    M = np.concatenate(MS) if MS else np.zeros((0, 256, 256), np.uint8)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "X.npy", X); np.save(OUT / "M.npy", M)
    with open(OUT / "meta.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["idx", "region", "family", "fold", "modality", "markup_type", "raster", "class_name", "crop_size"])
        w.writeheader()
        for i, row in enumerate(allrows): w.writerow({"idx": i, **row})
    json.dump({"folds": FOLDS, "regions": manifest}, open(OUT / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False)
    import collections
    summ = {"patches": len(allrows), "sec": round(time.time() - t0),
            "by_mod_class": collections.Counter(f'{r["modality"]}/{r["class_name"]}' for r in allrows),
            "by_fold": collections.Counter(r["fold"] for r in allrows), "errors": [r for r, v in manifest.items() if "error" in v]}
    json.dump(summ, open(OUT / "prep_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
