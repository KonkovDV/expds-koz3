# ── Or-подготовка (29.10): огромные ортофото Or (до 26,7 ГБ, полосы без сжатия, 1–2 см/пикс) → компактный GeoTIFF
# в исходной CRS с шагом ≈PREP_RES м (усреднение блоками строк, маска альфа/nodata → 0, первые ≤3 канала, dtype как в
# источнике). Дальше Scene, как для любого растра, приводит его к UTM 0,5 м (average). Источник удаляется сразу после
# подготовки (диск Kaggle). Используется и CPU-ядром koz3-dfine-or-prep, и GPU-ядром Or (как запасной путь).
PREP_RES = float(os.environ.get("DFINE_PREP_RES", 0.25)); PREP_MIN_BYTES = int(float(os.environ.get("DFINE_PREP_MIN_BYTES", 1.5e9)))
PREP_LOCK = threading.Lock()  # не больше одного огромного скачивания одновременно
def prep_name(key, size): return f"{Path(key).stem}__{size}__r{PREP_RES:g}.tif"
def prep_tif(src_path, dst_path, log=print):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from rasterio.warp import calculate_default_transform
    t = time.time()
    with rasterio.open(src_path) as s:
        try: nat = abs(calculate_default_transform(s.crs, "EPSG:3857", s.width, s.height, *s.bounds)[0].a)
        except Exception: nat = abs(s.transform.a)
        # 3857 растягивает метры на 1/cos(lat): поправка через UTM-подобную оценку ширины в метрах
        try:
            from pyproj import Geod, Transformer
            lon0, lat0 = Transformer.from_crs(s.crs, "EPSG:4326", always_xy=True).transform(*s.xy(s.height // 2, 0))
            lon1, lat1 = Transformer.from_crs(s.crs, "EPSG:4326", always_xy=True).transform(*s.xy(s.height // 2, s.width - 1))
            nat = Geod(ellps="WGS84").inv(lon0, lat0, lon1, lat1)[2] / max(1, s.width - 1)
        except Exception: pass
        f = max(1, int(PREP_RES / max(nat, 1e-6)))
        W, H = math.ceil(s.width / f), math.ceil(s.height / f); nb = min(3, s.count); R = max(1, int(64e6 // (s.width * f * 2 * s.count)))
        prof = dict(driver="GTiff", width=W, height=H, count=nb, dtype=s.dtypes[0], crs=s.crs, nodata=0, tiled=True,
                    blockxsize=256, blockysize=256, compress="deflate", predictor=2, BIGTIFF="IF_SAFER",
                    transform=s.transform * s.transform.scale(s.width / W, s.height / H))
        # 4-й канал uint8 = альфа, если так помечен или содержит только 0/255 (GDAL не всегда видит альфу без ExtraSamples)
        alpha = False
        if s.count >= 4 and s.dtypes[3] == "uint8":
            from rasterio.enums import ColorInterp
            v = np.unique(s.read(4, out_shape=(min(512, s.height), min(512, s.width)), resampling=Resampling.nearest))
            alpha = s.colorinterp[3] == ColorInterp.alpha or set(v.tolist()) <= {0, 255}
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True); tmp = str(dst_path) + ".part"
        with rasterio.open(tmp, "w", **prof) as d:
            for oy in range(0, H, R):
                r = min(R, H - oy); sy = oy * f; sh = min(s.height - sy, r * f)
                win = Window(0, sy, s.width, sh)
                a = s.read(list(range(1, nb + 1)), window=win, out_shape=(nb, r, W), resampling=Resampling.average)
                m = s.dataset_mask(window=win, out_shape=(r, W), resampling=Resampling.nearest)
                if alpha: m = np.minimum(m, s.read(4, window=win, out_shape=(r, W), resampling=Resampling.nearest))
                a[:, m == 0] = 0
                d.write(a, window=Window(0, oy, W, r))
        os.replace(tmp, dst_path)
    log(f"[prep] {Path(src_path).name}: нативно {nat:.4f} м/пикс, шаг ×{f} → {W}x{H}x{nb}, альфа {alpha}, "
        f"{Path(dst_path).stat().st_size / 1e6:.1f} МБ, {time.time() - t:.0f} с")
    return {"native_m": round(nat, 5), "factor": f, "alpha_band4": bool(alpha), "W": W, "H": H, "bands": nb, "sec": round(time.time() - t)}
