# ==== mn_lib: общие функции репро MataNerdy (S3, разметка, кропы по рецепту конкурента) ====
# Рецепт: competitors/MataNerdy-Geodata_Archaeology_CV/01_geodata_to_cv/build_multiclass_dataset.py
# (адаптивный кроп вокруг объекта, канал 1 растра, маска по площади убыв.), 03_.../arch_datasets/archaeology_dataset.py
# (bilinear/nearest resize до 256, z-score на патч). Ключи S3 — только Kaggle Secrets, не печатаются и не пишутся.
import os, re, json, time, math, threading, logging
from pathlib import Path
import numpy as np

BUCKET = "train-expds-3"
ENDPOINT = "https://s3.ru-7.storage.selcloud.ru"
SECRET_ID, SECRET_KEY = "KOZ3_S3_ACCESS_KEY_ID", "KOZ3_S3_SECRET_ACCESS_KEY"
CACHE = Path(os.environ.get("MN_CACHE", "/tmp/s3cache"))
CLASSES = ["background", "kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
RU2CLS = {"курганы_целые": "kurgany_tselye", "курганы_поврежденные": "kurgany_povrezhdennye",
          "городища": "gorodishcha", "фортификации": "fortifikatsii", "архитектуры": "arkhitektury"}
MOD2MARKUP = {"Li": "li", "Ae": "ae", "SpOr": "sp_or"}
log = logging.getLogger("mn")


def s3_client():
    """boto3-клиент: ключи из Kaggle Secrets (на box/тестах — из окружения). Значения никуда не выводятся."""
    import boto3
    from botocore.config import Config
    kid = sec = None
    try:
        from kaggle_secrets import UserSecretsClient
        u = UserSecretsClient(); kid, sec = u.get_secret(SECRET_ID), u.get_secret(SECRET_KEY)
    except Exception:
        kid, sec = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not kid or not sec:
        raise SystemExit(f"нет ключей S3: заведите Kaggle Secrets {SECRET_ID} и {SECRET_KEY} и привяжите к ядру")
    ep = os.environ.get("MN_S3_ENDPOINT", ENDPOINT)
    return boto3.client("s3", endpoint_url=ep, aws_access_key_id=kid, aws_secret_access_key=sec,
                        region_name=os.environ.get("AWS_DEFAULT_REGION", "ru-7"),
                        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=16))


def list_region(cli, region):
    out, tok = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": region + "/"}
        if tok: kw["ContinuationToken"] = tok
        r = cli.list_objects_v2(**kw)
        out += [(o["Key"], o["Size"]) for o in r.get("Contents", [])]
        if not r.get("IsTruncated"): return out
        tok = r["NextContinuationToken"]


def fetch(cli, key, size=None, tries=5):
    """Скачать ключ в кэш /tmp (докачка не нужна: проверка размера; повторы с паузой)."""
    from boto3.s3.transfer import TransferConfig
    dst = CACHE / key
    if dst.exists() and (size is None or dst.stat().st_size == size):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    cfg = TransferConfig(multipart_threshold=32 << 20, multipart_chunksize=16 << 20, max_concurrency=8)
    for a in range(tries):
        try:
            cli.download_file(BUCKET, key, str(tmp), Config=cfg)
            tmp.rename(dst); return dst
        except Exception as e:
            log.warning("S3 повтор %d %s: %s", a + 1, key, type(e).__name__); time.sleep(5 * (a + 1))
    raise RuntimeError(f"не скачан {key}")


def region_files(keys):
    """Из листинга участка: растры по модальностям (Li — только *_Li_g.tif, как у конкурента), разметка, UTM.json."""
    rast, marks, utm = {"Li": [], "Ae": [], "SpOr": []}, [], None
    for k, s in keys:
        n = k.rsplit("/", 1)[-1]
        if n == "UTM.json": utm = (k, s)
        elif k.lower().endswith((".tif", ".tiff")):
            if "_Li_" in k and n.lower().endswith("_g.tif"): rast["Li"].append((k, s))
            elif "_Ae_" in k or "_Ае_" in k: rast["Ae"].append((k, s))
            elif "_SpOr_" in k: rast["SpOr"].append((k, s))
        elif n.endswith(".geojson") and "разметка" in k:
            m = re.search(r"_(Li|Ae|Ае|SpOr)_(.+)\.geojson$", n)
            if m and m.group(2) in RU2CLS:
                marks.append((k, s, {"Ае": "Ae"}.get(m.group(1), m.group(1)), RU2CLS[m.group(2)]))
    for m in rast: rast[m].sort()
    if rast["Ae"]: rast["Ae"] = rast["Ae"][:1]  # конкурент: choose_raster_for_modality → первый tif
    return rast, marks, utm


def utm_crs(path):
    d = json.load(open(path, encoding="utf-8"))
    m = re.search(r"EPSG::?(\d+)", d.get("crs", ""))
    return f"EPSG:{m.group(1)}" if m else None


def raster_crs(ds, utm):
    """Li_карты лежат в LOCAL_CS без EPSG — берём CRS из UTM.json (как read_target_crs_from_utm у конкурента)."""
    try:
        if ds.crs is not None and ds.crs.to_epsg():
            return f"EPSG:{ds.crs.to_epsg()}"
    except Exception:
        pass
    return utm


def geojson_crs(d):
    """CRS geojson-файла: поле crs (в бакете встречается EPSG:3857), иначе RFC7946 → CRS84."""
    n = ((d.get("crs") or {}).get("properties") or {}).get("name", "")
    m = re.search(r"EPSG::?(\d+)", n)
    return f"EPSG:{m.group(1)}" if m else "OGC:CRS84"


def load_marks(paths_by_mod, crs):
    """{модальность: [(shapely геометрия в crs, class_id)]}; невалидные чинятся buffer(0)."""
    from shapely.geometry import shape
    from shapely.ops import transform as stransform
    from pyproj import Transformer
    out, trs = {}, {}
    for path, mod, cls in paths_by_mod:
        d = json.load(open(path, encoding="utf-8")); src = geojson_crs(d)
        if src not in trs: trs[src] = Transformer.from_crs(src, crs, always_xy=True).transform
        for f in d.get("features", []):
            g = f.get("geometry")
            if not g or g.get("type") not in ("Polygon", "MultiPolygon"): continue
            geom = stransform(trs[src], shape(g))
            if not geom.is_valid: geom = geom.buffer(0)
            if geom.is_empty: continue
            out.setdefault(mod, []).append((geom, CLASSES.index(cls)))
    return out


def resize_pair(img, mask, size=256):
    import torch, torch.nn.functional as F
    t = torch.from_numpy(img.astype(np.float32))[None, None]
    im = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
    mk = F.interpolate(torch.from_numpy(mask.astype(np.int64))[None, None].float(), size=(size, size), mode="nearest")[0, 0].long().numpy()
    return im, mk.astype(np.uint8)


def adaptive_patch(ds, geom, objs, min_crop=256, max_crop=4096, ctx=2.0):
    """Копия extract_adaptive_patch_and_multi_mask конкурента (build_multiclass_dataset.py:72–230)."""
    import rasterio
    from affine import Affine
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from shapely.geometry import box
    minx, miny, maxx, maxy = geom.bounds
    pw, ph = abs(ds.transform.a), abs(ds.transform.e)
    ow, oh = (maxx - minx) / pw, (maxy - miny) / ph
    crop = min(max_crop, max(min_crop, int(np.ceil(max(ow, oh) * ctx))))
    crop += crop % 2
    c = geom.centroid
    rc, cc = ds.index(c.x, c.y)
    r0, c0 = rc - crop // 2, cc - crop // 2
    rr0, cc0, rr1, cc1 = max(0, r0), max(0, c0), min(ds.height, r0 + crop), min(ds.width, c0 + crop)
    if rr1 <= rr0 or cc1 <= cc0: return None
    raw = ds.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0))
    patch = np.zeros((crop, crop), raw.dtype)
    patch[rr0 - r0:rr0 - r0 + raw.shape[0], cc0 - c0:cc0 - c0 + raw.shape[1]] = raw
    xl, yt = ds.xy(r0, c0, offset="ul")
    tr = Affine(ds.transform.a, ds.transform.b, xl, ds.transform.d, ds.transform.e, yt)
    xr, yb = rasterio.transform.xy(tr, crop, crop, offset="ul")
    win = box(xl, yb, xr, yt)
    items = sorted([(g, v) for g, v in objs if g.intersects(win)], key=lambda t: -t[0].area)
    if not items: return None
    mask = rasterize(items, out_shape=(crop, crop), transform=tr, fill=0, dtype="uint8")
    if not mask.any(): return None
    return patch, mask, crop


def gpu_guard(require=True):
    import torch
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    print("GPU:", names, flush=True)
    if require and n == 0:
        raise SystemExit("--require-gpu: GPU нет (сессия получила CPU — второй одновременный GPU-кернел?). Останов, чтобы не жечь квоту.")
    return n

FOLDS = json.loads(r'''{"source": "make_folds.py (s3/listing.txt, gt_train_v2, bot-10 family_clusters, folds_li_tin_59_t128)", "family_of": {"001_Гнёздово_Нивленское": "Гнёздово", "002_Тешилово": "Тешилово", "003_Лубно": "Лубно", "004_Боровичи_1": "Боровичи", "005_Боровичи_2": "Боровичи", "006_Осечки_1.1км": "Миловидово", "007_Армиево": "Армиево", "009_Хальвасборген": "Хальвасборген", "010_Эсбю_укрепление": "Эсбю", "011_Эрентуна": "Эрентуна", "012_Сож": "Сож", "013_Нора_Виннон": "Нора", "015_Демидовка": "Демидовка", "016_Хога": "Хога", "017_Юшково": "Юшково", "018_Курган_Хога": "Курган", "019_Селяне": "Селяне", "020_Романово": "Романово", "021_Новенькое": "Новенькое", "022_Лихуша": "Лихуша", "023_Бервенец": "Бервенец", "024_Стрекаловка": "Стрекаловка", "025_Вышенское": "Вышенское", "026_Золотаревка": "Золотаревка", "027_Вышегжа_1": "Вышегжа", "028_Вышегжа_2": "Вышегжа", "029_Вышегжа_3": "Вышегжа", "030_Эсбю": "Эсбю", "031_Новотитаровская": "Новотитаровская", "032_Красносельская_0.5км": "Красносельская", "033_Елизаветинское": "Елизаветинское", "034_Усть_река": "Усть", "035_Шумгора": "Шумгора", "036_Тимерево": "Тимерево", "037_Никола_Ленивец": "Никола", "038_Копанское": "Копанское", "039_Примаки_0.7км": "Миловидово", "040_Примаки_1.3км": "Миловидово", "041_Миловидово": "Миловидово", "042_Тору_Айгыр": "Тору", "043_Гочево": "Гочево", "044_Новотитаровская_1.8км": "Новотитаровская", "045_Новотитаровская_2.8км": "Новотитаровская", "046_Тыва_1": "Тыва", "047_Тыва_2": "Тыва", "048_Калмыкия_1": "Калмыкия", "049_Калмыкия_2": "Калмыкия", "050_Воронеж_1": "Воронеж", "051_Воронеж_2": "Воронеж", "052_Воронеж_3": "Воронеж", "053_Воронеж_4": "Воронеж", "054_Воронеж_5": "Воронеж", "055_Курменту": "Курменту", "057_Кабардино_Балкария": "Кабардино", "058_Перынь": "Перынь", "059_Шишкино": "Шишкино", "060_Сельцо": "Сельцо", "061_Граборг": "Граборг", "062_Каменка": "Каменка", "063_Верхнее_Казачье": "Верхнее", "064_Ксизово": "Ксизово", "065_Шниткино": "Шниткино", "066_Московиты": "Московиты", "067_Чудово": "Чудово", "068_Стеклянуха_1": "Стеклянуха", "069_Стеклянуха_2": "Стеклянуха", "070_Архангельское": "Архангельское", "071_Сторожевое": "Сторожевое", "072_Белая_Гора": "Белая", "073_Тоссор": "Тоссор", "074_Солдатское": "Солдатское", "075_Мостище": "Мостище", "076_Маяцкая_крепость": "Маяцкая", "077_Русская_Тростянка": "Русская", "078_Верхняя_Покровка": "Верхняя", "079_Городище": "Городище", "080_Верхний_Карабут": "Верхний", "081_Костомаровское": "Костомаровское", "082_Артпозиции_Иссык_Куль": "Артпозиции", "083_Барскоон": "Барскоон", "084_Булан_Соготту": "Булан", "085_Каджы_Саз": "Иссык", "086_Кан_Дебе": "Кан", "087_Кароол_Дюбе": "Кароол", "088_Кель_Тор_Х": "Кель", "089_Конур_Елен": "Конур", "090_Корумду": "Корумду", "091_Сакские_курганы": "Сакские", "092_Шаты": "Шаты", "094_Иссык_Куль_курганы_и_выкладки": "Иссык", "095_Иссык_Куль_курганы_и_постройки_1": "Иссык", "096_Иссык_Куль_курганы_и_постройки_2": "Иссык", "097_Орнок": "Орнок", "098_Таш_Коро_1": "Таш", "099_Таш_Коро_2": "Таш", "100_Торт_Куль": "Торт", "101_Гнёздово_Днепровская": "Гнёздово", "102_Гнёздово_Заольшанская": "Гнёздово", "103_Гнёздово_Ольшанская": "Гнёздово", "104_Шетнево": "Шетнево", "105_Вщиж": "Вщиж", "106_Верхмарево": "Верхмарево", "107_Сары_Булун": "Сары", "108_Скупая_Полудань": "Скупая", "109_Россошки": "Россошки", "110_Саратов": "Саратов", "111_Каменные_выкладки_1": "Каменные", "112_Каменные_выкладки_2": "Каменные", "113_Каменные_выкладки_3": "Каменные", "114_Каменные_выкладки_4": "Каменные", "115_Каменные_выкладки_5": "Каменные", "116_Каменные_выкладки_6": "Каменные", "117_Каменные_выкладки_7": "Каменные", "118_Каменные_выкладки_8": "Каменные", "119_Каменные_выкладки_9": "Каменные", "120_Каменные_выкладки_10": "Каменные", "121_Каменные_выкладки_11": "Каменные", "122_Каменные_выкладки_12": "Каменные", "123_Каменные_выкладки_13": "Каменные", "124_Постройки_1": "Постройки", "125_Постройки_2": "Постройки", "126_Постройки_3": "Постройки", "127_Постройки_4": "Постройки", "128_Постройки_5": "Постройки", "129_Постройки_6": "Постройки", "130_Курганы_1": "Курганы", "131_Курганы_2": "Курганы", "132_Курганы_3": "Курганы", "133_Курганы_4": "Курганы", "134_Курганы_5": "Курганы", "135_Курганы_6": "Курганы", "136_Курганы_7": "Курганы", "137_Курганы_8": "Курганы", "138_Курганы_9": "Курганы", "139_Курганы_10": "Курганы", "140_Курганы_11": "Курганы", "141_Курганы_12": "Курганы", "142_Курганы_13": "Курганы", "143_Курганы_14": "Курганы", "144_Курганы_15": "Курганы", "145_Курганы_16": "Курганы", "146_Курганы_17": "Курганы", "147_Курганы_18": "Курганы", "148_Курганы_19": "Курганы", "149_Курганы_20": "Курганы", "150_Курганы_21": "Курганы", "151_Курганы_22": "Курганы", "152_Курганы_23": "Курганы", "153_Курганы_24": "Курганы", "154_Курганы_25": "Курганы", "155_Курганы_26": "Курганы", "156_Курганы_27": "Курганы", "157_Курганы_28": "Курганы", "158_Курганы_29": "Курганы", "159_Курганы_30": "Курганы", "160_Курганы_31": "Курганы", "161_Курганы_32": "Курганы", "162_Курганы_33": "Курганы", "163_Курганы_34": "Курганы", "164_Кисловодская_котловина_1": "Кисловодская", "165_Кисловодская_котловина_2": "Кисловодская", "166_Кисловодская_котловина_3": "Кисловодская", "167_Кисловодская_котловина_4": "Кисловодская", "168_Кисловодская_котловина_5": "Кисловодская", "169_Кисловодская_котловина_6": "Кисловодская", "170_Кисловодская_котловина_7": "Кисловодская", "171_Кисловодская_котловина_8": "Кисловодская", "172_Кисловодская_котловина_9": "Кисловодская", "173_Кисловодская_котловина_10": "Кисловодская", "174_Кисловодская_котловина_11": "Кисловодская", "175_Кисловодская_котловина_12": "Кисловодская", "176_Кисловодская_котловина_13": "Кисловодская", "177_Кисловодская_котловина_14": "Кисловодская", "178_Кисловодская_котловина_15": "Кисловодская", "179_Гнёздово_Лесная_1": "Гнёздово", "180_Гнёздово_Лесная_2": "Гнёздово", "181_Гнёздово_Лесная_3": "Гнёздово", "182_Красносельская_1.7км": "Красносельская", "183_Сарское": "Сарское", "184_Шурскол": "Шурскол", "185_Изборск": "Изборск", "186_Семибратнее_1": "Семибратнее", "187_Судьбищи_1": "Судьбищи", "188_Руна": "Руна", "189_Макеево": "Макеево", "190_Муравлево": "Муравлево", "191_Осечки_1.3км": "Миловидово", "192_КЧР": "КЧР", "200_Судьбищи_2": "Судьбищи"}, "folds": [{"fold": 0, "test_families": ["Гнёздово"], "val_family": "Тимерево", "test_regions": ["001_Гнёздово_Нивленское", "101_Гнёздово_Днепровская", "102_Гнёздово_Заольшанская", "103_Гнёздово_Ольшанская", "179_Гнёздово_Лесная_1", "180_Гнёздово_Лесная_2", "181_Гнёздово_Лесная_3"]}, {"fold": 1, "test_families": ["Булан", "Верхняя", "Воронеж", "Кан", "Корумду", "Курменту", "Постройки", "Россошки", "Солдатское", "Сторожевое", "Тимерево", "Тыва", "Шаты", "Шишкино", "Шниткино"], "val_family": "Гнёздово", "test_regions": ["036_Тимерево", "046_Тыва_1", "047_Тыва_2", "050_Воронеж_1", "051_Воронеж_2", "052_Воронеж_3", "053_Воронеж_4", "054_Воронеж_5", "055_Курменту", "059_Шишкино", "065_Шниткино", "071_Сторожевое", "074_Солдатское", "078_Верхняя_Покровка", "084_Булан_Соготту", "086_Кан_Дебе", "090_Корумду", "092_Шаты", "109_Россошки", "124_Постройки_1", "125_Постройки_2", "126_Постройки_3", "127_Постройки_4", "128_Постройки_5", "129_Постройки_6"]}, {"fold": 2, "test_families": ["Артпозиции", "Верхнее", "Вышенское", "Городище", "Демидовка", "Золотаревка", "КЧР", "Кароол", "Костомаровское", "Курганы", "Лихуша", "Московиты", "Никола", "Новотитаровская", "Русская", "Сакские", "Саратов", "Таш", "Торт", "Усть", "Эсбю"], "val_family": "Гнёздово", "test_regions": ["010_Эсбю_укрепление", "015_Демидовка", "022_Лихуша", "025_Вышенское", "026_Золотаревка", "030_Эсбю", "031_Новотитаровская", "034_Усть_река", "037_Никола_Ленивец", "044_Новотитаровская_1.8км", "045_Новотитаровская_2.8км", "063_Верхнее_Казачье", "066_Московиты", "077_Русская_Тростянка", "079_Городище", "081_Костомаровское", "082_Артпозиции_Иссык_Куль", "087_Кароол_Дюбе", "091_Сакские_курганы", "098_Таш_Коро_1", "099_Таш_Коро_2", "100_Торт_Куль", "110_Саратов", "130_Курганы_1", "131_Курганы_2", "132_Курганы_3", "133_Курганы_4", "134_Курганы_5", "135_Курганы_6", "136_Курганы_7", "137_Курганы_8", "138_Курганы_9", "139_Курганы_10", "140_Курганы_11", "141_Курганы_12", "142_Курганы_13", "143_Курганы_14", "144_Курганы_15", "145_Курганы_16", "146_Курганы_17", "147_Курганы_18", "148_Курганы_19", "149_Курганы_20", "150_Курганы_21", "151_Курганы_22", "152_Курганы_23", "153_Курганы_24", "154_Курганы_25", "155_Курганы_26", "156_Курганы_27", "157_Курганы_28", "158_Курганы_29", "159_Курганы_30", "160_Курганы_31", "161_Курганы_32", "162_Курганы_33", "163_Курганы_34", "192_КЧР"]}, {"fold": 3, "test_families": ["Бервенец", "Боровичи", "Верхмарево", "Вщиж", "Гочево", "Елизаветинское", "Иссык", "Кель", "Ксизово", "Макеево", "Миловидово", "Новенькое", "Нора", "Орнок", "Руна", "Сарское", "Сельцо", "Семибратнее", "Скупая", "Тешилово", "Тору", "Чудово", "Шетнево", "Шурскол"], "val_family": "Гнёздово", "test_regions": ["002_Тешилово", "004_Боровичи_1", "005_Боровичи_2", "006_Осечки_1.1км", "013_Нора_Виннон", "021_Новенькое", "023_Бервенец", "033_Елизаветинское", "039_Примаки_0.7км", "040_Примаки_1.3км", "041_Миловидово", "042_Тору_Айгыр", "043_Гочево", "060_Сельцо", "064_Ксизово", "067_Чудово", "085_Каджы_Саз", "088_Кель_Тор_Х", "094_Иссык_Куль_курганы_и_выкладки", "095_Иссык_Куль_курганы_и_постройки_1", "096_Иссык_Куль_курганы_и_постройки_2", "097_Орнок", "104_Шетнево", "105_Вщиж", "106_Верхмарево", "108_Скупая_Полудань", "183_Сарское", "184_Шурскол", "186_Семибратнее_1", "188_Руна", "189_Макеево", "191_Осечки_1.3км"]}, {"fold": 4, "test_families": ["Армиево", "Архангельское", "Барскоон", "Белая", "Верхний", "Вышегжа", "Граборг", "Изборск", "Кабардино", "Калмыкия", "Каменка", "Каменные", "Кисловодская", "Конур", "Копанское", "Красносельская", "Курган", "Лубно", "Маяцкая", "Мостище", "Муравлево", "Перынь", "Романово", "Сары", "Селяне", "Сож", "Стеклянуха", "Стрекаловка", "Судьбищи", "Тоссор", "Хальвасборген", "Хога", "Шумгора", "Эрентуна", "Юшково"], "val_family": "Гнёздово", "test_regions": ["003_Лубно", "007_Армиево", "009_Хальвасборген", "011_Эрентуна", "012_Сож", "016_Хога", "017_Юшково", "018_Курган_Хога", "019_Селяне", "020_Романово", "024_Стрекаловка", "027_Вышегжа_1", "028_Вышегжа_2", "029_Вышегжа_3", "032_Красносельская_0.5км", "035_Шумгора", "038_Копанское", "048_Калмыкия_1", "049_Калмыкия_2", "057_Кабардино_Балкария", "058_Перынь", "061_Граборг", "062_Каменка", "068_Стеклянуха_1", "069_Стеклянуха_2", "070_Архангельское", "072_Белая_Гора", "073_Тоссор", "075_Мостище", "076_Маяцкая_крепость", "080_Верхний_Карабут", "083_Барскоон", "089_Конур_Елен", "107_Сары_Булун", "111_Каменные_выкладки_1", "112_Каменные_выкладки_2", "113_Каменные_выкладки_3", "114_Каменные_выкладки_4", "115_Каменные_выкладки_5", "116_Каменные_выкладки_6", "117_Каменные_выкладки_7", "118_Каменные_выкладки_8", "119_Каменные_выкладки_9", "120_Каменные_выкладки_10", "121_Каменные_выкладки_11", "122_Каменные_выкладки_12", "123_Каменные_выкладки_13", "164_Кисловодская_котловина_1", "165_Кисловодская_котловина_2", "166_Кисловодская_котловина_3", "167_Кисловодская_котловина_4", "168_Кисловодская_котловина_5", "169_Кисловодская_котловина_6", "170_Кисловодская_котловина_7", "171_Кисловодская_котловина_8", "172_Кисловодская_котловина_9", "173_Кисловодская_котловина_10", "174_Кисловодская_котловина_11", "175_Кисловодская_котловина_12", "176_Кисловодская_котловина_13", "177_Кисловодская_котловина_14", "178_Кисловодская_котловина_15", "182_Красносельская_1.7км", "185_Изборск", "187_Судьбищи_1", "190_Муравлево", "200_Судьбищи_2"]}]}''')

# ==== Ядро 2 (GPU T4×2, интернет): обучение DeepLabV3+ R34 по фолдам семей + OOF-полигоны по полным растрам ====
# Гиперпараметры — configs/archaeology_5class_research_split_v1.yaml конкурента (80 эп., patience 12, batch 16, Adam 1e-4,
# plateau 0.5/5, clip 1.0, CE 0.7 (веса [0.2,1,1,1.4,1.8,1.8]) + Dice 0.3, 256 px, 1 канал, без предобучения, без аугментаций);
# постобработка Stage C: порог 0.3, мин. площадь 8 px, opening 3×3.
import sys, csv, argparse, subprocess, traceback
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
OUT = Path(os.environ.get("MN_OUT", "/kaggle/working"))
CFG = dict(epochs=int(os.environ.get("MN_EPOCHS", 80)), patience=12, batch=16, lr=1e-4, clip=1.0, ce_w=0.7, dice_w=0.3,
           class_weights=[0.2, 1.0, 1.0, 1.4, 1.8, 1.8], thr=0.3, min_area=8, opening=3,
           scales=[1, 4], win=256, stride=192, block=4096, seed=101, budget_h=11.0, train_deadline_h=8.0)


def find_prep():
    for base in [Path(os.environ.get("MN_PREP", "/kaggle/input"))]:
        for p in sorted(base.rglob("manifest.json")):
            if (p.parent / "X.npy").exists(): return p.parent
    raise SystemExit("не найден выход ядра подготовки (manifest.json + X.npy) в /kaggle/input")


def ensure_smp():
    try:
        import segmentation_models_pytorch  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.5.0"])


def build_model():
    import segmentation_models_pytorch as smp
    return smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=len(CLASSES))


def zscore(x):
    import torch
    m = x.mean(dim=(-1, -2), keepdim=True); s = x.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    return (x - m) / s


def loss_fn(logits, y, w):
    import torch, torch.nn.functional as F
    ce = F.cross_entropy(logits, y, weight=w)
    p = logits.softmax(1); oh = F.one_hot(y, p.shape[1]).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    return CFG["ce_w"] * ce + CFG["dice_w"] * dice


def train_fold(k, dev, X, M, meta, folds, t_start, n_left):
    import torch
    torch.manual_seed(CFG["seed"] + k); np.random.seed(CFG["seed"] + k)
    fo = folds[k]; valf = fo["val_family"]
    tr = [i for i, r in enumerate(meta) if int(r["fold"]) != k and r["family"] != valf]
    va = [i for i, r in enumerate(meta) if int(r["fold"]) != k and r["family"] == valf]
    assert not set(fo["test_regions"]) & {meta[i]["region"] for i in tr + va}, "утечка: участок теста в обучении"
    model = build_model().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    w = torch.tensor(CFG["class_weights"], device=dev)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    deadline = t_start + CFG["train_deadline_h"] * 3600
    best, bad, hist, best_state = 1e9, 0, [], None
    for ep in range(CFG["epochs"]):
        te = time.time(); model.train(); perm = np.random.permutation(tr); tl = 0.0
        for b in range(0, len(perm), CFG["batch"]):
            idx = np.sort(perm[b:b + CFG["batch"]])
            xb = zscore(torch.from_numpy(X[idx].astype(np.float32))[:, None].to(dev))
            yb = torch.from_numpy(M[idx].astype(np.int64)).to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                l = loss_fn(model(xb).float(), yb, w)
            opt.zero_grad(set_to_none=True); scaler.scale(l).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"]); scaler.step(opt); scaler.update()
            tl += float(l) * len(idx)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for b in range(0, len(va), 32):
                idx = va[b:b + 32]
                xb = zscore(torch.from_numpy(X[idx].astype(np.float32))[:, None].to(dev))
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    vl += float(loss_fn(model(xb).float(), torch.from_numpy(M[idx].astype(np.int64)).to(dev), w)) * len(idx)
        vl = vl / max(1, len(va)) if va else tl / max(1, len(tr)); sch.step(vl)
        hist.append({"ep": ep, "train": tl / max(1, len(tr)), "val": vl, "sec": round(time.time() - te, 1)})
        print(f"[fold {k}] ep {ep} train {hist[-1]['train']:.4f} val {vl:.4f} {hist[-1]['sec']} с", flush=True)
        if vl < best: best, bad, best_state = vl, 0, {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        else: bad += 1
        ep_sec = hist[-1]["sec"]
        if bad >= CFG["patience"]: break
        if time.time() + ep_sec * n_left > deadline: print(f"[fold {k}] стоп по времени", flush=True); break
    model.load_state_dict(best_state)
    torch.save({"state": best_state, "cfg": CFG, "fold": k, "val_family": valf, "test_regions": fo["test_regions"],
                "n_train": len(tr), "n_val": len(va), "hist": hist}, OUT / f"weights/fold{k}.pt")
    return model, {"fold": k, "n_train": len(tr), "n_val": len(va), "best_val": best, "epochs": len(hist)}


def postprocess(cls):
    import cv2
    def rm_small(m):
        out = m.copy()
        for c in range(1, len(CLASSES)):
            n, lab, st, _ = cv2.connectedComponentsWithStats((m == c).astype(np.uint8), connectivity=8)
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] < CFG["min_area"]: out[lab == i] = 0
        return out
    m = rm_small(cls)
    if CFG["opening"] > 1:
        ker = np.ones((CFG["opening"],) * 2, np.uint8); o = np.zeros_like(m)
        for c in range(1, len(CLASSES)):
            o[cv2.morphologyEx((m == c).astype(np.uint8), cv2.MORPH_OPEN, ker) > 0] = c
        m = rm_small(o)
    return m


def infer_raster(model, dev, path, crs, scale, region, markup, fold):
    """Полный растр блоками (с полем), окна 256 / шаг 192, z-score на окно, порог 0.3, полигоны → EPSG:3857."""
    import torch, rasterio, cv2
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    from rasterio.features import shapes
    from affine import Affine
    from shapely.geometry import shape as sshape
    from shapely.ops import transform as stransform
    from pyproj import Transformer
    to3857 = Transformer.from_crs(crs, "EPSG:3857", always_xy=True).transform
    feats = []
    W, S = CFG["win"], CFG["stride"]
    with rasterio.open(path) as ds:
        nod = ds.nodata
        core, halo = CFG["block"] * scale, 128 * scale
        for r0 in range(0, ds.height, core):
            for c0 in range(0, ds.width, core):
                rr0, cc0 = max(0, r0 - halo), max(0, c0 - halo)
                rr1, cc1 = min(ds.height, r0 + core + halo), min(ds.width, c0 + core + halo)
                h, w = max(1, (rr1 - rr0) // scale), max(1, (cc1 - cc0) // scale)
                a = ds.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0), out_shape=(h, w),
                            resampling=Resampling.bilinear if scale > 1 else Resampling.nearest).astype(np.float32)
                valid = np.isfinite(a) & (a != 0) & ((a != nod) if nod is not None else True)
                if valid.mean() < 0.02: continue
                a = np.nan_to_num(a)
                ph, pw = max(h, W), max(w, W)
                A = np.zeros((ph, pw), np.float32); A[:h, :w] = a
                ys = list(range(0, ph - W + 1, S)); xs = list(range(0, pw - W + 1, S))
                if ys[-1] != ph - W: ys.append(ph - W)
                if xs[-1] != pw - W: xs.append(pw - W)
                P = torch.zeros((len(CLASSES), ph, pw), device=dev); C = torch.zeros((ph, pw), device=dev)
                At = torch.from_numpy(A).to(dev); Vt = torch.from_numpy(np.pad(valid, ((0, ph - h), (0, pw - w)))).to(dev)
                wins = [(y, x) for y in ys for x in xs if Vt[y:y + W, x:x + W].float().mean() > 0.1]
                with torch.no_grad():
                    for b in range(0, len(wins), 64):
                        bw = wins[b:b + 64]
                        xb = zscore(torch.stack([At[y:y + W, x:x + W] for y, x in bw])[:, None])
                        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                            pr = model(xb).float().softmax(1)
                        for (y, x), p in zip(bw, pr):
                            P[:, y:y + W, x:x + W] += p; C[y:y + W, x:x + W] += 1
                P = (P / C.clamp_min(1)).cpu().numpy()[:, :h, :w]
                mx, cl = P.max(0), P.argmax(0).astype(np.uint8)
                cl[(mx < CFG["thr"]) | ~valid] = 0
                cl = postprocess(cl)
                tr = ds.transform * Affine.translation(cc0, rr0) * Affine.scale(scale)
                # ядро блока в координатах уменьшенной сетки
                ky0, kx0 = (r0 - rr0) // scale, (c0 - cc0) // scale
                ky1, kx1 = ky0 + core // scale, kx0 + core // scale
                for c in range(1, len(CLASSES)):
                    n, lab = cv2.connectedComponents((cl == c).astype(np.uint8), connectivity=8)
                    if n <= 1: continue
                    conf = np.bincount(lab.ravel(), weights=P[c].ravel(), minlength=n) / np.maximum(np.bincount(lab.ravel(), minlength=n), 1)
                    for g, v in shapes(lab.astype(np.int32), mask=lab > 0, transform=tr):
                        geom = sshape(g)
                        cx, cy = ~tr * (geom.centroid.x, geom.centroid.y)  # (col, row) уменьшенной сетки
                        if not (ky0 <= cy < ky1 and kx0 <= cx < kx1): continue
                        g3 = stransform(to3857, geom)
                        if g3.geom_type != "Polygon": g3 = max(getattr(g3, "geoms", [g3]), key=lambda q: q.area)
                        feats.append({"type": "Feature", "geometry": g3.__geo_interface__,
                                      "properties": {"class_name": CLASSES[c], "confidence": round(float(conf[int(v)]), 5),
                                                     "region_name": region, "crs": "EPSG:3857", "markup_type": markup,
                                                     "original_crs": crs, "threshold_group": f"mn_{markup}",
                                                     "fold": fold, "scale": scale}})
    return feats


def wait_file(p, size, timeout=3600):
    t = time.time()
    while not (p.exists() and p.stat().st_size == size):
        if time.time() - t > timeout: raise TimeoutError(str(p))
        time.sleep(5)
    return p


def worker(g, fold_ids, t_start):
    import torch
    dev = torch.device(f"cuda:{g}" if torch.cuda.is_available() else "cpu")
    prep = find_prep(); man = json.load(open(prep / "manifest.json", encoding="utf-8"))
    folds = man["folds"]["folds"]
    X = np.load(prep / "X.npy", mmap_mode="r"); M = np.load(prep / "M.npy", mmap_mode="r")
    meta = list(csv.DictReader(open(prep / "meta.csv", encoding="utf-8")))
    for j, k in enumerate(fold_ids):
        try:
            model, rep = train_fold(k, dev, X, M, meta, folds, t_start, len(fold_ids) - j)
            model.eval(); feats = []; ti = time.time()
            for region in folds[k]["test_regions"]:
                info = man["regions"].get(region, {})
                for mod, lst in info.get("rasters", {}).items():
                    for r in lst:
                        if "crs" not in r: continue
                        p = wait_file(CACHE / r["key"], r["size"])
                        for sc in CFG["scales"]:
                            feats += infer_raster(model, dev, p, r["crs"], sc, region, MOD2MARKUP[mod], k)
                        if os.environ.get("MN_KEEP", "0") != "1": p.unlink()  # фолды не пересекаются — растр больше не нужен
            rep.update(infer_sec=round(time.time() - ti), n_polys=len(feats))
            json.dump({"type": "FeatureCollection", "features": feats}, open(OUT / f"oof/fold{k}.geojson", "w", encoding="utf-8"), ensure_ascii=False)
        except Exception as e:
            rep = {"fold": k, "error": repr(e), "tb": traceback.format_exc()[-2000:]}
        json.dump(rep, open(OUT / f"logs/fold{k}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("[fold]", json.dumps(rep, ensure_ascii=False)[:500], flush=True)


def downloader(man, order):
    cli = s3_client()
    for region in order:
        for mod, lst in man["regions"].get(region, {}).get("rasters", {}).items():
            for r in lst:
                if "crs" in r: fetch(cli, r["key"], r["size"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--require-gpu", action="store_true", default=True)
    ap.add_argument("--allow-cpu", action="store_true"); ap.add_argument("--folds", default="0,1,2,3,4")
    a, _ = ap.parse_known_args()
    t_start = time.time()
    n = gpu_guard(require=not a.allow_cpu)
    ensure_smp()
    for d in ("weights", "oof", "logs"): (OUT / d).mkdir(parents=True, exist_ok=True)
    prep = find_prep(); man = json.load(open(prep / "manifest.json", encoding="utf-8"))
    fids = [int(x) for x in a.folds.split(",")]
    order = [r for k in fids for r in man["folds"]["folds"][k]["test_regions"]]
    th = threading.Thread(target=downloader, args=(man, order), daemon=True); th.start()
    import torch.multiprocessing as mp
    ng = max(1, min(n, 2))
    groups = [fids[g::ng] for g in range(ng)]
    if ng == 1: worker(0, groups[0], t_start)
    else:
        ps = [mp.get_context("spawn").Process(target=worker, args=(g, groups[g], t_start)) for g in range(ng)]
        for p in ps: p.start()
        for p in ps: p.join()
    feats = []
    for k in fids:
        f = OUT / f"oof/fold{k}.geojson"
        if f.exists(): feats += json.load(open(f, encoding="utf-8"))["features"]
    for i, f in enumerate(feats): f["properties"]["fid"] = i
    json.dump({"type": "FeatureCollection", "features": feats}, open(OUT / "oof_mn_deeplab.geojson", "w", encoding="utf-8"), ensure_ascii=False)
    logs = [json.load(open(p, encoding="utf-8")) for p in sorted((OUT / "logs").glob("fold*.json"))]
    json.dump({"cfg": CFG, "gpus": n, "total_sec": round(time.time() - t_start), "folds": logs, "n_polys": len(feats)},
              open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("ГОТОВО", len(feats), "полигонов,", round((time.time() - t_start) / 3600, 2), "ч", flush=True)


if __name__ == "__main__":
    main()
