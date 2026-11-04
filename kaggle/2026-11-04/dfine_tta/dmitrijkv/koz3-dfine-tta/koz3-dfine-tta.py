# KOZ3 · TTA D-FINE на спутнике SpOr: 8 видов D4 на весах фолдов прод-модели sp_dfine (только инференс).
# PREREG: kaggle/2026-11-04/dfine_tta/PREREG.md. Ядро собрано скриптом make_dfine_tta.py из кода ядра D-FINE-2
# (тот же путь, что дал OOF sp_dfine базы гейта) и функций expds/orsp_tta.py (прод-путь флага [orsp_tta]).
# Входы — приватные датасеты Kaggle и вывод ядер с весами; ключей и сетевых учёток в тексте нет.
# Выход: oof_{V0,T8N,T8W}_f{k}.geojson, oof_{V}_spor.geojson, tta_report.json.

import os, sys, subprocess, importlib
PINS = {"transformers": "transformers==5.21.0", "rasterio": "rasterio==1.4.3", "pyproj": "pyproj==3.7.1",
        "shapely": "shapely==2.1.1", "cv2": "opencv-python-headless==4.11.0.86", "scipy": "scipy==1.16.3"}
HF_REPO, HF_REV = "ustc-community/dfine-large-coco", "aa8f8211545e4cff5adf52ea512f9ac74a9824ec"
def _deps():
    need = []
    for mod, pin in PINS.items():
        try:
            m = importlib.import_module(mod)
            if mod == "transformers" and m.__version__ != pin.split("==")[1]: need.append(pin)  # D-FINE API — строго пин
        except Exception: need.append(pin)
    if need:
        print("pip install:", need, flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", *need])
        if r.returncode: raise SystemExit(f"pip install не удался ({need}); нужен enable_internet=true")
        for k in [k for k in sys.modules if k.split(".")[0] in ("transformers", "huggingface_hub", "tokenizers")]: del sys.modules[k]
        importlib.invalidate_caches()
    bad = []
    for mod in ["torch", "torchvision", "huggingface_hub", "safetensors", *PINS]:
        try: m = importlib.import_module(mod); print(f"  {mod} {getattr(m, '__version__', '?')}", flush=True)
        except Exception as e: bad.append(f"{mod}: {e!r}")
    try: from transformers import DFineForObjectDetection  # noqa
    except Exception as e: bad.append(f"DFineForObjectDetection: {e!r}")
    try:
        import transformers as _t
        if _t.__version__ != PINS["transformers"].split("==")[1]: bad.append(f"transformers {_t.__version__} != пин {PINS['transformers']}")
    except Exception as e: bad.append(f"transformers: {e!r}")
    if bad: raise SystemExit("импорт-чек не пройден: " + "; ".join(bad))
_deps()
FOLDS = __import__('json').loads(r'''{"family_of": {"001_Гнёздово_Нивленское": "Гнёздово", "002_Тешилово": "Тешилово", "003_Лубно": "Лубно", "004_Боровичи_1": "Боровичи", "005_Боровичи_2": "Боровичи", "006_Осечки_1.1км": "Миловидово", "007_Армиево": "Армиево", "009_Хальвасборген": "Хальвасборген", "010_Эсбю_укрепление": "Эсбю", "011_Эрентуна": "Эрентуна", "012_Сож": "Сож", "013_Нора_Виннон": "Нора", "015_Демидовка": "Демидовка", "016_Хога": "Хога", "017_Юшково": "Юшково", "018_Курган_Хога": "Курган", "019_Селяне": "Селяне", "020_Романово": "Романово", "021_Новенькое": "Новенькое", "022_Лихуша": "Лихуша", "023_Бервенец": "Бервенец", "024_Стрекаловка": "Стрекаловка", "025_Вышенское": "Вышенское", "026_Золотаревка": "Золотаревка", "027_Вышегжа_1": "Вышегжа", "028_Вышегжа_2": "Вышегжа", "029_Вышегжа_3": "Вышегжа", "030_Эсбю": "Эсбю", "031_Новотитаровская": "Новотитаровская", "032_Красносельская_0.5км": "Красносельская", "033_Елизаветинское": "Елизаветинское", "034_Усть_река": "Усть", "035_Шумгора": "Шумгора", "036_Тимерево": "Тимерево", "037_Никола_Ленивец": "Никола", "038_Копанское": "Копанское", "039_Примаки_0.7км": "Миловидово", "040_Примаки_1.3км": "Миловидово", "041_Миловидово": "Миловидово", "042_Тору_Айгыр": "Тору", "043_Гочево": "Гочево", "044_Новотитаровская_1.8км": "Новотитаровская", "045_Новотитаровская_2.8км": "Новотитаровская", "046_Тыва_1": "Тыва", "047_Тыва_2": "Тыва", "048_Калмыкия_1": "Калмыкия", "049_Калмыкия_2": "Калмыкия", "050_Воронеж_1": "Воронеж", "051_Воронеж_2": "Воронеж", "052_Воронеж_3": "Воронеж", "053_Воронеж_4": "Воронеж", "054_Воронеж_5": "Воронеж", "055_Курменту": "Курменту", "057_Кабардино_Балкария": "Кабардино", "058_Перынь": "Перынь", "059_Шишкино": "Шишкино", "060_Сельцо": "Сельцо", "061_Граборг": "Граборг", "062_Каменка": "Каменка", "063_Верхнее_Казачье": "Верхнее", "064_Ксизово": "Ксизово", "065_Шниткино": "Шниткино", "066_Московиты": "Московиты", "067_Чудово": "Чудово", "068_Стеклянуха_1": "Стеклянуха", "069_Стеклянуха_2": "Стеклянуха", "070_Архангельское": "Архангельское", "071_Сторожевое": "Сторожевое", "072_Белая_Гора": "Белая", "073_Тоссор": "Тоссор", "074_Солдатское": "Солдатское", "075_Мостище": "Мостище", "076_Маяцкая_крепость": "Маяцкая", "077_Русская_Тростянка": "Русская", "078_Верхняя_Покровка": "Верхняя", "079_Городище": "Городище", "080_Верхний_Карабут": "Верхний", "081_Костомаровское": "Костомаровское", "082_Артпозиции_Иссык_Куль": "Артпозиции", "083_Барскоон": "Барскоон", "084_Булан_Соготту": "Булан", "085_Каджы_Саз": "Иссык", "086_Кан_Дебе": "Кан", "087_Кароол_Дюбе": "Кароол", "088_Кель_Тор_Х": "Кель", "089_Конур_Елен": "Конур", "090_Корумду": "Корумду", "091_Сакские_курганы": "Сакские", "092_Шаты": "Шаты", "094_Иссык_Куль_курганы_и_выкладки": "Иссык", "095_Иссык_Куль_курганы_и_постройки_1": "Иссык", "096_Иссык_Куль_курганы_и_постройки_2": "Иссык", "097_Орнок": "Орнок", "098_Таш_Коро_1": "Таш", "099_Таш_Коро_2": "Таш", "100_Торт_Куль": "Торт", "101_Гнёздово_Днепровская": "Гнёздово", "102_Гнёздово_Заольшанская": "Гнёздово", "103_Гнёздово_Ольшанская": "Гнёздово", "104_Шетнево": "Шетнево", "105_Вщиж": "Вщиж", "106_Верхмарево": "Верхмарево", "107_Сары_Булун": "Сары", "108_Скупая_Полудань": "Скупая", "109_Россошки": "Россошки", "110_Саратов": "Саратов", "111_Каменные_выкладки_1": "Каменные", "112_Каменные_выкладки_2": "Каменные", "113_Каменные_выкладки_3": "Каменные", "114_Каменные_выкладки_4": "Каменные", "115_Каменные_выкладки_5": "Каменные", "116_Каменные_выкладки_6": "Каменные", "117_Каменные_выкладки_7": "Каменные", "118_Каменные_выкладки_8": "Каменные", "119_Каменные_выкладки_9": "Каменные", "120_Каменные_выкладки_10": "Каменные", "121_Каменные_выкладки_11": "Каменные", "122_Каменные_выкладки_12": "Каменные", "123_Каменные_выкладки_13": "Каменные", "124_Постройки_1": "Постройки", "125_Постройки_2": "Постройки", "126_Постройки_3": "Постройки", "127_Постройки_4": "Постройки", "128_Постройки_5": "Постройки", "129_Постройки_6": "Постройки", "130_Курганы_1": "Курганы", "131_Курганы_2": "Курганы", "132_Курганы_3": "Курганы", "133_Курганы_4": "Курганы", "134_Курганы_5": "Курганы", "135_Курганы_6": "Курганы", "136_Курганы_7": "Курганы", "137_Курганы_8": "Курганы", "138_Курганы_9": "Курганы", "139_Курганы_10": "Курганы", "140_Курганы_11": "Курганы", "141_Курганы_12": "Курганы", "142_Курганы_13": "Курганы", "143_Курганы_14": "Курганы", "144_Курганы_15": "Курганы", "145_Курганы_16": "Курганы", "146_Курганы_17": "Курганы", "147_Курганы_18": "Курганы", "148_Курганы_19": "Курганы", "149_Курганы_20": "Курганы", "150_Курганы_21": "Курганы", "151_Курганы_22": "Курганы", "152_Курганы_23": "Курганы", "153_Курганы_24": "Курганы", "154_Курганы_25": "Курганы", "155_Курганы_26": "Курганы", "156_Курганы_27": "Курганы", "157_Курганы_28": "Курганы", "158_Курганы_29": "Курганы", "159_Курганы_30": "Курганы", "160_Курганы_31": "Курганы", "161_Курганы_32": "Курганы", "162_Курганы_33": "Курганы", "163_Курганы_34": "Курганы", "164_Кисловодская_котловина_1": "Кисловодская", "165_Кисловодская_котловина_2": "Кисловодская", "166_Кисловодская_котловина_3": "Кисловодская", "167_Кисловодская_котловина_4": "Кисловодская", "168_Кисловодская_котловина_5": "Кисловодская", "169_Кисловодская_котловина_6": "Кисловодская", "170_Кисловодская_котловина_7": "Кисловодская", "171_Кисловодская_котловина_8": "Кисловодская", "172_Кисловодская_котловина_9": "Кисловодская", "173_Кисловодская_котловина_10": "Кисловодская", "174_Кисловодская_котловина_11": "Кисловодская", "175_Кисловодская_котловина_12": "Кисловодская", "176_Кисловодская_котловина_13": "Кисловодская", "177_Кисловодская_котловина_14": "Кисловодская", "178_Кисловодская_котловина_15": "Кисловодская", "179_Гнёздово_Лесная_1": "Гнёздово", "180_Гнёздово_Лесная_2": "Гнёздово", "181_Гнёздово_Лесная_3": "Гнёздово", "182_Красносельская_1.7км": "Красносельская", "183_Сарское": "Сарское", "184_Шурскол": "Шурскол", "185_Изборск": "Изборск", "186_Семибратнее_1": "Семибратнее", "187_Судьбищи_1": "Судьбищи", "188_Руна": "Руна", "189_Макеево": "Макеево", "190_Муравлево": "Муравлево", "191_Осечки_1.3км": "Миловидово", "192_КЧР": "КЧР", "200_Судьбищи_2": "Судьбищи"}, "folds": [{"fold": 0, "test_families": ["Гнёздово"], "test_regions": ["001_Гнёздово_Нивленское", "101_Гнёздово_Днепровская", "102_Гнёздово_Заольшанская", "103_Гнёздово_Ольшанская", "179_Гнёздово_Лесная_1", "180_Гнёздово_Лесная_2", "181_Гнёздово_Лесная_3"]}, {"fold": 1, "test_families": ["Булан", "Верхняя", "Воронеж", "Кан", "Корумду", "Курменту", "Постройки", "Россошки", "Солдатское", "Сторожевое", "Тимерево", "Тыва", "Шаты", "Шишкино", "Шниткино"], "test_regions": ["036_Тимерево", "046_Тыва_1", "047_Тыва_2", "050_Воронеж_1", "051_Воронеж_2", "052_Воронеж_3", "053_Воронеж_4", "054_Воронеж_5", "055_Курменту", "059_Шишкино", "065_Шниткино", "071_Сторожевое", "074_Солдатское", "078_Верхняя_Покровка", "084_Булан_Соготту", "086_Кан_Дебе", "090_Корумду", "092_Шаты", "109_Россошки", "124_Постройки_1", "125_Постройки_2", "126_Постройки_3", "127_Постройки_4", "128_Постройки_5", "129_Постройки_6"]}, {"fold": 2, "test_families": ["Артпозиции", "Верхнее", "Вышенское", "Городище", "Демидовка", "Золотаревка", "КЧР", "Кароол", "Костомаровское", "Курганы", "Лихуша", "Московиты", "Никола", "Новотитаровская", "Русская", "Сакские", "Саратов", "Таш", "Торт", "Усть", "Эсбю"], "test_regions": ["010_Эсбю_укрепление", "015_Демидовка", "022_Лихуша", "025_Вышенское", "026_Золотаревка", "030_Эсбю", "031_Новотитаровская", "034_Усть_река", "037_Никола_Ленивец", "044_Новотитаровская_1.8км", "045_Новотитаровская_2.8км", "063_Верхнее_Казачье", "066_Московиты", "077_Русская_Тростянка", "079_Городище", "081_Костомаровское", "082_Артпозиции_Иссык_Куль", "087_Кароол_Дюбе", "091_Сакские_курганы", "098_Таш_Коро_1", "099_Таш_Коро_2", "100_Торт_Куль", "110_Саратов", "130_Курганы_1", "131_Курганы_2", "132_Курганы_3", "133_Курганы_4", "134_Курганы_5", "135_Курганы_6", "136_Курганы_7", "137_Курганы_8", "138_Курганы_9", "139_Курганы_10", "140_Курганы_11", "141_Курганы_12", "142_Курганы_13", "143_Курганы_14", "144_Курганы_15", "145_Курганы_16", "146_Курганы_17", "147_Курганы_18", "148_Курганы_19", "149_Курганы_20", "150_Курганы_21", "151_Курганы_22", "152_Курганы_23", "153_Курганы_24", "154_Курганы_25", "155_Курганы_26", "156_Курганы_27", "157_Курганы_28", "158_Курганы_29", "159_Курганы_30", "160_Курганы_31", "161_Курганы_32", "162_Курганы_33", "163_Курганы_34", "192_КЧР"]}, {"fold": 3, "test_families": ["Бервенец", "Боровичи", "Верхмарево", "Вщиж", "Гочево", "Елизаветинское", "Иссык", "Кель", "Ксизово", "Макеево", "Миловидово", "Новенькое", "Нора", "Орнок", "Руна", "Сарское", "Сельцо", "Семибратнее", "Скупая", "Тешилово", "Тору", "Чудово", "Шетнево", "Шурскол"], "test_regions": ["002_Тешилово", "004_Боровичи_1", "005_Боровичи_2", "006_Осечки_1.1км", "013_Нора_Виннон", "021_Новенькое", "023_Бервенец", "033_Елизаветинское", "039_Примаки_0.7км", "040_Примаки_1.3км", "041_Миловидово", "042_Тору_Айгыр", "043_Гочево", "060_Сельцо", "064_Ксизово", "067_Чудово", "085_Каджы_Саз", "088_Кель_Тор_Х", "094_Иссык_Куль_курганы_и_выкладки", "095_Иссык_Куль_курганы_и_постройки_1", "096_Иссык_Куль_курганы_и_постройки_2", "097_Орнок", "104_Шетнево", "105_Вщиж", "106_Верхмарево", "108_Скупая_Полудань", "183_Сарское", "184_Шурскол", "186_Семибратнее_1", "188_Руна", "189_Макеево", "191_Осечки_1.3км"]}, {"fold": 4, "test_families": ["Армиево", "Архангельское", "Барскоон", "Белая", "Верхний", "Вышегжа", "Граборг", "Изборск", "Кабардино", "Калмыкия", "Каменка", "Каменные", "Кисловодская", "Конур", "Копанское", "Красносельская", "Курган", "Лубно", "Маяцкая", "Мостище", "Муравлево", "Перынь", "Романово", "Сары", "Селяне", "Сож", "Стеклянуха", "Стрекаловка", "Судьбищи", "Тоссор", "Хальвасборген", "Хога", "Шумгора", "Эрентуна", "Юшково"], "test_regions": ["003_Лубно", "007_Армиево", "009_Хальвасборген", "011_Эрентуна", "012_Сож", "016_Хога", "017_Юшково", "018_Курган_Хога", "019_Селяне", "020_Романово", "024_Стрекаловка", "027_Вышегжа_1", "028_Вышегжа_2", "029_Вышегжа_3", "032_Красносельская_0.5км", "035_Шумгора", "038_Копанское", "048_Калмыкия_1", "049_Калмыкия_2", "057_Кабардино_Балкария", "058_Перынь", "061_Граборг", "062_Каменка", "068_Стеклянуха_1", "069_Стеклянуха_2", "070_Архангельское", "072_Белая_Гора", "073_Тоссор", "075_Мостище", "076_Маяцкая_крепость", "080_Верхний_Карабут", "083_Барскоон", "089_Конур_Елен", "107_Сары_Булун", "111_Каменные_выкладки_1", "112_Каменные_выкладки_2", "113_Каменные_выкладки_3", "114_Каменные_выкладки_4", "115_Каменные_выкладки_5", "116_Каменные_выкладки_6", "117_Каменные_выкладки_7", "118_Каменные_выкладки_8", "119_Каменные_выкладки_9", "120_Каменные_выкладки_10", "121_Каменные_выкладки_11", "122_Каменные_выкладки_12", "123_Каменные_выкладки_13", "164_Кисловодская_котловина_1", "165_Кисловодская_котловина_2", "166_Кисловодская_котловина_3", "167_Кисловодская_котловина_4", "168_Кисловодская_котловина_5", "169_Кисловодская_котловина_6", "170_Кисловодская_котловина_7", "171_Кисловодская_котловина_8", "172_Кисловодская_котловина_9", "173_Кисловодская_котловина_10", "174_Кисловодская_котловина_11", "175_Кисловодская_котловина_12", "176_Кисловодская_котловина_13", "177_Кисловодская_котловина_14", "178_Кисловодская_котловина_15", "182_Красносельская_1.7км", "185_Изборск", "187_Судьбищи_1", "190_Муравлево", "200_Судьбищи_2"]}]}''')
import json, time, re, math, glob, random, threading, argparse, collections, concurrent.futures as cf
from pathlib import Path
import numpy as np, torch

DATASETS_HINT = ["dmitrijkv/koz3-w2-dfine2-raw-a", "dmitrijkv/koz3-w2-dfine2-raw-b"]
CLASSES = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
E = lambda k, d: type(d)(os.environ.get(k, d))
CFG = dict(res=0.5, tile=640, overlap=160, bs=E("TTA_BS", 8), min_score=0.05, nms_iou=0.5, max_det=100, wbf_iou=0.55,
           max_windows=E("TTA_MAX_WINDOWS", 0), regions=os.environ.get("TTA_REGIONS", ""), folds=os.environ.get("TTA_FOLDS", "0,1,2,3,4"),
           deadline_h=E("TTA_DEADLINE_H", 8.0), group="sp_dfine")
VIEWS = [(k, f) for k in range(4) for f in (False, True)]          # == orsp_tta._view_tuples("d8")
V0_VIEWS = [VIEWS.index((0, False)), VIEWS.index((0, True))]       # путь исходного OOF: как есть + отражение
OUT = Path(os.environ.get("TTA_OUT", "/kaggle/working")); OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time(); DEADLINE = T0 + CFG["deadline_h"] * 3600
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
log("CFG", json.dumps(CFG, ensure_ascii=False))
AP = argparse.ArgumentParser(); AP.add_argument("--allow-cpu", action="store_true"); ARGS, _ = AP.parse_known_args()

# ── Шаг 1. --require-gpu ─────────────────────────────────────────────────────────────────────────────────────
NG = torch.cuda.device_count() if torch.cuda.is_available() else 0
log("GPU:", [torch.cuda.get_device_name(i) for i in range(NG)])
if NG == 0 and not ARGS.allow_cpu:
    raise SystemExit("--require-gpu: GPU нет (сессия получила CPU — второй одновременный GPU-кернел?). Останов.")
DEVS = [torch.device(f"cuda:{i}") for i in range(NG)][:2] or [torch.device("cpu")]

def _inputs():
    roots = [Path(p) for p in os.environ.get("DFINE_RAW", "").split(":") if p] or [Path("/kaggle/input")]
    # Kaggle вырезает не-ASCII из имён файлов датасета, поэтому там лежат <NNN>__<kk>_SpOr.tif + manifest_*.json
    # {ascii_имя: {"region", "file"}}; локально (без манифеста) имя <участок>__<файл>.tif разбирается как есть.
    files, gts, man = {}, [], {}
    for r in roots:
        for p in sorted(r.rglob("*")):
            if not p.is_file(): continue
            if p.name.lower().endswith((".tif", ".tiff")) and "__" in p.name and "SpOr" in p.name: files.setdefault(p.name, p)
            elif p.name == "gt_train_v2.geojson": gts.append(p)
            elif p.name.startswith("manifest_") and p.suffix == ".json": man.update(json.load(open(p, encoding="utf-8")))
    lost = sorted(set(man) - set(files))
    if lost: raise SystemExit(f"файлы из manifest не найдены: {len(lost)} (напр. {lost[:3]}); датасеты {DATASETS_HINT}")
    tifs = {}  # путь → (участок, исходное имя файла)
    for n, p in files.items():
        tifs[str(p)] = (man[n]["region"], man[n]["file"]) if n in man else tuple(n.split("__", 1))
    gt = Path(os.environ["DFINE_GT"]) if os.environ.get("DFINE_GT") else (gts[0] if gts else None)
    return tifs, gt
TIF_INFO, GT_PATH = _inputs()
TIF_PATHS = sorted(Path(k) for k in TIF_INFO)
if not TIF_PATHS or GT_PATH is None or not GT_PATH.exists():
    raise SystemExit(f"входы не найдены: растров {len(TIF_PATHS)}, GT {GT_PATH} (датасеты {DATASETS_HINT})")
def _load_gt_all(p):
    from shapely.geometry import shape
    out = collections.defaultdict(list)
    for f in json.load(open(p, encoding="utf-8"))["features"]:
        q = f.get("properties") or {}; g = f.get("geometry")
        if q.get("markup_type") != "sp_or" or q.get("class_name") not in CLASSES or not g: continue
        if g.get("type") not in ("Polygon", "MultiPolygon"): continue
        out[q["region_name"]].append((shape(g), CLASSES.index(q["class_name"])))  # EPSG:3857
    return out
GT_BY = _load_gt_all(GT_PATH)
log(f"входы OK: растров {len(TIF_PATHS)}, GT {GT_PATH.name}: {sum(map(len, GT_BY.values()))} объектов sp_or, участков {len(GT_BY)}")
from transformers import PreTrainedModel as _PTM
_PTM_ORIG = {n: _PTM.__dict__[n] for n in ("tie_weights", "init_weights") if n in _PTM.__dict__}  # снято в главном потоке до загрузок
MODEL_LOCK = threading.Lock()
def _meta_names(m):
    return [n for n, p in m.named_parameters(remove_duplicate=False) if p.is_meta] + \
           [n for n, b in m.named_buffers(remove_duplicate=False) if b.is_meta]
def _repair_meta(m, meta):
    # 1) повторная привязка весов исходной tie_weights (class_embed/bbox_embed → model.decoder.*)
    log(f"модель: на meta {len(meta)} тензоров ({', '.join(meta[:3])}…) — tie_weights()")
    _PTM_ORIG.get("tie_weights", type(m).tie_weights)(m)
    meta = _meta_names(m)
    if meta:  # 2) материализация оставшихся: to_empty + _init_weights модуля, затем снова привязка
        log(f"модель: после tie_weights на meta ещё {len(meta)} — to_empty + _init_weights")
        mods = {n.rsplit(".", 1)[0] if "." in n else "" for n in meta}
        for mn in sorted(mods):
            mod = m.get_submodule(mn); mod.to_empty(device="cpu", recurse=False); m._init_weights(mod)
        _PTM_ORIG.get("tie_weights", type(m).tie_weights)(m)
        meta = _meta_names(m)
    return meta
def check_model(m):
    """Жёсткие проверки ДО .to(device): нет meta, голова на 5 классов, привязка верхних голов к декодеру."""
    meta = _meta_names(m)
    if meta: raise RuntimeError(f"D-FINE: {len(meta)} параметров/буферов на meta ({', '.join(meta[:4])}…): "
                                "веса не материализованы (гонка from_pretrained/tie_weights?) — .to(device) упадёт")
    assert not any(p.is_meta for p in m.parameters()) and not any(b.is_meta for b in m.buffers())
    K, d = len(CLASSES), m.config.d_model; dec = m.model.decoder
    assert m.config.num_labels == K, f"num_labels {m.config.num_labels} != {K}"
    for i, l in enumerate(dec.class_embed):
        assert tuple(l.weight.shape) == (K, d) and tuple(l.bias.shape) == (K,), f"decoder.class_embed[{i}] {tuple(l.weight.shape)}"
    assert tuple(m.model.enc_score_head.weight.shape) == (K, d), f"enc_score_head {tuple(m.model.enc_score_head.weight.shape)}"
    if m.config.num_denoising > 0:
        assert m.model.denoising_class_embed.num_embeddings == K + 1, f"denoising_class_embed {m.model.denoising_class_embed.num_embeddings}"
    assert len(m.class_embed) == len(dec.class_embed) and len(m.bbox_embed) == len(dec.bbox_embed)
    for a, b in [(m.class_embed, dec.class_embed), (m.bbox_embed, dec.bbox_embed)]:
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert pa.data_ptr() == pb.data_ptr(), "class_embed/bbox_embed не привязаны к model.decoder.*"
    return {"params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2), "num_labels": K,
            "heads": [len(dec.class_embed), K, d]}
# ── Шаг 3. Фолды по семьям (проверка) и инвентарь участков ─────────────────────────────────────────────────────
FAM = FOLDS["family_of"]; FOLD = {r: f["fold"] for f in FOLDS["folds"] for r in f["test_regions"]}
ff = collections.defaultdict(set)
for r, f in FOLD.items(): ff[FAM[r]].add(f)
assert not {k: v for k, v in ff.items() if len(v) > 1}, "семья в двух фолдах"
log("фолды OK:", len(FOLD), "участков,", len(ff), "семей")

def fetch(key, size):  # растр уже смонтирован датасетом Kaggle — путь как есть
    return Path(key)

def region_inv(region, paths):
    return {"region": region, "tifs": sorted((str(p), p.stat().st_size) for p in paths), "marks": bool(GT_BY.get(region)), "utm": None}

_by_reg = collections.defaultdict(list)
for _p in TIF_PATHS: _by_reg[TIF_INFO[str(_p)][0]].append(_p)
regions = sorted(_by_reg)
if CFG["regions"]: regions = [r for r in regions if r in CFG["regions"].split(",")]
INV = [region_inv(r, _by_reg[r]) for r in regions]
skipped = [i["region"] for i in INV if i["region"] not in FOLD]
INV = [i for i in INV if i["region"] in FOLD]
log(f"участков со SpOr в фолдах: {len(INV)} (с GT {sum(bool(i['marks']) for i in INV)}); вне фолдов (пропуск): {len(skipped)}")

def target_crs(inv, ds):
    if inv["utm"]:
        d = json.load(open(fetch(*inv["utm"]), encoding="utf-8")); m = re.search(r"EPSG::?(\d+)", d.get("crs", ""))
        if m: return f"EPSG:{m.group(1)}"
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(*ds.xy(ds.height // 2, ds.width // 2))
    return f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"

def load_gt(inv, crs):
    from shapely.ops import transform as st
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:3857", crs, always_xy=True).transform; out = []
    for g, c in GT_BY.get(inv["region"], []):
        g = st(tr, g); g = g if g.is_valid else g.buffer(0)
        if not g.is_empty: out.append((g.bounds, c))
    return out

class Scene:
    """Растр SpOr, приведённый на лету к UTM 0,5 м (WarpedVRT; при уменьшении — average, иначе bilinear), RGB uint8."""
    def __init__(self, inv, key, size):
        import rasterio
        from rasterio.vrt import WarpedVRT
        from rasterio.enums import Resampling
        from rasterio.warp import calculate_default_transform
        self.src = rasterio.open(fetch(key, size)); self.crs = target_crs(inv, self.src)
        tr, w, h = calculate_default_transform(self.src.crs, self.crs, self.src.width, self.src.height, *self.src.bounds,
                                               resolution=CFG["res"])
        try: src_m = abs(calculate_default_transform(self.src.crs, self.crs, self.src.width, self.src.height, *self.src.bounds)[0].a)
        except Exception: src_m = CFG["res"]
        rs = Resampling.average if src_m < CFG["res"] else Resampling.bilinear
        self.vrt = WarpedVRT(self.src, crs=self.crs, transform=tr, width=w, height=h, resampling=rs, nodata=0)
        self.W, self.H, self.T = w, h, tr; self.bands = list(range(1, min(3, self.src.count) + 1))
        self.stretch = None
        if self.src.dtypes[0] != "uint8":
            ov = self.src.read(1, out_shape=(min(1024, self.src.height), min(1024, self.src.width)))
            v = ov[ov > 0]; self.stretch = tuple(np.percentile(v, [1, 99])) if v.size else (0, 1)
    def read(self, x0, y0, n):
        from rasterio.windows import Window
        # WarpedVRT не умеет boundless — читаем пересечение и дополняем нулями вручную
        a = np.zeros((len(self.bands), n, n), dtype=self.src.dtypes[0])
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(self.W, x0 + n), min(self.H, y0 + n)
        if cx1 > cx0 and cy1 > cy0:
            a[:, cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = self.vrt.read(self.bands, window=Window(cx0, cy0, cx1 - cx0, cy1 - cy0))
        if a.shape[0] < 3: a = np.repeat(a[:1], 3, 0)
        if self.stretch:
            lo, hi = self.stretch; a = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        return np.ascontiguousarray(a.astype(np.uint8).transpose(1, 2, 0))
    def to_px(self, b):
        inv = ~self.T; (x0, y1), (x1, y0) = inv * (b[0], b[1]), inv * (b[2], b[3])
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    def close(self): self.vrt.close(); self.src.close()


# ── Поправка 1 PREREG: зона UTM участка — как в исходном OOF и в проде (из UTM.json), а не по центру растра ────────
REGION_CRS = {"015_Демидовка": "EPSG:32636", "026_Золотаревка": "EPSG:32638", "031_Новотитаровская": "EPSG:32637", "033_Елизаветинское": "EPSG:32637", "034_Усть_река": "EPSG:32636", "035_Шумгора": "EPSG:32636", "040_Примаки_1.3км": "EPSG:32637", "041_Миловидово": "EPSG:32637", "042_Тору_Айгыр": "EPSG:32643", "043_Гочево": "EPSG:32636", "046_Тыва_1": "EPSG:32646", "047_Тыва_2": "EPSG:32646", "048_Калмыкия_1": "EPSG:32638", "049_Калмыкия_2": "EPSG:32638", "050_Воронеж_1": "EPSG:32637", "051_Воронеж_2": "EPSG:32637", "052_Воронеж_3": "EPSG:32637", "053_Воронеж_4": "EPSG:32637", "054_Воронеж_5": "EPSG:32637", "055_Курменту": "EPSG:32644", "057_Кабардино_Балкария": "EPSG:32638", "062_Каменка": "EPSG:32637", "063_Верхнее_Казачье": "EPSG:32637", "064_Ксизово": "EPSG:32637", "070_Архангельское": "EPSG:32637", "071_Сторожевое": "EPSG:32637", "072_Белая_Гора": "EPSG:32637", "073_Тоссор": "EPSG:32643", "074_Солдатское": "EPSG:32637", "075_Мостище": "EPSG:32637", "076_Маяцкая_крепость": "EPSG:32637", "077_Русская_Тростянка": "EPSG:32637", "078_Верхняя_Покровка": "EPSG:32637", "079_Городище": "EPSG:32643", "080_Верхний_Карабут": "EPSG:32637", "081_Костомаровское": "EPSG:32637", "082_Артпозиции_Иссык_Куль": "EPSG:32643", "083_Барскоон": "EPSG:32643", "084_Булан_Соготту": "EPSG:32643", "085_Каджы_Саз": "EPSG:32643", "086_Кан_Дебе": "EPSG:32643", "087_Кароол_Дюбе": "EPSG:32643", "088_Кель_Тор_Х": "EPSG:32643", "089_Конур_Елен": "EPSG:32643", "090_Корумду": "EPSG:32644", "091_Сакские_курганы": "EPSG:32643", "092_Шаты": "EPSG:32644", "094_Иссык_Куль_курганы_и_выкладки": "EPSG:32644", "095_Иссык_Куль_курганы_и_постройки_1": "EPSG:32644", "096_Иссык_Куль_курганы_и_постройки_2": "EPSG:32644", "097_Орнок": "EPSG:32644", "098_Таш_Коро_1": "EPSG:32644", "099_Таш_Коро_2": "EPSG:32644", "100_Торт_Куль": "EPSG:32644", "101_Гнёздово_Днепровская": "EPSG:32636", "103_Гнёздово_Ольшанская": "EPSG:32636", "107_Сары_Булун": "EPSG:32643", "108_Скупая_Полудань": "EPSG:32637", "109_Россошки": "EPSG:32637", "111_Каменные_выкладки_1": "EPSG:32644", "112_Каменные_выкладки_2": "EPSG:32644", "113_Каменные_выкладки_3": "EPSG:32644", "114_Каменные_выкладки_4": "EPSG:32644", "115_Каменные_выкладки_5": "EPSG:32644", "116_Каменные_выкладки_6": "EPSG:32644", "117_Каменные_выкладки_7": "EPSG:32644", "118_Каменные_выкладки_8": "EPSG:32644", "119_Каменные_выкладки_9": "EPSG:32644", "120_Каменные_выкладки_10": "EPSG:32644", "121_Каменные_выкладки_11": "EPSG:32644", "122_Каменные_выкладки_12": "EPSG:32644", "123_Каменные_выкладки_13": "EPSG:32644", "124_Постройки_1": "EPSG:32644", "125_Постройки_2": "EPSG:32644", "126_Постройки_3": "EPSG:32644", "127_Постройки_4": "EPSG:32644", "128_Постройки_5": "EPSG:32644", "129_Постройки_6": "EPSG:32644", "130_Курганы_1": "EPSG:32644", "131_Курганы_2": "EPSG:32644", "132_Курганы_3": "EPSG:32644", "133_Курганы_4": "EPSG:32644", "134_Курганы_5": "EPSG:32644", "135_Курганы_6": "EPSG:32644", "136_Курганы_7": "EPSG:32644", "137_Курганы_8": "EPSG:32644", "138_Курганы_9": "EPSG:32644", "139_Курганы_10": "EPSG:32644", "140_Курганы_11": "EPSG:32644", "141_Курганы_12": "EPSG:32644", "142_Курганы_13": "EPSG:32644", "143_Курганы_14": "EPSG:32644", "144_Курганы_15": "EPSG:32644", "145_Курганы_16": "EPSG:32644", "146_Курганы_17": "EPSG:32644", "147_Курганы_18": "EPSG:32644", "148_Курганы_19": "EPSG:32644", "149_Курганы_20": "EPSG:32644", "150_Курганы_21": "EPSG:32644", "151_Курганы_22": "EPSG:32644", "152_Курганы_23": "EPSG:32644", "153_Курганы_24": "EPSG:32644", "154_Курганы_25": "EPSG:32644", "155_Курганы_26": "EPSG:32637", "156_Курганы_27": "EPSG:32637", "157_Курганы_28": "EPSG:32637", "158_Курганы_29": "EPSG:32637", "159_Курганы_30": "EPSG:32637", "160_Курганы_31": "EPSG:32637", "161_Курганы_32": "EPSG:32637", "162_Курганы_33": "EPSG:32638", "163_Курганы_34": "EPSG:32638", "164_Кисловодская_котловина_1": "EPSG:32638", "165_Кисловодская_котловина_2": "EPSG:32638", "166_Кисловодская_котловина_3": "EPSG:32638", "167_Кисловодская_котловина_4": "EPSG:32638", "168_Кисловодская_котловина_5": "EPSG:32638", "169_Кисловодская_котловина_6": "EPSG:32638", "170_Кисловодская_котловина_7": "EPSG:32638", "171_Кисловодская_котловина_8": "EPSG:32638", "172_Кисловодская_котловина_9": "EPSG:32638", "173_Кисловодская_котловина_10": "EPSG:32638", "174_Кисловодская_котловина_11": "EPSG:32638", "175_Кисловодская_котловина_12": "EPSG:32638", "176_Кисловодская_котловина_13": "EPSG:32638", "177_Кисловодская_котловина_14": "EPSG:32638", "178_Кисловодская_котловина_15": "EPSG:32638", "183_Сарское": "EPSG:32637", "184_Шурскол": "EPSG:32637", "185_Изборск": "EPSG:32635", "186_Семибратнее_1": "EPSG:32637", "192_КЧР": "EPSG:32637"}
_target_crs_auto = target_crs
def target_crs(inv, ds):
    return REGION_CRS.get(inv["region"]) or _target_crs_auto(inv, ds)
log("зоны UTM из исходного OOF:", len(REGION_CRS), "участков")

import numpy as np
def _inv_point(x: float, y: float, rot: int, hflip: bool, size: float) -> tuple[float, float]:
    """Точка из координат повёрнутого кадра в исходный. size — сторона квадратного тайла.

    Прямое преобразование кадра: rot90 против часовой (k раз), затем отражение по ширине.
    Непрерывные координаты на отрезке [0, size], не индексы пикселей.
    """
    if hflip:
        x = size - x
    rot %= 4
    if rot == 0:
        return x, y
    if rot == 1:  # был CCW 90: (x, y) -> (y, size - x)
        return size - y, x
    if rot == 2:
        return size - x, size - y
    return y, size - x

def invert_xyxy(xyxy: np.ndarray, rot: int, hflip: bool, size: float) -> np.ndarray:
    """xyxy в пикселях повёрнутого тайла → xyxy исходного тайла."""
    src = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(src)
    for i, (x0, y0, x1, y1) in enumerate(src):
        pts = (_inv_point(x0, y0, rot, hflip, size), _inv_point(x1, y0, rot, hflip, size),
               _inv_point(x1, y1, rot, hflip, size), _inv_point(x0, y1, rot, hflip, size))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out[i] = (min(xs), min(ys), max(xs), max(ys))
    return out

def _iou(b: np.ndarray, other: np.ndarray) -> np.ndarray:
    w = np.clip(np.minimum(b[2], other[:, 2]) - np.maximum(b[0], other[:, 0]), 0, None)
    h = np.clip(np.minimum(b[3], other[:, 3]) - np.maximum(b[1], other[:, 1]), 0, None)
    inter = w * h
    ua = (b[2] - b[0]) * (b[3] - b[1]) + (other[:, 2] - other[:, 0]) * (other[:, 3] - other[:, 1]) - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-9), 0.0)

def wbf(boxes, scores, labels, n_views: int, iou: float = 0.55):
    """Weighted boxes fusion по классу. Уверенность — сумма / max(n_views, число членов), обрезка до 1.

    Объект, видимый на всех видах, не штрафуется. Одиночный вид делится на n_views — меньше ложных.
    Координаты — среднее с весами-уверенностями. Разные классы не сливаются.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    oB, oS, oL = [], [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        fused: list[np.ndarray] = []
        members: list[list[int]] = []
        for i in idx:
            j = -1
            if fused:
                ious = _iou(boxes[i], np.asarray(fused))
                k = int(np.argmax(ious))
                if ious[k] >= iou:
                    j = k
            if j < 0:
                fused.append(boxes[i].copy())
                members.append([int(i)])
            else:
                members[j].append(int(i))
                m = np.asarray(members[j])
                w = scores[m]
                fused[j] = (boxes[m] * w[:, None]).sum(0) / max(float(w.sum()), 1e-12)
        for fb, m in zip(fused, members):
            oB.append(fb)
            oS.append(float(scores[m].sum() / max(n_views, len(m))))
            oL.append(int(c))
    if not oB:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return np.asarray(oB), np.clip(np.asarray(oS), 0, 1), np.asarray(oL, dtype=int)

def nms(boxes, scores, labels, iou: float = 0.5):
    """NMS по классам. Остаётся бокс с большей уверенностью, пересечения подавляются."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    keep: list[int] = []
    for c in np.unique(labels):
        idx = list(np.where(labels == c)[0][np.argsort(-scores[labels == c], kind="stable")])
        while idx:
            i = idx.pop(0)
            keep.append(i)
            if idx:
                ious = _iou(boxes[i], boxes[np.asarray(idx)])
                idx = [j for j, v in zip(idx, ious) if v < iou]
    keep = sorted(keep)
    if not keep:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return boxes[keep], scores[keep], labels[keep]

def fuse_views(parts: list, how: str, iou: float, nms_iou: float):
    """Слияние видов одного растра. Пустой вид входит в n_views, но не добавляет боксов.

    Один вид возвращается как есть (без деления уверенности). Разные растры сюда не передавать.
    """
    n = len(parts)
    boxes, scores, labels = [], [], []
    for B, L, C in parts:
        if len(np.asarray(L).reshape(-1)) == 0:
            continue
        boxes.append(np.asarray(B, dtype=np.float64).reshape(-1, 4))
        scores.append(np.asarray(L, dtype=np.float64).reshape(-1))
        labels.append(np.asarray(C, dtype=int).reshape(-1))
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    B = np.concatenate(boxes)
    L = np.concatenate(scores)
    C = np.concatenate(labels)
    if n <= 1:
        return B, L, C
    if how == "nms":
        return nms(B, L, C, nms_iou)
    return wbf(B, L, C, n, iou)
WEIGHTS_HINT = ["dmitrijkv/koz3-dfine-spor-full", "dmitrijkv/koz3-dfine-spor-full-rest"]

# ── Веса фолдов прод-модели sp_dfine: вывод ядер koz3-dfine-spor-full (f1, f2, f4) и -full-rest (f0, f3) ─────────
def _weight_dirs():
    roots = [Path(p) for p in os.environ.get("TTA_WEIGHTS", "").split(":") if p] or [Path("/kaggle/input")]
    out = {}
    for r in roots:
        for p in sorted(r.rglob("model.safetensors")):
            m = re.fullmatch(r"dfine_f(\d)", p.parent.name)
            if m and (p.parent / "config.json").exists():
                out.setdefault(int(m.group(1)), p.parent)
    return out
WDIRS = _weight_dirs()
log("веса фолдов:", {k: str(v) for k, v in sorted(WDIRS.items())})

def load_fold(k):
    from transformers import DFineForObjectDetection, PreTrainedModel
    with MODEL_LOCK:
        for n, f in _PTM_ORIG.items():
            if PreTrainedModel.__dict__.get(n) is not f:
                setattr(PreTrainedModel, n, f)
        m = DFineForObjectDetection.from_pretrained(str(WDIRS[k]), device_map=None)
        meta = _meta_names(m)
        if meta:
            meta = _repair_meta(m, meta)
        info = check_model(m)
    return m.eval(), info


# ── Проверка: векторное обращение видов == invert_xyxy прод-пути (до GPU-работы) ──────────────────────────────────
def inv_boxes(xyxy, rot, hflip, size):
    """Векторный аналог invert_xyxy: xyxy (N,4) torch в пикселях повёрнутого тайла → исходный тайл."""
    x0, y0, x1, y1 = xyxy.unbind(1)
    if hflip:
        x0, x1 = size - x1, size - x0
    rot %= 4
    if rot == 0:
        return torch.stack([x0, y0, x1, y1], 1)
    if rot == 1:
        return torch.stack([size - y1, x0, size - y0, x1], 1)
    if rot == 2:
        return torch.stack([size - x1, size - y1, size - x0, size - y0], 1)
    return torch.stack([y0, size - x1, y1, size - x0], 1)
_g = torch.Generator().manual_seed(0)
_a = torch.rand(500, 2, generator=_g, dtype=torch.float64) * 600; _wh = torch.rand(500, 2, generator=_g, dtype=torch.float64) * 40
_bx = torch.cat([_a, _a + _wh], 1)
for _r, _f in VIEWS:
    _ref = invert_xyxy(_bx.numpy(), _r, _f, 640.0)
    assert np.allclose(inv_boxes(_bx, _r, _f, 640.0).numpy(), _ref, atol=1e-9), f"inv_boxes != invert_xyxy для вида {(_r, _f)}"
log("обращение видов совпадает с invert_xyxy для всех 8 видов")

@torch.no_grad()
def infer_region_tta(model, inv, dev):
    """Окна сцены × 8 видов. Возвращает {V0, T8N, T8W: [features]} в формате исходного OOF."""
    from torchvision.ops import batched_nms
    from shapely.geometry import Polygon, mapping
    from shapely.ops import transform as st
    from pyproj import Transformer
    T, S = CFG["tile"], CFG["tile"] - CFG["overlap"]; out = {"V0": [], "T8N": [], "T8W": []}; stat = {"windows": 0}
    for key, size in inv["tifs"]:
        sc = Scene(inv, key, size)
        xs = list(range(0, max(1, sc.W - CFG["overlap"]), S)) or [0]; ys = list(range(0, max(1, sc.H - CFG["overlap"]), S)) or [0]
        wins = [(x, y) for y in ys for x in xs]
        if CFG["max_windows"]: wins = wins[:CFG["max_windows"]]
        acc = [[[], [], []] for _ in VIEWS]; t_r = time.time()
        for b in range(0, len(wins), CFG["bs"]):
            chunk, imgs = [], []
            for x, y in wins[b:b + CFG["bs"]]:
                im = sc.read(x, y, T)
                if (im.max(2) == 0).mean() > 0.95: continue
                chunk.append((x, y)); imgs.append(im)
            if not imgs: continue
            px = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2).astype(np.float32) / 255).to(dev)
            for vi, (rot, hflip) in enumerate(VIEWS):
                t = torch.rot90(px, rot, dims=(2, 3)) if rot else px
                if hflip: t = t.flip(3)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    o = model(pixel_values=t)
                prob = o.logits.float().sigmoid(); bx = o.pred_boxes.float()
                Q, C = prob.shape[1:]
                sc_, idx = prob.flatten(1).topk(min(CFG["max_det"], Q * C), 1)
                for i, (x, y) in enumerate(chunk):
                    keep = sc_[i] >= CFG["min_score"]; q = idx[i][keep] // C; c = idx[i][keep] % C; bb = bx[i][q] * T
                    xyxy = torch.stack([bb[:, 0] - bb[:, 2] / 2, bb[:, 1] - bb[:, 3] / 2, bb[:, 0] + bb[:, 2] / 2, bb[:, 1] + bb[:, 3] / 2], 1)
                    xyxy = inv_boxes(xyxy.double().cpu(), rot, hflip, float(T)) + torch.tensor([x, y, x, y], dtype=torch.float64)
                    acc[vi][0].append(xyxy); acc[vi][1].append(sc_[i][keep].double().cpu()); acc[vi][2].append(c.cpu())
        stat["windows"] += len(wins)
        log(f"[infer] {inv['region']} {key.rsplit('/', 1)[-1]}: {sc.W}x{sc.H} px, окон {len(wins)}, {time.time() - t_r:.0f} с")
        def cat(vi):
            B, L, Cc = acc[vi]
            if not B: return torch.zeros((0, 4), dtype=torch.float64), torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.long)
            return torch.cat(B), torch.cat(L), torch.cat(Cc)
        res = {}
        # V0: оба вида исходного OOF вместе, один NMS (как infer_region ядра D-FINE-2)
        B = [cat(vi) for vi in V0_VIEWS]
        Bt, Lt, Ct = (torch.cat([p[j] for p in B]) for j in range(3))
        k_ = batched_nms(Bt, Lt, Ct, CFG["nms_iou"]) if len(Lt) else torch.zeros(0, dtype=torch.long)
        res["V0"] = (Bt[k_].numpy(), Lt[k_].numpy(), Ct[k_].numpy().astype(int))
        # T8*: NMS по окнам внутри вида (orsp_tta._nms_per_model), затем слияние видов (orsp_tta.fuse_views)
        parts = []
        for vi in range(len(VIEWS)):
            Bt, Lt, Ct = cat(vi)
            k_ = batched_nms(Bt, Lt, Ct, CFG["nms_iou"]) if len(Lt) else torch.zeros(0, dtype=torch.long)
            parts.append((Bt[k_].numpy(), Lt[k_].numpy(), Ct[k_].numpy().astype(int)))
        res["T8N"] = fuse_views(parts, "nms", CFG["wbf_iou"], CFG["nms_iou"])
        res["T8W"] = fuse_views(parts, "wbf", CFG["wbf_iou"], CFG["nms_iou"])
        to3857 = Transformer.from_crs(sc.crs, "EPSG:3857", always_xy=True).transform; ang = np.linspace(0, 2 * np.pi, 33)[:-1]
        for v, (BB, LL, CC) in res.items():
            for (x0, y0, x1, y1), s, c in zip(BB, LL, CC):
                cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, max((x1 - x0) / 2, .5), max((y1 - y0) / 2, .5)
                ring = [sc.T * (cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in ang]
                out[v].append({"type": "Feature", "geometry": mapping(st(to3857, Polygon(ring))),
                               "properties": {"class_name": CLASSES[int(c)], "confidence": round(float(s), 5), "region_name": inv["region"],
                                              "crs": "EPSG:3857", "markup_type": "sp_or", "original_crs": sc.crs, "group": CFG["group"],
                                              "variant": v, "fold": FOLD[inv["region"]],
                                              "tif": TIF_INFO[key][1] if key in TIF_INFO else key.rsplit("/", 1)[-1]}})
        sc.close()
    return out, stat

# ── Оркестрация: фолды по 2 GPU в потоках; каждый фолд — свои тестовые участки ──────────────────────────────────
folds_run = [int(f) for f in CFG["folds"].split(",")]
miss = [k for k in folds_run if k not in WDIRS]
if miss: raise SystemExit(f"нет весов фолдов {miss}: подключите вывод ядер {WEIGHTS_HINT}")
REPORT, lock = {"cfg": CFG, "views": VIEWS, "weights": {k: str(v) for k, v in WDIRS.items()}, "folds": {}}, threading.Lock()
queue = sorted(folds_run, key=lambda k: -sum(s for i in INV if FOLD[i["region"]] == k for _, s in i["tifs"]))
def gpu_worker(gi, dev):
    while True:
        with lock:
            if not queue: return
            k = queue.pop(0)
        tag = f"[gpu{gi} f{k}]"
        try:
            model, minfo = load_fold(k); model = model.to(dev)
            test = [i for i in INV if FOLD[i["region"]] == k]
            log(tag, f"веса {WDIRS[k]}, тестовых участков {len(test)}")
            ti = time.time(); feats = {"V0": [], "T8N": [], "T8W": []}; rep = {"model_check": minfo, "windows": 0, "errors": []}
            for inv in test:
                if time.time() > DEADLINE:
                    rep["errors"].append(f"deadline:{inv['region']}"); log(tag, "потолок времени, пропуск", inv["region"]); continue
                try:
                    f, s = infer_region_tta(model, inv, dev); rep["windows"] += s["windows"]
                    for v in feats: feats[v] += f[v]
                except Exception as e:
                    rep["errors"].append(f"{inv['region']}: {e!r}"[:300]); log(tag, "ошибка", inv["region"], repr(e)[:200])
            for v, fs in feats.items():
                json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
                           "features": fs}, open(OUT / f"oof_{v}_f{k}.geojson", "w", encoding="utf-8"), ensure_ascii=False)
            rep.update({"test_regions": [i["region"] for i in test], "n_det": {v: len(fs) for v, fs in feats.items()},
                        "infer_sec": round(time.time() - ti)})
            if dev.type == "cuda": rep["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
            log(tag, "готово", json.dumps(rep, ensure_ascii=False)[:600])
            del model; torch.cuda.empty_cache() if dev.type == "cuda" else None
        except Exception as e:
            import traceback; rep = {"error": repr(e), "tb": traceback.format_exc()[-2000:]}; log(tag, "ОШИБКА", repr(e))
        with lock:
            REPORT["folds"][k] = rep
            json.dump(REPORT, open(OUT / "tta_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
ths = [threading.Thread(target=gpu_worker, args=(gi, d)) for gi, d in enumerate(DEVS)]
for th in ths: th.start()
for th in ths: th.join()
for v in ("V0", "T8N", "T8W"):
    allf = []
    for k in folds_run:
        p = OUT / f"oof_{v}_f{k}.geojson"
        if p.exists(): allf += json.load(open(p, encoding="utf-8"))["features"]
    json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}}, "features": allf},
              open(OUT / f"oof_{v}_spor.geojson", "w", encoding="utf-8"), ensure_ascii=False)
    REPORT.setdefault("n_det", {})[v] = len(allf)
REPORT["total_sec"] = round(time.time() - T0)
json.dump(REPORT, open(OUT / "tta_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
errs = {k: v.get("error") or v.get("errors") for k, v in REPORT["folds"].items() if v.get("error") or v.get("errors")}
log("ГОТОВО", REPORT["n_det"], "ошибки:", errs)
if errs: raise SystemExit(f"фолды с ошибкой: {list(errs)}")

