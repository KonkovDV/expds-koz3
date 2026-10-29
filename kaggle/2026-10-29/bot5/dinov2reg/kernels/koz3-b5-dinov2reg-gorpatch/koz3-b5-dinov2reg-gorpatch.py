# KOZ3 · bot-9 для bot-5: FOR_BOT9_DINOV2REG часть A — признаки виньеток городищ (384 м + контекст 768 м, 224 px) энкодерами
# DINOv2-with-registers B и L (Apache-2.0, ревизии пиннуты). Код bot-5 ветка bot5-dinov2reg @a0ac2efca20450f91606d6c3b9e7e58d5d032f5f — без изменений.
# Режим full: сначала smoke bot-5 (LIMIT_FILES=2, assert 4×C), затем полные B и L. ТОЛЬКО замер OOF, не сдача.
# Данные «Сканирование» (ackrome/satelite-scan) запрещены — не используются. Часть B (кроп-верификатор Or) здесь НЕТ: нет STACK/GT на Kaggle.
KERNEL_ID = "dmitrijkv/koz3-b5-dinov2reg-gorpatch"
COMMIT = "a0ac2efca20450f91606d6c3b9e7e58d5d032f5f"
MODE_DEFAULT = "full"
DEADLINE_H_DEFAULT = 11.0
MODELS = {"B": ["facebook/dinov2-with-registers-base", "a1d738ccfa7ae170945f210395d99dde8adb1805"], "L": ["facebook/dinov2-with-registers-large", "e4c89a4e05589de9b3e188688a303d0f3c04d0f3"]}
SOURCES = {
    "scripts/gor_patch/extract_feats.py": ("adf2e0d2785f1bd8569c55b6d4f096007461f10525ba172d67c53a7111f9c9d5", "H4sIAAAAAAAC/51ZbW8bxxH+zl+xVdJyzz4eX2Q5Nl0aUC0pMWDZhiU3KQiCOPGW5EXU3eHuKJFWBfilaQrYiNsgQd8So+2Xfkg/KLYVK5bsAPkFd3+hv6TPzPJVYpygAo53tzczOzs788zMam5ubsOPcwuWSP6VPEn+nvwn+VvyRCT/Tj5P/omhL5Ovkr+K5M/JH5MvMPQPIZNXyYFInidH6WOR3sPDfvpxcmhAwKfJYfIqfZQcpPeTl8mhWAtuhGL+wjmRHAuZ/iHZT56J4sUSXg1xVoDkNcjvJwfJy/Reel+8c/4CUab3BA8ei+QbzJT+Hs8gS++C/DiTuyySF5B0nLzmkW/oK74/TL4VS1ev39guCbkY2I22ypWsglEW1SvX1kyxpWxPBnbcaIvY31ReZNRIgX3wP4eUbwXUo9ej9CEW8sVoCZjrOV7upY8ETPG5SA6x5O/AcYR1vkgfpJ9As3vpYyuTfJo+TD/Cl+dl8faN2+v5prLj6O31xXctL7gj5Ipodnw7Lp6ntW+p2OZPH0a+B9s8gAn3yQjgf2mK9HdQ5dvkyJxYPaYW6YPkGakokqcnTJ08FbfXV8WEIKzGgFZPYNy74NN2OkwOymJpcX1RyDi0XS+neoET5eYNU0BlU6zeWFq+hi0+Ij3YHI/AtY+dxqL3cWGQNv0VvcBkrzHRvnhvBcqB51C4jkmUr0nTY7YS1KF5009E026oDd/fzDuu52+XctGW3elcElgRbAYKtiMEZU4S7rhxOxeqlhvFKoxyu8xobtiRMjt22FJ74r93PxPwhgPIOqTFY8WwESn3NYbJHg9hsf3hZmG1t5Z/fXXt6o3rZmYJT1eWhWwE3d82uo6Nj9gYU1y7unp1vb5y9drymkV7vy+Sr8kqWD1seOXmbVOUeK1605J9UD3R28VWIp2iRugGcZRv+WGdnW/8VFfbdscK+oJsOSHampuby7hbgR/Gwo9MEfXxQ15iilbH34An23HbFLG7pUwRqsjvhg01ZPC6W5BoR8IL8NEme7k+iP2w0c40Q39rNGjtuJ7j70RiwPk+v56gUZA3orilInsr6LheS1MF/SD0Pxx+XQ9tL2r64ZYKMzydFam4Dv563A6V7UTS9WLpR5C57Ya+Z7VULLPr791aXlxay5oiW8oahpFZEhVxkojclSjyO364GSG0Vd6xYzs/5cBZ4xJ58Ax2jJ7g3vTvzNejth0qJ8/Yl7c9u9OP3ImNyhqZ1VnKcICQvJneTFrAn2Ytgr2MGOFnRAYfm0GGUaLB7Nh3C5q0LdeLVBjLgknEPPKh73py+OK4oWdvqdG7vRHRXdbrTbej6nUD/py1LFhXb1s83qjR1i52Y/+K7zXdlsnPq76jOprcwS6yoevt5pDcjkijuufXCcYyt2YueBhgWUP4objue0qItyhegFs6VgnCXuA6ZPzQ+cR1BOMGMJfwJNNotiB7rJ9FStWDULFOypGrSxQF227k+l4Fc8Ky12/eWgZTETBLPgdt7DgOJUTBEuSRQyip60wAgxdYSfxCRaQLcJ5CE6mTC+GQkdkaKMWG+nGdLIp1CdWmLSe3ALhLGCZxW1bsS3iJMWGmyb2qVBasUtGCGyTPGNBfMGQ+IuRng72i9DgJSAcZt0neaEWxHcYRoajMEsJlkRX1lG2705SIOoobT/ViuQUnCuFOBLTSMCwn7gcqcxMJdA3Z4cr6B6a4+QGIkdRJFSRzuiFx061UOpdZXV68TsICyw5Duy+rBevchQVY2Dq3cJ5vhfM1o1o22SX0b+0SpC+d4CqVLhI5ZOrbwimuDCM02GYhywR6U0AVyP/jAogJOC36of2gXcBYFRqs8D1DQRPhMYKjYysJci36kc3s7tJe/kz+DBU1uEFG8wxJhZF5srIY8vK9WubRWsZRTRHF8I5GW9qmgCe4jlHOCPzR3LYFtIWRJRbPBcJ8CYr53Vjb444K/ajecTeVtA1mgj+IDawZMO21lJwfyKK/bZJX3ahVeZLaaBwqbluRe0eJX4pioVCGd8bd0GMzjog6SBRtV88aKCQVL8Y65LYpqiVTXLxQ02pBvKZpdNxASppO5MBsiDxSU09CRI5lFVXuPLCHvEQrPpgUMjK0hqZriiatg7KMCu1YSTbcYD2IMzKlFSHlYD/zWaOam9dLUkHUGmw7JUYLVnOkHyhvsEm74N3Loxzi6gp7VM02wihbG8oqk6xiTWtFYTHOeFqKQSnUGds1DjHdRIbTEQ+Z0rHwa7JGprA7O3Y/qvf6lfWwq4wRe6MHIzT65H1g2PC7nhNZHdWMgTOj99BttWMyIow9pkJ6ioHCE3SxHzDVSPpbXEEADlAxD+pgPBzokpFrCwITri2onCsLlGQHDG8v6NPBAIsBIVTYvWa651yyfTTFiuvbQZk5mrsHh+6TF8ehNQIrWdUL7hVY79G4ZYsz5H41eBSswSapja3UihzCJMph7X7gx7JnV4vkW7gTT3/w2qdXMgFETRiBa959XAzZpDYtfdCl6BrwfvoYZTBXy1SiJk95Ycfpg1G9jUc5Ll+p8H6EIpcbDG4SuGUhmxyO9Y4arHZPFk12SXp0UF45VKY5VlsNN7ZYKJ0zJvioYACrY1F9JBEWVJMEqkJOhb2Ox9win8c09M5i9avB5d+gJKuMqzPL3kY0tSb8D+FPIjt+CP1UGFTnaxbVDKJSEVm7E7TtbFlsM+qRTvgsLouxbVUnUoPvBNCdjtRkZdD9jF3a86kg0/MMn1GD+rHO+yRA5IrqIsHBhAFiDbnD9zcA2xDc9MQEcFFtJm5GsWUHiGFH/iCKTaDhZfIhrR7VV6WFhYn9CQIEdsBK3lzD9sE/KQvqp8no+xp936BKIBf5Dj3YfYozBJVuHIfeRZF3SF7K0cnNzpPks+QvyZfsoYhHan0f4MvdQQwOWjbK68/JK7mFpg5s1KwKeXMtj5QMp0Z25qfjPPVp6T1YmzrlYS/9ktvl79KHI92Xlq9gcbt7UzvAhsEmSEr6EHliF5p1QjEmyqMUMLQ5Lon34Ozv70xHQkiIJcdODGaDytEZJAO/1hRTE6IgeHOQ6Jl/ejT89IiwKTwxvw6IYumdKRmDsGCiQVwQ7f8ZFeIXLMTrTwopTMULA+5EVSDl/MTqJ8JgesMoqupTYSV61Y36dBbHnDSUQwDRwwBhaaMGA8WpjzOy+6SGUvYA9TCNASaqCEkYSrwpwsAmvIfDcgYD4B4N8tCBxmCA8xEdNL3ig55jRA7571PGb+qtv/8q/Tj9EwYJmg++PzKEPpchnqfcTVOV/GyYA4b+TzIQZPp9P308rXxw2r5IYiWsBuqyh4/epm2OcSkL08s1ZkDUiemopIWsMhnjLG/m+O39Hdqj3iWxPa3VG1VCS9oxiKX6ZrHYnClVgAVVimr6JnsB6lRczTozQh7aHIqEEYvXO5n0kLYbyu3IUTDnCEHJEsBQg+xThByv/wa+AU7MYJxCKLc/9mavfxKfiKA3QdA7QcAFWdBD9RGQLi55KqH8WZpVl1+Y4MTYKQl8tjfoHPxNrg676jTVj+Pp2BWmTA6Jox2ZSd9A5IWFQRWsURSrYgSFARFV0JqNx04ySdSfTTRbK3JAdtKwUA6pnruJDrBRKDf0M1bPrQY5VFD9AZqZkoGKbJkKZ1fbc7Qgi89pDfQpBWuhrE27YgMpL4kNZIDNmbK4hNeHTuisWyESxQ9Ymf588kAZuD3VqWPOrooqmpeLej5Bo4C1o4aP2qHV9buRbkp7RpX7TmPYqfN9HV1yB3Feb7uOo7w6mu1YASJ1xEvDagRdaczUhp1oWK9oHRpgqfqMsH6VDjLKNW2SgoF5tXIGHxGcPu7kY/mn+gg6fZzep2J9qpY44ENckH9MtXxmxpZQcvI3y4KX7s3w6A/gdr8pcELu9cmZKI56rFDo7+DF74jcZUR4gEc/dCIhxyfYp63QRRx2+yf7Bz3Jaeq3xuff8xfO5fRph17P/snOZBQnHKVDbeGQEQf3MNQpZjDWG48ZOmp7dPLKcStzwKpcEdgqJx/4nuMH47SLN7o07dSqzjS06AZJbUSneegwYugNjttAxKpWBZfJBwqV4QEfnXvziV8TGnR7FW3DSpe2wg89OipqdE2qyiq4jNNmXJkokbHRcDnIo+6b3PFUziqeH4gIqFZCh13dnThCycWFslVo7kU1sdt0zxbRfHeog+ZGfk9wJ86d3S5+ylapuSf4eFVFgglpzaALo4ho9WE2HeCECDqg92jo1u21xXeX62vL11YMK+zWkULAk6eWiudf/VUWZup0o/ZE7w2HhiacQMTPRalAYFMQfPRAg/Q6VnYMGFh5hKrxDp0m3Li9vqf/kbO7vvju3u5cPVZRPCeGhz66ipub26P/8UCFlQpxx3ZjU65MmJ6PKRxELy8YsT04rGDxNPRj0vkgwxTZnSw2XXlRN1R1IJTrVhgdjYzenawDgALdyLRgGdib7JOduXVZI/M/9//gVgUcAAA="),
    "scripts/det_train_hf.py": ("eb392861b0b3069d2610697cf0ee41e1f45f364bf4676a3f167321a434351ce0", "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"),
}
# ── общий блок (bot-9): пины, лог, распаковка исходников bot-5 с проверкой sha256 ──────────────────────────────
import os, sys, subprocess, importlib, json, time, hashlib, base64, gzip, glob, shutil
from pathlib import Path
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
def deps(pins):
    need = []
    for mod, pin in pins.items():
        try:
            m = importlib.import_module(mod)
            if "==" in pin and getattr(m, "__version__", "") != pin.split("==")[1]: need.append(pin)
        except Exception: need.append(pin)
    if need and not os.environ.get("B9_DRY"):
        log("pip install:", need)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", *need])
        if r.returncode: raise SystemExit(f"pip install не удался ({need}); нужен enable_internet=true")
    vers = {}
    for mod in ["numpy", "scipy", "torch", *pins]:
        try:
            out = subprocess.run([sys.executable, "-c", f"import {mod}; print({mod}.__version__)"], capture_output=True, text=True)
            vers[mod] = out.stdout.strip() or out.stderr.strip().splitlines()[-1]
        except Exception as e: vers[mod] = repr(e)
    log("версии:", json.dumps(vers))
    bad = {m: vers.get(m) for m, p in pins.items() if "==" in p and vers.get(m) != p.split("==")[1]}
    if bad and not os.environ.get("B9_DRY"): raise SystemExit(f"версии не совпали с пинами bot-5: {bad}")
    return vers
SRC_DIR = Path(os.environ.get("B9_SRC", "/tmp/b5repo"))
def unpack_sources():
    """Исходники bot-5 @COMMIT (git archive), встроены gzip+base64; sha256 сверяется с git-блобом при сборке."""
    for rel, (sha, b64) in SOURCES.items():
        data = gzip.decompress(base64.b64decode(b64))
        assert hashlib.sha256(data).hexdigest() == sha, f"sha256 не совпал: {rel}"
        p = SRC_DIR / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    log(f"исходники bot-5 @{COMMIT}: {len(SOURCES)} файлов → {SRC_DIR}")
def run_step(name, args, env_extra, logfile, timeout_s):
    """Запуск скрипта bot-5 подпроцессом; stdout+stderr → консоль и logfile. Возврат: (код, секунды)."""
    env_extra = {k: str(v) for k, v in env_extra.items()}; env = dict(os.environ); env.update(env_extra); t = time.time()
    log(f"▶ {name}: {' '.join(map(str, args))} | env {json.dumps(env_extra, ensure_ascii=False)} | timeout {timeout_s:.0f} s")
    import threading
    with open(logfile, "w", encoding="utf-8") as fh:
        p = subprocess.Popen([sys.executable, "-u", *map(str, args)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        killed = []; tm = threading.Timer(max(1.0, timeout_s), lambda: (killed.append(1), p.kill())); tm.start()
        for line in p.stdout:
            fh.write(line); fh.flush(); print("   ", line, end="", flush=True)
        rc = p.wait(); tm.cancel()
        if killed: rc = "timeout"
    dt = time.time() - t; log(f"■ {name}: rc={rc}, {dt:.0f} s"); return rc, dt
def sha256_file(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()
# ── S3 (как в остальных кернелах bot-9): Kaggle Secrets KOZ3_S3_* → env → /kaggle/input/**/cred.json ─────────────
# Значения ключей никогда не печатаются и не пишутся в /kaggle/working.
BUCKET, ENDPOINT = "train-expds-3", os.environ.get("B9_S3_ENDPOINT", "https://s3.ru-7.storage.selcloud.ru")
SECRET_ID, SECRET_KEY = "KOZ3_S3_ACCESS_KEY_ID", "KOZ3_S3_SECRET_ACCESS_KEY"
def _cred_from_inputs():
    for _p in sorted(glob.glob("/kaggle/input/**/cred.json", recursive=True)):
        try:
            with open(_p, encoding="utf-8") as _fh: _c = json.load(_fh)
        except Exception as _e:
            log("cred.json не читается:", type(_e).__name__); continue
        if not isinstance(_c, dict): continue
        _k = _c.get("id") or _c.get("aws_access_key_id"); _s = _c.get("key") or _c.get("aws_secret_access_key")
        if _k and _s:
            log("ключи S3 из приватного датасета"); return _k, _s
    return None, None
def s3_client():
    import boto3
    from botocore.config import Config
    kid = sec = None
    try:
        from kaggle_secrets import UserSecretsClient
        u = UserSecretsClient(); kid, sec = u.get_secret(SECRET_ID), u.get_secret(SECRET_KEY)
    except Exception as e:
        log("Kaggle Secrets недоступны:", type(e).__name__, "— пробую env / приватный датасет")
    if not kid or not sec:
        kid, sec = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not kid or not sec:
        kid, sec = _cred_from_inputs()
    if not kid or not sec:
        raise SystemExit(f"нет ключей S3: Kaggle Secrets {SECRET_ID}/{SECRET_KEY} или датасет dmitrijkv/koz3-s3-cred (cred.json)")
    return boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=kid, aws_secret_access_key=sec, region_name="ru-7",
                        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=32))
def list_prefix(cli, prefix, delim=None):
    out, tok = [], None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=prefix)
        if delim: kw["Delimiter"] = delim
        if tok: kw["ContinuationToken"] = tok
        r = cli.list_objects_v2(**kw)
        out += [p["Prefix"] for p in r.get("CommonPrefixes", [])] if delim else [(o["Key"], o["Size"]) for o in r.get("Contents", [])]
        if not r.get("IsTruncated"): return out
        tok = r["NextContinuationToken"]
def fetch(cli, key, size, root):
    from boto3.s3.transfer import TransferConfig
    dst = Path(root) / key
    if dst.exists() and dst.stat().st_size == size: return dst
    dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_name(dst.name + ".part")
    for a in range(5):
        try:
            cli.download_file(BUCKET, key, str(tmp), Config=TransferConfig(max_concurrency=4)); tmp.rename(dst)
            assert dst.stat().st_size == size, f"размер {key}"; return dst
        except Exception as e: log("повтор", a + 1, key, type(e).__name__); time.sleep(5 * (a + 1))
    raise RuntimeError("не скачан " + key)
# ── main: признаки виньеток extract_feats.py (bot-5 @COMMIT) DINOv2-reg B и L на GPU — часть A FOR_BOT9_DINOV2REG ─────
PINS = {"transformers": "transformers==5.21.0", "rasterio": "rasterio==1.5.2", "pyproj": "pyproj==3.8.0",
        "shapely": "shapely==2.2.0", "boto3": "boto3==1.40.61"}   # первые 4 — kaggle/requirements_gor.txt bot-5 (его ноутбук ставит их же)
VERS = deps(PINS)
import numpy as np
MODE = os.environ.get("B9_MODE", MODE_DEFAULT); assert MODE in ("smoke", "full"), MODE
DEADLINE = T0 + float(os.environ.get("B9_DEADLINE_H", DEADLINE_H_DEFAULT)) * 3600
WORK = Path(os.environ.get("B9_WORK", "/kaggle/working")); GP = WORK / "gp"; GP.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get("B9_DRY_DATA") or "/tmp/data")
SUMMARY = dict(kernel=KERNEL_ID, commit=COMMIT, mode=MODE, versions=VERS, hf=MODELS, steps={})
def dump_summary(): json.dump(SUMMARY, open(WORK / "run_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
def remaining(): return DEADLINE - time.time()
log("MODE", MODE, "WORK", WORK, "DATA", DATA)

import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
log("GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if DEV == "cuda" else "нет")
if DEV != "cuda" and not os.environ.get("B9_ALLOW_CPU"): raise SystemExit("GPU нет (второй одновременный GPU-кернел?). Останов.")
unpack_sources()
EXTRACT = SRC_DIR / "scripts/gor_patch/extract_feats.py"

# веса HF (ранний фейл, до данных); dry: B9_MODEL_DIR — один крошечный каталог для обоих
CK, HID = {}, {}
for t, (rid, rev) in MODELS.items():
    if os.environ.get("B9_MODEL_DIR"): CK[t] = os.environ["B9_MODEL_DIR"]
    else:
        from huggingface_hub import snapshot_download
        try: CK[t] = snapshot_download(rid, revision=rev, allow_patterns=["*.json", "*.safetensors"])
        except Exception as e: raise SystemExit(f"веса {rid}@{rev[:8]} не скачались: {e!r}")
    cfg = json.load(open(Path(CK[t]) / "config.json")); HID[t] = cfg["hidden_size"]
    log(f"веса {t}: {rid}@{rev[:8]} (Apache-2.0) hidden {HID[t]} регистров {cfg.get('num_register_tokens')} → {CK[t]}")

# данные: все SpOr *.tif + UTM.json этих участков (extract_feats глобит {DATA}/*/*SpOr*/*.tif*)
if not os.environ.get("B9_DRY_DATA"):
    import concurrent.futures as cf
    CLI = s3_client()
    try: CLI.list_objects_v2(Bucket=BUCKET, MaxKeys=1)
    except Exception as e: raise SystemExit(f"S3 недоступен ({type(e).__name__}): проверьте секреты/интернет (детали скрыты).")
    regs = [p.rstrip("/") for p in list_prefix(CLI, "", "/")]
    def inv(reg):
        tif, utm = [], []
        for k, s in list_prefix(CLI, reg + "/"):
            parts = k.split("/")
            if parts[-1] == "UTM.json" and len(parts) == 2: utm.append((k, s))
            elif len(parts) == 3 and "SpOr" in parts[1] and parts[-1].lower().endswith((".tif", ".tiff")): tif.append((k, s))
        return tif, (utm if tif else [])
    with cf.ThreadPoolExecutor(16) as ex: INV = list(ex.map(inv, regs))
    TIF = sorted(x for t, _ in INV for x in t); UTM = {k.split("/")[0]: (k, s) for _, u in INV for k, s in u}
    log(f"S3: участков {len(regs)}, SpOr tif {len(TIF)}, UTM.json {len(UTM)}, {sum(s for _, s in TIF) / 1e9:.2f} ГБ")
    if MODE == "full": assert len(TIF) == 175 and len(UTM) == 132, f"ожидалось 175 tif / 132 UTM (bot-9/s3/listing.txt), есть {len(TIF)}/{len(UTM)}"
    if MODE == "smoke": TIF = TIF[:2]   # как LIMIT_FILES=2 у bot-5: первые 2 файла по сортировке путей
    KEYS = TIF + sorted({UTM[k.split("/")[0]] for k, _ in TIF})
    t = time.time()
    with cf.ThreadPoolExecutor(8) as ex: list(ex.map(lambda ks: fetch(CLI, ks[0], ks[1], DATA), KEYS))
    SUMMARY["download"] = dict(tif=len(TIF), gb=round(sum(s for _, s in KEYS) / 1e9, 3), s=round(time.time() - t)); dump_summary()
    log("данные скачаны", SUMMARY["download"])

BASE_ENV = dict(DATA=DATA, OUT=GP, DEVICE=DEV, THREADS=os.environ.get("B9_THREADS", "4"), REVISION="", HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)
def check(t, test):
    sfx = "_test" if test else ""
    F = np.load(GP / f"feats_reg{t}{sfx}.npz")["F"]; meta = json.load(open(GP / f"meta_reg{t}{sfx}.json", encoding="utf-8"))
    info = dict(shape=list(F.shape), finite=bool(np.isfinite(F.astype(np.float32)).all()), meta=len(meta), regions=len({m["reg"] for m in meta}))
    log(f"проверка feats_reg{t}{sfx}:", json.dumps(info))
    assert info["shape"][1] == 4 * HID[t] and info["finite"] and info["meta"] == info["shape"][0], \
        f"reg{t}: ожидается 4×{HID[t]} и всё конечное — СТОП (assert ноутбука bot-5)"
    return info

# smoke bot-5 (первым): LIMIT_FILES=2 для B и L, размерность 4×C
for t in MODELS:
    rc, dt = run_step(f"smoke_{t}", [EXTRACT], dict(BASE_ENV, MODEL=CK[t], TAG=f"_reg{t}", LIMIT_FILES=2), WORK / f"smoke_reg{t}.log", min(remaining(), 1800))
    SUMMARY["steps"][f"smoke_{t}"] = dict(rc=rc, s=round(dt)); dump_summary()
    if rc != 0: raise SystemExit(f"smoke reg{t} упал — СТОП (хвост: smoke_reg{t}.log)")
    SUMMARY["steps"][f"smoke_{t}"]["check"] = check(t, True); dump_summary()
for p in GP.glob("*_test.*"): p.unlink()
if MODE == "smoke": log("SMOKE OK"); sys.exit(0)

# полный: B, затем L (порядок ноутбука bot-5); extract_feats сам сохраняет каждые 20 файлов
for t in MODELS:
    if remaining() < 600: SUMMARY["steps"][t] = dict(rc="skipped_deadline"); dump_summary(); continue
    rc, dt = run_step(f"full_{t}", [EXTRACT], dict(BASE_ENV, MODEL=CK[t], TAG=f"_reg{t}", LIMIT_FILES=0), WORK / f"extract_reg{t}.log", remaining() - 300)
    SUMMARY["steps"][t] = dict(rc=rc, s=round(dt)); dump_summary()
    if rc == 0:
        SUMMARY["steps"][t]["check"] = c = check(t, False)
        if not os.environ.get("B9_DRY_DATA") and c["meta"] != 3982:
            log("ВНИМАНИЕ: виньеток", c["meta"], "≠ 3982 у DINOv2-S (bot-5/analysis/gor_patch/extract_feats.log) — сообщить bot-5")
        dump_summary()
if not os.environ.get("B9_DRY_DATA"): shutil.rmtree(DATA, ignore_errors=True)
import tarfile
with tarfile.open(WORK / "gor_patch_reg.tgz", "w:gz") as tf:
    for p in sorted(GP.glob("feats_reg*.npz")) + sorted(GP.glob("meta_reg*.json")): tf.add(p, arcname=p.name)
ok = all(SUMMARY["steps"].get(t, {}).get("rc") == 0 for t in MODELS)
log("ГОТОВО" if ok else "ЧАСТИЧНО", json.dumps({k: v.get("rc") for k, v in SUMMARY["steps"].items()}), "| выход: /kaggle/working/gor_patch_reg.tgz")
if not ok: sys.exit(1)
