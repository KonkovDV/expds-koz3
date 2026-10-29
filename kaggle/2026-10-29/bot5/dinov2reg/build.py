"""bot-9: Kaggle-кернел для bot-5 FOR_BOT9_DINOV2REG, ЧАСТЬ A (признаки виньеток городищ DINOv2-reg B и L).
Исходники bot-5 читаются через `git show a0ac2ef:<path>` (только чтение), встраиваются gzip+base64 с sha256.
Общие блоки (common.py, s3.py) — из ../gorloc/tpl. Запуск: python3 build.py"""
import json, subprocess, hashlib, gzip, base64, pathlib
H = pathlib.Path(__file__).parent
REPO, COMMIT = "/workspace/koz3_shared/bot-5/repo", "a0ac2ef"
FULL = subprocess.run(["git", "-C", REPO, "rev-parse", COMMIT], capture_output=True, text=True, check=True).stdout.strip()
FILES = ["scripts/gor_patch/extract_feats.py", "scripts/det_train_hf.py"]
USER, IMG = "dmitrijkv", "gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"
MODELS = {"B": ["facebook/dinov2-with-registers-base", "a1d738ccfa7ae170945f210395d99dde8adb1805"],
          "L": ["facebook/dinov2-with-registers-large", "e4c89a4e05589de9b3e188688a303d0f3c04d0f3"]}   # обе apache-2.0 (HF API, 29.10.2026)
SRC = {}
for f in FILES:
    b = subprocess.run(["git", "-C", REPO, "show", f"{FULL}:{f}"], capture_output=True, check=True).stdout
    assert b"satelite-scan" not in b and b"ackrome" not in b, f
    SRC[f] = (hashlib.sha256(b).hexdigest(), base64.b64encode(gzip.compress(b, 9, mtime=0)).decode())
T = lambda p: (H / p).read_text(encoding="utf-8")
man = dict(commit=FULL, sources={f: s for f, (s, _) in SRC.items()}, kernels={})
for slug, mode, dl in (("koz3-b5-dinov2reg-gorpatch", "full", 11.0),):
    d = H / "kernels" / slug; d.mkdir(parents=True, exist_ok=True)
    head = (f"# KOZ3 · bot-9 для bot-5: FOR_BOT9_DINOV2REG часть A — признаки виньеток городищ (384 м + контекст 768 м, 224 px) энкодерами\n"
            f"# DINOv2-with-registers B и L (Apache-2.0, ревизии пиннуты). Код bot-5 ветка bot5-dinov2reg @{FULL} — без изменений.\n"
            f"# Режим full: сначала smoke bot-5 (LIMIT_FILES=2, assert 4×C), затем полные B и L. ТОЛЬКО замер OOF, не сдача.\n"
            f"# Данные «Сканирование» (ackrome/satelite-scan) запрещены — не используются. Часть B (кроп-верификатор Or) здесь НЕТ: нет STACK/GT на Kaggle.\n")
    consts = (f"KERNEL_ID = {json.dumps(USER + '/' + slug)}\nCOMMIT = {json.dumps(FULL)}\nMODE_DEFAULT = {json.dumps(mode)}\n"
              f"DEADLINE_H_DEFAULT = {dl}\nMODELS = {json.dumps(MODELS)}\n")
    emb = "SOURCES = {\n" + "".join(f"    {json.dumps(f)}: ({json.dumps(s)}, {json.dumps(b)}),\n" for f, (s, b) in SRC.items()) + "}\n"
    code = head + consts + emb + T("../gorloc/tpl/common.py") + T("../gorloc/tpl/s3.py") + T("tpl/gorpatch_main.py")
    py = d / f"{slug}.py"; py.write_text(code, encoding="utf-8")
    md = {"id": f"{USER}/{slug}", "title": slug, "code_file": py.name, "language": "python", "kernel_type": "script", "is_private": True,
          "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
          "dataset_sources": [f"{USER}/koz3-s3-cred"], "competition_sources": [], "kernel_sources": [], "model_sources": [], "docker_image": IMG}
    json.dump(md, open(d / "kernel-metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    man["kernels"][slug] = dict(py_sha256=hashlib.sha256(py.read_bytes()).hexdigest(), meta_sha256=hashlib.sha256((d / "kernel-metadata.json").read_bytes()).hexdigest())
json.dump(man, open(H / "MANIFEST.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(json.dumps(man, indent=1))
