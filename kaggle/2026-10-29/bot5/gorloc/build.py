"""bot-9: сборка Kaggle-кернелов для задачи bot-5 «плотный локализатор городищ» (FOR_BOT9_GORLOC.md, ветка bot5-gor-dense-loc @6984c67).
Исходники берутся ТОЛЬКО ЧТЕНИЕМ из репо bot-5: `git -C <repo> show <commit>:<path>`, встраиваются gzip+base64 с sha256.
Запуск: python3 build.py  → kernels/<slug>/{<slug>.py, kernel-metadata.json} + MANIFEST.json"""
import json, subprocess, hashlib, gzip, base64, pathlib
H = pathlib.Path(__file__).parent
REPO, COMMIT = "/workspace/koz3_shared/bot-5/repo", "6984c67"
FULL = subprocess.run(["git", "-C", REPO, "rev-parse", COMMIT], capture_output=True, text=True, check=True).stdout.strip()
FILES = ["scripts/gor_dense/dense_feats.py", "scripts/gor_dense/dense_probe.py", "kaggle/gor_dense_inputs.json",
         "expds/__init__.py", "expds/constants.py", "expds/naming.py", "expds/metric.py"]
USER, IMG = "dmitrijkv", "gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"
SRC = {}
for f in FILES:
    b = subprocess.run(["git", "-C", REPO, "show", f"{FULL}:{f}"], capture_output=True, check=True).stdout
    assert b"satelite-scan" not in b and b"ackrome" not in b, f"ссылка на запрещённые данные «Сканирование» в {f}"
    SRC[f] = (hashlib.sha256(b).hexdigest(), base64.b64encode(gzip.compress(b, 9, mtime=0)).decode())
tpl = lambda n: (H / "tpl" / n).read_text(encoding="utf-8")
FEATS_ID = f"{USER}/koz3-b5-gorloc-feats"
LAY_ALL = ["fam10", "seed0", "seed1", "seed2", "seed3", "seed4", "LOFO"]
KERNELS = [
    dict(slug="koz3-b5-gorloc-smoke", kind="feats", gpu=True, consts=dict(MODE_DEFAULT="smoke", DEADLINE_H_DEFAULT=1.0, EXPECT_DIM=2048),
         doc="SMOKE (необязательный, ~0,3 ч T4): только 185_Изборск, GSDS=1,2, проверка F.shape[1]==2048 и конечности."),
    dict(slug="koz3-b5-gorloc-feats", kind="feats", gpu=True, consts=dict(MODE_DEFAULT="full", DEADLINE_H_DEFAULT=11.0, EXPECT_DIM=2048),
         doc="ПОЛНЫЙ (GPU T4): smoke-гейт bot-5 → признаки _L24 (2/4 м/пкс) и _L12 (1/2 м/пкс) по 175 сценам SpOr; выход /kaggle/working/gd/*.npz для кернелов пробы."),
    dict(slug="koz3-b5-gorloc-probe-l24", kind="probe", gpu=False, consts=dict(TAG_DEFAULT="_L24", LAYOUTS_DEFAULT=LAY_ALL, DEADLINE_H_DEFAULT=11.5, OUT_SUFFIX="all"),
         doc="ПРОБА _L24 (CPU, без GPU): все раскрои fam10, seed0–4, LOFO."),
    dict(slug="koz3-b5-gorloc-probe-l12a", kind="probe", gpu=False, consts=dict(TAG_DEFAULT="_L12", LAYOUTS_DEFAULT=LAY_ALL[:6], DEADLINE_H_DEFAULT=11.5, OUT_SUFFIX="fam10seeds"),
         doc="ПРОБА _L12 (CPU): fam10 + seed0–4."),
    dict(slug="koz3-b5-gorloc-probe-l12b", kind="probe", gpu=False, consts=dict(TAG_DEFAULT="_L12", LAYOUTS_DEFAULT=["LOFO"], DEADLINE_H_DEFAULT=11.5, OUT_SUFFIX="lofo"),
         doc="ПРОБА _L12 (CPU): LOFO (103 фолда, 58 с отложенными сценами SpOr)."),
]
man = dict(commit=FULL, sources={f: s for f, (s, _) in SRC.items()}, kernels={})
for k in KERNELS:
    d = H / "kernels" / k["slug"]; d.mkdir(parents=True, exist_ok=True)
    head = (f"# KOZ3 · bot-9 для bot-5: плотный локализатор городищ на спутнике SpOr (DINOv2-L-reg, линейная проба, PU-ignore,\n"
            f"# pu_edge, приор центра). Задание: bot-5/FOR_BOT9_GORLOC.md; код bot-5 ветка bot5-gor-dense-loc @{FULL} — без изменений.\n"
            f"# {k['doc']}\n# ТОЛЬКО честный замер, НЕ сдача. Данные «Сканирование» (ackrome/satelite-scan) запрещены — не используются, Input не подключать.\n"
            f"# Сборка: bot-9/kaggle/bot5/gorloc/build.py. Режим/параметры — константы ниже или env B9_*.\n")
    consts = f"KERNEL_ID = {json.dumps(USER + '/' + k['slug'])}\nCOMMIT = {json.dumps(FULL)}\nFEATS_KERNEL = {json.dumps(FEATS_ID)}\n" + \
             "".join(f"{a} = {json.dumps(v, ensure_ascii=False)}\n" for a, v in k["consts"].items())
    need = FILES if k["kind"] == "feats" else [f for f in FILES if f != "scripts/gor_dense/dense_feats.py"]
    emb = "SOURCES = {\n" + "".join(f"    {json.dumps(f, ensure_ascii=False)}: ({json.dumps(SRC[f][0])}, {json.dumps(SRC[f][1])}),\n" for f in need) + "}\n"
    code = head + consts + emb + tpl("common.py") + (tpl("s3.py") if k["kind"] == "feats" else "") + tpl(f"{k['kind']}_main.py")
    py = d / f"{k['slug']}.py"; py.write_text(code, encoding="utf-8")
    md = {"id": f"{USER}/{k['slug']}", "title": k["slug"], "code_file": py.name, "language": "python", "kernel_type": "script",
          "is_private": True, "enable_gpu": k["gpu"], "enable_tpu": False, "enable_internet": True}
    if k["gpu"]: md["machine_shape"] = "NvidiaTeslaT4"
    md.update({"dataset_sources": [f"{USER}/koz3-s3-cred"] if k["kind"] == "feats" else [], "competition_sources": [],
               "kernel_sources": [FEATS_ID] if k["kind"] == "probe" else [], "model_sources": [], "docker_image": IMG})
    json.dump(md, open(d / "kernel-metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    man["kernels"][k["slug"]] = dict(py_sha256=hashlib.sha256(py.read_bytes()).hexdigest(),
                                     meta_sha256=hashlib.sha256((d / "kernel-metadata.json").read_bytes()).hexdigest(), bytes=py.stat().st_size)
json.dump(man, open(H / "MANIFEST.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(man["kernels"], indent=1))
