# Сборка ядра dmitrijkv/koz3-bench-t4: src/bench_main.py + вшитые kernel/bench_tile.py (байт-в-байт копия бота 8) и hf_configs.
# Kaggle-скрипт заливает только code_file, поэтому всё вшивается. Шаблон метаданных — ../nolidar_verifier/build.py.
import json, pathlib, base64, hashlib
H = pathlib.Path(__file__).parent; K = H / "kernel"
ORIG = pathlib.Path("/workspace/koz3_shared/bot-8/bench/bench_tile.py")
b = (K / "bench_tile.py").read_bytes()
assert b == ORIG.read_bytes(), "kernel/bench_tile.py отличается от файла бота 8"
sha = hashlib.sha256(b).hexdigest()
cfgs, meta = {}, {}
for name, repo, rev in (("dfine_l", "ustc-community/dfine-large-coco", "aa8f8211545e4cff5adf52ea512f9ac74a9824ec"),
                        ("rtdetrv2_r50", "PekingU/rtdetr_v2_r50vd", "282494075698cab9faa1096ae26856890030c817")):
    t = (K / "hf_configs" / name / "config.json").read_bytes(); json.loads(t)
    cfgs[name] = base64.b64encode(t).decode(); meta[name] = {"repo": repo, "revision": rev, "sha256": hashlib.sha256(t).hexdigest()}
src = (H / "src/bench_main.py").read_text(encoding="utf-8")
head, rest = src.split("import os, sys, subprocess", 1)
emb = (f"BENCH_SHA256 = {sha!r}\nBENCH_B64 = {base64.b64encode(b).decode()!r}\n"
       f"HF_CONFIGS_B64 = {json.dumps(cfgs)}\nHF_CONFIG_META = {json.dumps(meta)}\n")
code = head + emb + "import os, sys, subprocess" + rest
compile(code, "koz3-bench-t4.py", "exec")
(K / "koz3-bench-t4.py").write_text(code, encoding="utf-8")
md = {"id": "dmitrijkv/koz3-bench-t4", "title": "koz3-bench-t4", "code_file": "koz3-bench-t4.py",
      "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
      "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
      "kernel_sources": [], "model_sources": [],
      "docker_image": "gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"}
json.dump(md, open(K / "kernel-metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("kernel bytes", len(code.encode()), "bench_tile sha256", sha)
