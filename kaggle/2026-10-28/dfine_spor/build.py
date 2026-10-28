# Сборка двух ядер из src/dfine_main.py: smoke (DFINE_MODE=smoke, 10 ит.) и full. Фолды — те же, что у верификатора.
import json, pathlib
H = pathlib.Path(__file__).parent
F = json.load(open(H.parent / "nolidar_verifier/folds_family_k5.json", encoding="utf-8"))
folds = {"family_of": F["family_of"], "folds": [{k: f[k] for k in ("fold", "test_families", "test_regions")} for f in F["folds"]]}
src = (H / "src/dfine_main.py").read_text(encoding="utf-8")
# Тренер бота 5 (адаптер, вендорено из bot-5/repo@0256251) встраивается строкой: Kaggle загружает только файл кода.
bot5 = (H / "src/dfine_train_bot5.py").read_text(encoding="utf-8")
assert "'''" not in bot5 and not bot5.endswith("\\"), "dfine_train_bot5.py нельзя встроить в r'''…'''"
import hashlib; print("dfine_train_bot5.py sha256", hashlib.sha256(bot5.encode("utf-8")).hexdigest())
head, rest = src.split("import json, time, re", 1)
# Бюджет недели (решение 22:26): full = 3 фолда 4,2,1 (2 GPU: 4 ‖ 2, затем 1), потолок 5 ч; full_rest = 0,3 на следующую неделю.
# 6000 ит. на фолд: 0,92 с/ит (смоук v2, T4) → 1,53 ч обучения; см. DFINE.md §11.
VARIANTS = [("smoke", "smoke", "koz3-dfine-spor-smoke", {}),
            ("full", "full", "koz3-dfine-spor-full", {"FOLDS_DEFAULT": "4,2,1", "DEADLINE_H_DEFAULT": 5.0, "FULL_ITERS_DEFAULT": 6000}),
            ("full_rest", "full", "koz3-dfine-spor-full-rest", {"FOLDS_DEFAULT": "0,3", "DEADLINE_H_DEFAULT": 3.5, "FULL_ITERS_DEFAULT": 6000})]
for d, mode, name, ov in VARIANTS:
    emb = f"DFINE_MODE_DEFAULT = {mode!r}\n" + "".join(f"{k} = {v!r}\n" for k, v in ov.items())
    emb += "FOLDS = __import__('json').loads(r'''" + json.dumps(folds, ensure_ascii=False) + "''')\n"
    emb += "BOT5_TRAINER_SRC = r'''" + bot5 + "'''\n"
    (H / d).mkdir(exist_ok=True)
    (H / d / f"{name}.py").write_text(head + emb + "import json, time, re" + rest, encoding="utf-8")
    md = {"id": f"dmitrijkv/{name}", "title": name, "code_file": f"{name}.py", "language": "python", "kernel_type": "script",
          "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
          "dataset_sources": ["dmitrijkv/koz3-s3-cred"], "competition_sources": [], "kernel_sources": [], "model_sources": [],
          "docker_image": "gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"}
    json.dump(md, open(H / d / "kernel-metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(d, name, len((H / d / f"{name}.py").read_bytes()), "байт", hashlib.sha256((H / d / f"{name}.py").read_bytes()).hexdigest()[:16])
