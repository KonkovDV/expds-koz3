# Сборка двух ядер из src/dfine_main.py: smoke (DFINE_MODE=smoke, 10 ит.) и full. Фолды — те же, что у верификатора.
import json, pathlib
H = pathlib.Path(__file__).parent
F = json.load(open(H.parent / "nolidar_verifier/folds_family_k5.json", encoding="utf-8"))
folds = {"family_of": F["family_of"], "folds": [{k: f[k] for k in ("fold", "test_families", "test_regions")} for f in F["folds"]]}
src = (H / "src/dfine_main.py").read_text(encoding="utf-8")
head, rest = src.split("import json, time, re", 1)
for mode in ("smoke", "full"):
    emb = f"DFINE_MODE_DEFAULT = {mode!r}\nFOLDS = __import__('json').loads(r'''" + json.dumps(folds, ensure_ascii=False) + "''')\n"
    name = f"koz3-dfine-spor-{mode}"
    (H / mode).mkdir(exist_ok=True)
    (H / mode / f"{name}.py").write_text(head + emb + "import json, time, re" + rest, encoding="utf-8")
    md = {"id": f"dmitrijkv/{name}", "title": name, "code_file": f"{name}.py", "language": "python", "kernel_type": "script",
          "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
          "dataset_sources": ["dmitrijkv/koz3-s3-cred"], "competition_sources": [], "kernel_sources": [], "model_sources": [],
          "docker_image": "gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"}
    json.dump(md, open(H / mode / "kernel-metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(mode, len((H / mode / f"{name}.py").read_bytes()), "байт")
