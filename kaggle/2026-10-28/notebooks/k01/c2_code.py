# ── Ячейка 3. smp и веса ImageNet из бандла (интернет не нужен) ────────────────────────────
def resolve_code_dir():
    global CODE_DIR
    ok = lambda d: (d / "expds_code.tar.gz").exists() or (d / "expds_code").is_dir()
    if not ok(CODE_DIR) and Path("/kaggle/input").exists():  # как find_bundle в koz3-gpu-s4.py:29–39
        hits = sorted(p.parent for p in Path("/kaggle/input").rglob("expds_code*") if ok(p.parent))
        if hits:
            CODE_DIR = hits[0]
    return CODE_DIR

resolve_code_dir()

def setup_smp():
    try:
        import segmentation_models_pytorch  # noqa: F401
        return "preinstalled"
    except ImportError:
        pass
    tars = sorted(CODE_DIR.rglob("expds_code.tar.gz")) if CODE_DIR.exists() else []
    dirs = sorted(p.parent for p in CODE_DIR.rglob("segmentation_models_pytorch/__init__.py")) if CODE_DIR.exists() else []
    dst = WORK / "code"
    if tars:
        with tarfile.open(tars[0], "r:gz") as t:
            t.extractall(dst)
    elif dirs:
        shutil.copytree(dirs[0], dst, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(f"нет segmentation_models_pytorch: подключите dmitrijkv/koz3-li-tin-59-t128-bundle "
                                f"(expds_code.tar.gz). Содержимое {CODE_DIR}: {_ls(CODE_DIR, 30)}")
    sys.path.insert(0, str(dst))
    import segmentation_models_pytorch  # noqa: F401
    return str(tars[0] if tars else dirs[0])

HF_SPECS = {  # пути внутри бандла dmitrijkv/koz3-li-tin-59-t128-bundle — как в /workspace/kpull/s4/koz3-gpu-s4.py:55–68
    "resnet18": ("models--smp-hub--resnet18.imagenet", "3f2325ff978283d47aa6a1d6878ca20565622683",
                 "models--smp-hub--resnet18.imagenet/blobs/879c59b3870194ae248b52920479fed460415e62",
                 "blobs/d5/d5da100346311aff6748ca74f737a7e9178dae9e48b071f61a8ccf9fa3ad88d7"),
    "resnet34": ("models--smp-hub--resnet34.imagenet", "7a57b34f723329ff020b3f8bc41771163c519d0c",
                 "models--smp-hub--resnet34.imagenet/blobs/879c59b3870194ae248b52920479fed460415e62",
                 "blobs/bc/bcb6928a067cda2ccbcf44c2c810cf759f69ff578332b4ac57ff2a6eb3984b13"),
}

def setup_hf_cache():
    """Kaggle-датасет теряет симлинки кэша HF: собираем snapshots/<hash>/{config.json,model.safetensors} заново."""
    src = next(iter(sorted(p for p in CODE_DIR.rglob("hf_hub_cache") if p.is_dir())), None) if CODE_DIR.exists() else None
    dst = WORK / "hf_hub_cache"
    if src is None and (CODE_DIR / "hf_hub_cache.tar.gz").exists():
        with tarfile.open(CODE_DIR / "hf_hub_cache.tar.gz", "r:gz") as t:  # архив хранит симлинки — кэш готов как есть
            t.extractall(dst)
        os.environ["HF_HUB_CACHE"] = str(dst); os.environ["HF_HUB_OFFLINE"] = "1"
        return {"from": "hf_hub_cache.tar.gz"}
    found = {}
    for enc, (repo, snap, cfg_rel, w_rel) in HF_SPECS.items():
        if src is None:
            break
        cfg, w = src / cfg_rel, src / w_rel
        if not (cfg.is_file() and w.is_file() and w.stat().st_size > 1_000_000):
            continue
        sd = dst / repo / "snapshots" / snap
        sd.mkdir(parents=True, exist_ok=True)
        (dst / repo / "refs").mkdir(parents=True, exist_ok=True)
        (dst / repo / "refs" / "main").write_text(snap)
        for name, p in (("config.json", cfg), ("model.safetensors", w)):
            if not (sd / name).exists():
                os.symlink(p.resolve(), sd / name)
        found[enc] = {"weights": str(w), "bytes": w.stat().st_size}
    need = sorted({m["encoder"] for m in CFG["models"]})
    if CFG["pretrained"] and not all(e in found for e in need):
        raise FileNotFoundError(f"нет весов ImageNet для {[e for e in need if e not in found]} в {src}. "
                                "Подключите dmitrijkv/koz3-li-tin-59-t128-bundle (hf_hub_cache).")
    os.environ["HF_HUB_CACHE"] = str(dst)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return found

SMP_FROM = setup_smp()
HF = setup_hf_cache() if CFG["pretrained"] else {}
print("smp:", SMP_FROM, "| imagenet в кэше:", json.dumps(HF, ensure_ascii=False), flush=True)
import segmentation_models_pytorch as smp

def make_model(encoder):
    weights = "imagenet" if CFG["pretrained"] else None
    try:
        enc = smp.encoders.get_encoder(encoder, in_channels=3, depth=5, weights=weights)
    except Exception as e:  # без молчаливого отката на случайную инициализацию
        raise RuntimeError(f"не загрузились веса {encoder}/{weights}: {e!r}. Проверьте бандл hf_hub_cache "
                           "или задайте CFG['pretrained']=false осознанно.") from e
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            self.drop = torch.nn.Dropout(CFG["dropout"])
            self.head = torch.nn.Linear(enc.out_channels[-1], 1)
        def forward(self, x):
            return self.head(self.drop(self.enc(x)[-1].mean(dim=(2, 3)))).squeeze(1)
    return Net()

for _enc in sorted({m["encoder"] for m in CFG["models"]}):
    make_model(_enc)  # падает здесь, до обучения, если весов нет
print("модели строятся:", sorted({m["encoder"] for m in CFG["models"]}), "pretrained", CFG["pretrained"], flush=True)
