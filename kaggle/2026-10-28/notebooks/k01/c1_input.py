# ── Ячейка 2. Вход: наличие и схема npz ─────────────────────────────────────────────────
def _ls(root, n=60):
    root = Path(root)
    return [str(p) for p in sorted(root.rglob("*"))[:n]] if root.exists() else []

def find_npz():
    global INPUT_DIR
    if not INPUT_DIR.exists() and Path("/kaggle/input").exists():  # новый формат монтирования: /kaggle/input/datasets/<owner>/<slug>
        alt = [p for p in Path("/kaggle/input").rglob(INPUT_DIR.name) if p.is_dir()]
        if alt:
            print(f"{INPUT_DIR} нет, датасет найден в {alt[0]}", flush=True)
            INPUT_DIR = alt[0]
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Нет входного датасета {INPUT_DIR}. Подключите dmitrijkv/koz3-kurgan-subclass-crops "
            f"(kernel-metadata.json → dataset_sources). Содержимое /kaggle/input: {_ls('/kaggle/input')}")
    named = sorted(INPUT_DIR.rglob(CFG["npz_name"]))
    allz = sorted(INPUT_DIR.rglob("*.npz"))
    if named:
        return named[0]
    if len(allz) == 1:
        print(f"файл {CFG['npz_name']} не найден, беру единственный npz: {allz[0]}", flush=True)
        return allz[0]
    raise FileNotFoundError(f"В {INPUT_DIR} нет {CFG['npz_name']} (npz найдено: {[str(p) for p in allz]}). "
                            f"Содержимое: {_ls(INPUT_DIR)}")

def load_and_check(path):
    D = np.load(path, allow_pickle=False)
    keys = set(D.files)
    errs = []
    miss = [k for k in CFG["schema"]["required"] if k not in keys]
    if miss:
        raise ValueError(f"{path}: нет обязательных полей {miss}; есть {sorted(keys)}. "
                         "Ожидается вывод scripts/bot9_kurgan_subclass.py crops: "
                         "X uint8 (N,3,H,W), y int (−1 нет метки / 0 целый / 1 повреждённый), region str (N,)")
    X, y, region = D["X"], D["y"], D["region"]
    N = len(y)
    if X.ndim != 4:
        errs.append(f"X.ndim={X.ndim}, нужно 4 (N,C,H,W)")
    else:
        if X.shape[1] not in (1, 2, 3, 4) and X.shape[-1] in (1, 2, 3, 4):
            errs.append(f"X.shape={X.shape}: похоже на NHWC, нужно NCHW")
        if X.shape[1] != CFG["schema"]["channels"]:
            errs.append(f"каналов {X.shape[1]}, нужно {CFG['schema']['channels']} (lrm_s, lrm_l, slope)")
        if X.shape[2] != X.shape[3] or X.shape[2] < 32:
            errs.append(f"окно {X.shape[2]}×{X.shape[3]}: нужно квадратное ≥ 32")
        if len(X) != N:
            errs.append(f"len(X)={len(X)} ≠ len(y)={N}")
    if X.dtype != np.uint8:
        errs.append(f"X.dtype={X.dtype}, нужно uint8 (нормировка (x/255−0.5)/0.25 как у кэша обучения)")
    if not np.issubdtype(y.dtype, np.integer):
        errs.append(f"y.dtype={y.dtype}, нужно целое")
    else:
        bad = sorted(set(np.unique(y).tolist()) - {-1, 0, 1})
        if bad:
            errs.append(f"y содержит {bad}, допустимо только −1/0/1")
    if region.dtype.kind not in "UO" or len(region) != N:
        errs.append(f"region: dtype {region.dtype}, len {len(region)} — нужен str (N,)")
    for k in CFG["schema"]["optional_len_n"]:
        if k in keys and len(D[k]) != N:
            errs.append(f"{k}: len {len(D[k])} ≠ {N}")
    if errs:
        raise ValueError(f"{path}: схема не совпала:\n  - " + "\n  - ".join(errs))
    lab = y >= 0
    if lab.sum() < CFG["schema"]["min_labeled"]:
        raise ValueError(f"размеченных окон {int(lab.sum())} < {CFG['schema']['min_labeled']}")
    if len(set(y[lab].tolist())) < 2:
        raise ValueError("в разметке один класс — AUC не определён")
    return D

NPZ = find_npz()
D = load_and_check(NPZ)
X = np.ascontiguousarray(D["X"])
y = D["y"].astype(np.int64)
region = D["region"].astype(str)
kind = D["kind"].astype(str) if "kind" in D.files else np.array(["gt"] * len(y))
stack_idx = D["stack_idx"].astype(np.int64) if "stack_idx" in D.files else np.full(len(y), -1)
cand_class = D["cand_class"].astype(str) if "cand_class" in D.files else np.array([""] * len(y))
lab = y >= 0
print(json.dumps({"npz": str(NPZ), "N": int(len(y)), "shape": list(X.shape), "labeled": int(lab.sum()),
                  "povr_share": round(float((y[lab] == 1).mean()), 4), "kinds": {k: int((kind == k).sum()) for k in sorted(set(kind))},
                  "regions": int(len(set(region)))}, ensure_ascii=False), flush=True)
