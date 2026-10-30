# bot-1 (29.10, кандидат v4): ядра D-FINE всех классов на Ae и Or = ядра бота 9 (koz3-dfine-ae / koz3-dfine-or, команда)
# + своя вставка aeor_patch.py: частичная разметка (класс без файла разметки на участке не штрафуется как фон),
# TTA flip × 2 масштаба (1.0, 1.5), WBF видов внутри одного снимка. Порог-группы ae_dfine_pl / or_dfine_pl (своя OOF).
# Ничего не пушит. Сборка: python3 kaggle/aeor_dfine/build.py [путь к bot-9/kaggle/dfine_spor]
# Пуш (вручную, после решения бота 1/10):
#   KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p kaggle/aeor_dfine/out/ae_pl
#   KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p kaggle/aeor_dfine/out/or_pl
import hashlib, json, pathlib, sys

H = pathlib.Path(__file__).resolve().parent
B9 = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/koz3_shared/bot-9/kaggle/dfine_spor")
PATCH = (H / "aeor_patch.py").read_text(encoding="utf-8")
STEP5 = "# ── Шаг 5."
INSERT = (PATCH + "\nREGION_MASK = {i['region']: region_class_mask(i) for i in INV}\n"
          "log('частичная разметка:', PARTIAL, '| размечено классов по участкам:', {r: sum(m) for r, m in REGION_MASK.items()})\n"
          "from transformers import DFineForObjectDetection as _DFOD\ninstall_partial_labels(_DFOD)\n"
          "SP_GROUP = SP_GROUP.replace('_dfine', '_dfine_pl')\nlog('TTA', tta_views(), 'слияние', FUSE, '| группа', SP_GROUP)\n\n")
PATCHES = [
    (STEP5, INSERT + STEP5),
    ('yield torch.from_numpy(img), {"class_labels"',
     'yield torch.from_numpy(img), {"class_mask": torch.tensor(REGION_MASK[r["region"]]), "class_labels"'),
    ('sc = Scene(inv, key, size); B, L, Cc = [], [], []',
     'sc = Scene(inv, key, size); VIEWS = tta_views(); PV = [([], [], []) for _ in VIEWS]'),
    ('            for flip in (False, True):\n'
     '                inp = px.flip(3) if flip else px\n'
     '                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):\n'
     '                    o = model(pixel_values=inp)\n'
     '                prob = o.logits.float().sigmoid(); bx = o.pred_boxes.float()\n'
     '                if flip: bx = bx.clone(); bx[..., 0] = 1 - bx[..., 0]\n',
     '            for vi, (sf, flip) in enumerate(VIEWS):\n'
     '                prob, bx = run_view(model, px, sf, flip, dev); B, L, Cc = PV[vi]\n'),
    ('        if B:\n            B, L, Cc = torch.cat(B), torch.cat(L), torch.cat(Cc); k = batched_nms(B, L, Cc, CFG["nms_iou"]); B, L, Cc = B[k], L[k], Cc[k]\n',
     '        _fz = fuse_region_views(PV, CFG["nms_iou"])\n        if _fz is not None:\n            B, L, Cc = _fz\n'),
]
VARIANTS = [("ae_pl", B9 / "ae_o2c/koz3-dfine-ae.py", "koz3-aeor-pl-ae", B9 / "ae_o2c/kernel-metadata.json"),
            ("or_pl", B9 / "or_o2c/koz3-dfine-or.py", "koz3-aeor-pl-or", B9 / "or_o2c/kernel-metadata.json")]


def apply(text):
    for old, new in PATCHES:
        assert text.count(old) == 1, ("патч не найден ровно 1 раз", old[:70])
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    for d, src, name, meta in VARIANTS:
        out = H / "out" / d
        out.mkdir(parents=True, exist_ok=True)
        code = apply(src.read_text(encoding="utf-8"))
        compile(code, name, "exec")
        (out / f"{name}.py").write_text(code, encoding="utf-8")
        md = json.loads(meta.read_text(encoding="utf-8"))
        md.update({"id": f"dmitrijkv/{name}", "title": name, "code_file": f"{name}.py"})
        (out / "kernel-metadata.json").write_text(json.dumps(md, ensure_ascii=False, indent=1), encoding="utf-8")
        print(d, name, len(code.encode()), "байт", hashlib.sha256(code.encode()).hexdigest()[:16],
              "| база", src.name, hashlib.sha256(src.read_bytes()).hexdigest()[:16])
