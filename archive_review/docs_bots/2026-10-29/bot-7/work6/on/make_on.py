"""bot-7: ТЕСТОВЫЙ архив v3 с флагами ВКЛ. (не финал, не для отправки): из zip «выкл.» + модули за флагом,
веса D-FINE f0 (fp32), вендор колёс bot-9 (распакованы в work6/vendor_test/lib), конфиг: [sp_dfine] R + пороги bot-9
(в выборке), gor_gate и kurgan_in_gor ВЫКЛ. (вариант R, как принято ботом 10)."""
import re, sys, zipfile
from pathlib import Path
SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
REPO = Path("/workspace/koz3_shared/bot-7/repo"); W = Path("/workspace/koz3_shared/bot-7/work6")
MODE = sys.argv[3] if len(sys.argv) > 3 else "R"   # R — D-FINE вариант R; gate — только шлюз городищ bot-5 (без R)
F0 = Path(sys.argv[4] if len(sys.argv) > 4 else "/workspace/kaggle_out/koz3-dfine-spor-full-rest/dfine_f0")
# Модель СДАЧИ — f0 (решение bot-10 29.10). Финал (koz3-dfine-spor-final/dfine_final) — резерв: НЕ замена каталога,
# нужны перенесённые по квантилям (или честно перекалиброванные) пороги sp_dfine (argv[5], JSON {класс: порог})
# и отдельная приёмка bot-10 — см. FINAL_VARIANT.md. Без файла порогов сборка с не-f0 весами запрещена.
MNAME = F0.name
THR = {"kurgany_povrezhdennye": 0.555265, "kurgany_tselye": 0.670285, "gorodishcha": 0.68847,
       "arkhitektury": 0.373015, "fortifikatsii": 0.869051}   # bot-9 submission_dfine/thr/sweep.log R_max (f0)
if MODE != "gate" and MNAME != "dfine_f0":
    if len(sys.argv) <= 5:
        sys.exit(f"{MNAME}: веса не f0 — нужны перенесённые пороги sp_dfine (argv[5]) и приёмка bot-10, см. FINAL_VARIANT.md")
    import json
    THR = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
    assert set(THR) == {"kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"}, THR
SNAP = next(Path("/workspace/koz3_shared/bot-5/tmp/hf/hub/models--facebook--dinov2-small/snapshots").glob("ed25f*"))
HEAD = Path("/workspace/koz3_shared/bot-5/analysis/gor_patch/gor_v3_head.npz")
zin = zipfile.ZipFile(SRC)
cfg = zin.read("configs/default.toml").decode()
if MODE == "gate":
    cfg, n = re.subn(r"(?m)^\[gor_gate\]\nenabled = false\n", '''[gor_gate]
enabled = true
backbone = "models/dinov2-small"
head = "models/gor_gate/gor_v3_head.npz"

[thresholds_by_markup.sp_or_gated]   # bot-5 gor_v3_taus.json (sweep_groups на всём трейне)
gorodishcha = 0.992648
''', cfg); assert n == 1
cfg, n = (cfg, 1) if MODE == "gate" else re.subn(r"(?m)^\[sp_dfine\]\nenabled = false\n", '''[sp_dfine]
enabled = true
weights = ["models/MNAME"]   # модель сдачи — dfine_f0 (final только после переноса порогов и приёмки)
flip_tta = true
fp16 = true
convert = "circle"
drop_groups = ["sp_or", "sp_focal", "sp_arch", "sp_focal2"]

[thresholds_by_markup.sp_dfine]   # THRSRC
THRLINES
'''.replace("MNAME", MNAME).replace("THRSRC", "bot-9 submission_dfine/thr/sweep.log R_max (f0, подбор В ВЫБОРКЕ)"
          if MNAME == "dfine_f0" else f"ПЕРЕНЕСЕНЫ для {MNAME}: {sys.argv[5] if len(sys.argv) > 5 else ''}").replace(
          "THRLINES", "\n".join(f"{k} = {v}" for k, v in THR.items())), cfg); assert n == 1
cfg, n = (cfg, 1) if MODE == "gate" else re.subn(r"(?m)^(\[thresholds_by_markup\.or\]\narkhitektury = )0\.065565", r"\g<1>0.483891", cfg); assert n == 1
# gor_gate при R бессмыслен (bot-10 §1), kurgan_in_gor отклонён (KIG) — оба ВЫКЛ.
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for i in zin.infolist():
        z.writestr(i, cfg if i.filename == "configs/default.toml" else zin.read(i.filename))
    if MODE == "gate":
        z.write(REPO / "expds" / "gor_gate.py", "expds/gor_gate.py")
        for f in ("config.json", "model.safetensors"):
            z.write(SNAP / f, f"models/dinov2-small/{f}", compress_type=zipfile.ZIP_STORED)
        z.write(HEAD, "models/gor_gate/gor_v3_head.npz")
        z.write(REPO / "expds" / "sp_dfine.py", "expds/sp_dfine.py")   # import_dfine (закреплённый transformers)
    else:
        z.write(REPO / "expds" / "sp_dfine.py", "expds/sp_dfine.py")
        for f in ("config.json", "model.safetensors"):
            z.write(F0 / f, f"models/{MNAME}/{f}", compress_type=zipfile.ZIP_STORED)
    lib = W / "vendor_test" / "lib"
    for p in sorted(lib.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            z.write(p, "inference/lib/vendor/" + p.relative_to(lib).as_posix())
print(OUT, OUT.stat().st_size)
