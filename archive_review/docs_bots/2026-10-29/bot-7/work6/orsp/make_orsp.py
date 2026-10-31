"""bot-7: ТЕСТОВЫЙ архив с ORSP bot-6 (abae911) ВКЛ. — только для сквозных проверок (a)–(e), не для отправки.
    python make_orsp.py <zip выкл.> <zip ORSP вкл.>
Конфиг: [post.sp_score_fusion] radius_m 8, gen_floor 0,05, пороги — медиана по фолдам (abae911)."""
import sys, zipfile
from pathlib import Path
SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
POVR = sys.argv[3] if len(sys.argv) > 3 else "0.2665015"   # sp_focal povr: 0.2665 (k10 медиана) / 0.3645 (v2)
BLOCK = '''

# bot-7 ТЕСТ ORSP (bot-6 abae911), не для отправки
[post.sp_score_fusion]
radius_m = 8.0
gen_floor = 0.05
thresholds.sp_focal = { kurgany_povrezhdennye = POVR, kurgany_tselye = 0.3696 }
thresholds.sp_focal2 = { kurgany_povrezhdennye = 0.514733, kurgany_tselye = 0.701621 }
'''
zin = zipfile.ZipFile(SRC)
cfg = zin.read("configs/default.toml").decode()
assert "\n[post.sp_score_fusion]" not in cfg
cfg += BLOCK.replace("POVR", POVR)
import tomllib; _c = tomllib.loads(cfg)["post"]["sp_score_fusion"]; assert _c["gen_floor"] == 0.05 and _c["thresholds"]["sp_focal"]["kurgany_povrezhdennye"] == float(POVR)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for it in zin.infolist():
        data = cfg.encode() if it.filename == "configs/default.toml" else zin.read(it)
        z.writestr(it, data)
print("ok", OUT, OUT.stat().st_size)
