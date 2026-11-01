"""configs/v4_on.toml = configs/v3_on.toml + w5_cloudfeat=true + [w5cf] + пороги li-групп из runs/v4_fit.json
(PREREG docs/PREREG_w4_v4_cloudfeat.md §1). Прочие строки v3_on — байт в байт."""
import json, re, sys
from pathlib import Path
R = Path(__file__).resolve().parents[5]
fit = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/workspace/koz3/orch/bot-W4/runs/v4_fit.json"))
CL = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "object_poly", "fortifikatsii", "finds_points"]
LIG = ["li", "li_dev", "li_mosaic", "li_coarse", "li_fort", "li_mask", "gor_phys"]
thr = fit["thresholds_li_groups"]; a = fit["alpha_deploy"]
src = (R / "configs/v3_on.toml").read_text(encoding="utf-8").splitlines(keepends=True)
out, cur, done = [], None, set()
hdr = ("# configs/v4_on.toml — v4 = v3_on + RR1s-cf (W1, J2 без M_rp) за флагом w5_cloudfeat (W4, bots/w4-v3on).\n"
       "# Сгенерирован docs/bots/2026-11-01/bot-W4/v4/make_v4_on.py: α и пороги li-групп — PREREG docs/PREREG_w4_v4_cloudfeat.md\n"
       f"# (α = мода вложенного выбора J2 = {a}; пороги li-групп — один sweep по всем фолдам в шкале S, нет порога → 1.01 выкл.).\n"
       "# Нужен модуль инференса признаков облака W1 (expds/w5cf_infer.py) и models/w5cf/ref.npz, иначе прогон\n"
       "# остановится в начале с W5CloudFeatMissing.\n")
out.append(hdr)
for line in src:
    m = re.match(r"^\[thresholds_by_markup\.([A-Za-z0-9_]+)\]", line)
    if m or re.match(r"^\[", line):
        cur = m.group(1) if m else None
        out.append(line)
        if cur in LIG:
            out.append(f"# W4 v4: шкала S (RR1s-cf, α={a}); значения v3_on заменены\n")
            for c in CL:
                v = thr.get(cur, {}).get(c)
                out.append(f"{c} = {v:.6f}\n" if v is not None else f"{c} = 1.01\n")
            done.add(cur)
        continue
    if cur in LIG and re.match(r"^\s*[a-z_]+\s*=", line):
        continue                     # старые пороги li-группы (шкала conf) — выкинуты
    out.append(line)
    if line.startswith("default_threshold"):
        out.append("w5_cloudfeat = true\n")
assert done == set(LIG), done
out.append(f"\n# W4 v4: RR1s-cf (expds/w5_cloudfeat.py)\n[w5cf]\nalpha = {a}\nmodule = \"expds.w5cf_infer\"\n"
           f"ref = \"models/w5cf/ref.npz\"\ngroups = {json.dumps(LIG)}\n")
(R / "configs/v4_on.toml").write_text("".join(out), encoding="utf-8")
print("ok", R / "configs/v4_on.toml")
