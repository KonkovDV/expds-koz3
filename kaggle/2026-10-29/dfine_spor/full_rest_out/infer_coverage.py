# Покрытие инференса full_rest: окна/tif из лога vs отчёт vs локальная инвентаризация (gt_coverage.json из full_out), плотность детекций.
import json, re, collections, statistics as S
log = open("log_plain.txt", encoding="utf-8").read().splitlines()
rep = json.load(open("dfine_report.json"))
gc = {r["region"]: r for r in json.load(open("../full_out/gt_coverage.json"))}
start = {}; done = collections.Counter(); ntif_done = collections.Counter()
for l in log:
    m = re.search(r"\[infer\] (\S+) (\S+\.tif): (\d+)x(\d+) px, окон (\d+)", l)
    if m: start[(m[1], m[2])] = int(m[5])
    m = re.search(r"\[infer\] (\S+): готово (\d+) окон", l)
    if m: done[m[1]] += int(m[2]); ntif_done[m[1]] += 1
oof = json.load(open("/workspace/kaggle_out/koz3-dfine-spor-full-rest/oof_dfine_spor.geojson"))["features"]
det = collections.Counter(f["properties"]["region_name"] for f in oof)
conf = collections.defaultdict(list)
for f in oof: conf[f["properties"]["fold"]].append(f["properties"]["confidence"])
out = {}
for fo, v in rep["folds"].items():
    regs = v["test_regions"]; loc = sorted(r for r, x in gc.items() if str(x["fold"]) == fo)
    w_st = sum(n for (r, t), n in start.items() if r in regs); w_dn = sum(done[r] for r in regs)
    tif_st = sum(1 for (r, t) in start if r in regs); tif_dn = sum(ntif_done[r] for r in regs)
    tif_exp = sum(gc[r]["n_tifs"] for r in regs)
    gt = sum(gc[r]["gt"] for r in regs); nd = sum(det[r] for r in regs)
    c = sorted(conf[int(fo)])
    out[fo] = dict(test_regions=len(regs), local_regions=len(loc), set_equal=set(regs) == set(loc), tifs_started=tif_st, tifs_done=tif_dn, tifs_expected=tif_exp,
                   windows=w_st, windows_done=w_dn, gt=gt, n_det_report=v["n_det"], n_det_oof=nd, det_per_window=round(nd / w_st, 1), gt_per_window=round(gt / w_st, 3),
                   det_per_gt=round(nd / gt, 1), conf_median=round(S.median(c), 4), conf_ge03=sum(x >= 0.3 for x in c), conf_ge05=sum(x >= 0.5 for x in c),
                   per_region={r: dict(tifs=sum(1 for (rr, t) in start if rr == r), windows=sum(n for (rr, t), n in start.items() if rr == r), gt=gc[r]["gt"], det=det[r]) for r in regs})
    print(f"fold {fo}: regions {len(regs)} (local {len(loc)}, equal {set(regs)==set(loc)}), tifs started {tif_st} done {tif_dn} expected {tif_exp}, windows {w_st} done {w_dn}, "
          f"gt {gt}, n_det report {v['n_det']} oof {nd}, det/window {nd/w_st:.1f}, gt/window {gt/w_st:.3f}, det/gt {nd/gt:.1f}, conf median {S.median(c):.4f}, conf>=0.3 {out[fo]['conf_ge03']}, >=0.5 {out[fo]['conf_ge05']}")
    for r, x in out[fo]["per_region"].items(): print("   ", r, x)
print("infer_skipped top-level:", rep.get("infer_skipped"), "| errors line:", [l for l in log if "ошибки фолдов" in l])
json.dump(out, open("infer_coverage.json", "w"), ensure_ascii=False, indent=1)
