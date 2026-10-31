"""bot-7: сводка проверок ORSP (a)–(e) по двум дампам run_hooked.py и выходам (выкл./вкл.).
    python orsp_report.py dump_off.json out_off.geojson dump_on.json out_on.geojson [e2e.json] > orsp_checks.txt"""
import collections, json, sys
d_off, o_off, d_on, o_on = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:5])
e2e = json.load(open(sys.argv[5], encoding="utf-8")) if len(sys.argv) > 5 else None
KUR = ("kurgany_povrezhdennye", "kurgany_tselye")
def kc(o):
    c = collections.Counter((f["properties"].get("markup_type"), f["properties"]["class_name"]) for f in o["features"]
                            if f["properties"]["class_name"] in KUR)
    return {f"{m}|{k}": n for (m, k), n in sorted(c.items())}
print("(a) курганы в выдаче по markup_type: выкл.", kc(o_off), "\n    вкл.", kc(o_on))
if e2e: print("    балл (официальная метрика, e2e_score):", json.dumps(e2e.get("total", e2e), ensure_ascii=False)[:400])
fu = d_on["fusion"]
print("(b)/(c) вызовов слияния:", len(fu), "(в выкл. —", len(d_off["fusion"]), ")")
for k, f in enumerate(fu):
    b = f["before"]; po = [x for x in b if x["partner_only"]]
    print(f"  [{k}] до слияния sp-курганов {len(b)}, из них sp_partner_only {len(po)} (группы "
          f"{dict(collections.Counter(x['group'] for x in po))}); после {len(f['after'])}; объектов {f['n_in']}→{f['n_out']}")
    print("      partner_only в выходе слияния:", sum(x["partner_only"] for x in f["after"]), "(должно быть 0)")
for k, v in enumerate(d_on["verify"]):
    if v["after"] is None: print(f"  verify[{k}]: проверка кропа вернула None"); continue
    ba = {x["id"]: x for x in v["before"]}
    ch = [x for x in v["after"] if x["id"] in ba and abs(x["conf"] - ba[x["id"]]["conf"]) > 1e-9]
    print(f"  verify[{k}] (до finish, т.е. ДО слияния): sp-курганов {len(v['before'])}→{len(v['after'])}, уверенность изменена у {len(ch)}")
if d_on["thresholds"]:
    last = d_on["thresholds"][-1]; fa = {x["id"]: x["conf"] for x in fu[-1]["after"]} if fu else {}
    same = sum(1 for x in last["sp_after"] if x["id"] in fa and abs(fa[x["id"]] - x["conf"]) < 1e-12)
    print(f"(b) после порогов sp-курганов {len(last['sp_after'])}, уверенность = слитой у {same} (сброса нет, если равно)")
print("(d) merge_within_markup, объектов по (группа|класс) до→после (вкл.):")
for m in d_on["merge"]:
    for k in sorted(set(m["before"]) | set(m["after"])):
        if m["before"].get(k, 0) != m["after"].get(k, 0): print("   ", k, m["before"].get(k, 0), "→", m["after"].get(k, 0))
print("    то же выкл.:")
for m in d_off["merge"]:
    for k in sorted(set(m["before"]) | set(m["after"])):
        if m["before"].get(k, 0) != m["after"].get(k, 0): print("   ", k, m["before"].get(k, 0), "→", m["after"].get(k, 0))
print("(e) правило: включать только при Δ>0 и P≥0,85 по честному замеру bot-10; здесь — один участок, это не приёмка.")
