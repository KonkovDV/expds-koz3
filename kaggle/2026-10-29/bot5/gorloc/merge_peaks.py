"""bot-9 → bot-5: склейка peaks{TAG}.json из нескольких кернелов (напр. _L12: probe-l12a [fam10+seed0-4] + probe-l12b [LOFO]).
layouts/stats объединяются без изменений (сиды фиксированы внутри dense_probe → те же числа, что при одном процессе;
проверено сухим прогоном, bot-9/kaggle/bot5/gorloc/dry_logs/). hn_log/cp_s_chosen — суммы счётчиков; при раздельных
процессах чуть больше фитов (нет общего _cache), на метрики не влияет.
Использование: python merge_peaks.py OUT.json IN1.json IN2.json ..."""
import json, sys, collections
out, ins = sys.argv[1], sys.argv[2:]; R = None
for f in ins:
    P = json.load(open(f, encoding="utf-8"))
    if R is None:
        R = {k: v for k, v in P.items() if k not in ("layouts", "stats", "hn_log", "cp_s_chosen")}
        R.update(layouts={}, stats={}, hn_log=collections.Counter(), cp_s_chosen={})
    else:
        for k in ("model", "gsds", "tag", "n_tokens", "token_labels"): assert P[k] == R[k], f"{f}: {k} не совпадает"
    dup = set(P["layouts"]) & set(R["layouts"]); assert not dup, f"раскрой дважды: {dup}"
    R["layouts"].update(P["layouts"]); R["stats"].update(P["stats"]); R["hn_log"].update(P.get("hn_log", {}))
    for v, c in P.get("cp_s_chosen", {}).items(): R["cp_s_chosen"].setdefault(v, collections.Counter()).update(c)
json.dump(R, open(out, "w", encoding="utf-8"), ensure_ascii=False); print(out, sorted(R["layouts"]))
