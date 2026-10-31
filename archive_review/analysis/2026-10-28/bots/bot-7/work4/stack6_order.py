"""Порядок sota_stack6: как мерил bot-10 (stack6_fam.py: multipole+filter на сыром стеке ДО порогов, потом finish без stack6)
против кандидата (watchdog.finish: пороги → merge → multipole → filter). Сравнивает выдачу по классу|группе и совпадение объектов."""
import copy, json, sys, tomllib
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from shapely.geometry import shape
from expds import post, watchdog

FAM = Path("/workspace/koz3_shared/bot-10/work/fam")
cfg = tomllib.loads(Path("configs/default.toml").read_text(encoding="utf-8"))
off = copy.deepcopy(cfg); off["post"]["multipole_object_poly"] = False; off["post"]["filter_finds_points"] = False
key = lambda d: f'{d.get("class_name")}|{d.get("threshold_group") or d.get("markup_type")}'
sig = lambda d: (d["class_name"], round(d["geom"].centroid.x, 1), round(d["geom"].centroid.y, 1), round(d["geom"].area, 1))
res = {}
for fam in sys.argv[1].split(","):
    feats = []
    for line in open(FAM / f"{fam}.jsonl", encoding="utf-8"):
        d = json.loads(line); p = dict(d["properties"]); p["geom"] = shape(d["geometry"]); feats.append(p)
    a_in = post.filter_finds_points_regimes(post.multipole_object_poly([dict(f) for f in feats], None), None)
    A = watchdog.finish(a_in, off)                      # порядок bot-10
    B = watchdog.finish([dict(f) for f in feats], cfg)  # порядок кандидата
    sa, sb = Counter(map(sig, A)), Counter(map(sig, B))
    res[fam] = {"bot10_order": dict(Counter(map(key, A))), "candidate_order": dict(Counter(map(key, B))),
                "n_bot10": len(A), "n_candidate": len(B), "same_objects": sum((sa & sb).values()),
                "only_bot10": sum((sa - sb).values()), "only_candidate": sum((sb - sa).values())}
    print(fam, json.dumps(res[fam], ensure_ascii=False), flush=True)
Path(__file__).with_name("stack6_order.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
