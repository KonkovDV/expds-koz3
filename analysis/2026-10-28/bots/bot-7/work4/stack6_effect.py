"""Работает ли sota_stack6 в кандидате вообще: finish(кандидат) против finish(stack6 выкл.) — счётчики по классу|группе."""
import copy, json, sys, tomllib
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from shapely.geometry import shape
from expds import watchdog
cfg = tomllib.loads(Path("configs/default.toml").read_text(encoding="utf-8"))
off = copy.deepcopy(cfg); off["post"]["multipole_object_poly"] = False; off["post"]["filter_finds_points"] = False
key = lambda d: f'{d.get("class_name")}|{d.get("threshold_group") or d.get("markup_type")}'
res = {}
for fam in sys.argv[1].split(","):
    feats = [dict(json.loads(l)["properties"], geom=shape(json.loads(l)["geometry"])) for l in open(f"/workspace/koz3_shared/bot-10/work/fam/{fam}.jsonl", encoding="utf-8")]
    on, of = Counter(map(key, watchdog.finish([dict(f) for f in feats], cfg))), Counter(map(key, watchdog.finish(feats, off)))
    res[fam] = {k: [of.get(k, 0), on.get(k, 0)] for k in sorted(set(on) | set(of)) if on.get(k, 0) != of.get(k, 0)}
    print(fam, "[выкл, вкл]", json.dumps(res[fam], ensure_ascii=False), flush=True)
Path(__file__).with_name("stack6_effect.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
