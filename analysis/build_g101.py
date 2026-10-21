"""Тайлы лидара только участка 101 — новые La_точки, тот же рецепт, что li_tin_hard2."""
import json
import logging
import sys

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from expds.tiles import build_cache

meta = build_cache(
    r"F:\expds3\train-expds-3",
    r"F:\expds3\work\gt_train.geojson",
    "li",
    r"F:\expds3\work\cache\li_g101",
    res=0.5,
    tile=128,
    dem={"res": 0.1, "fill": "delaunay"},
    only_regions=["101_Гнёздово_Днепровская"],
)
print(json.dumps(meta, ensure_ascii=False))
