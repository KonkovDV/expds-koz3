"""orjson.loads даёт те же объекты Python, что json.loads, на наших GeoJSON (включая repr float)."""
import json, time, glob, os
import orjson

FILES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "prof", "*.geojson")))


def _same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return a.hex() == b.hex()
    return a == b


def test_same_objects():
    assert FILES
    for f in FILES:
        t = open(f, encoding="utf-8-sig").read()
        assert _same(json.loads(t), orjson.loads(t)), f


if __name__ == "__main__":
    for f in FILES:
        t = open(f, encoding="utf-8-sig").read(); b = t.encode()
        best = lambda fn: min((lambda s: (fn(), time.perf_counter() - s)[1])(time.perf_counter()) for _ in range(5))
        a, o = best(lambda: json.loads(t)), best(lambda: orjson.loads(b))
        print(os.path.basename(f), len(b) // 1000, "KB json", round(a, 3), "orjson", round(o, 3), "x", round(a / o, 1))
