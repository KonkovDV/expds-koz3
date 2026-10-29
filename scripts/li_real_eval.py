"""Задача 9 (REPORT2 §9): декодирование РЕАЛЬНЫХ карт li (дамп nn_oof, 5 семейных фолдов CNN) и подмена
li-объектов в стеке bot-10 (work/fam/*.jsonl) с сохранением структуры групп стека.

Варианты (--mode):
  legacy  — как pipeline._li_with_model: курганы — кандидаты рельефа (cand_x/y из дампа) с вероятностями
            sample_probs (окно 5) и кругом [candidates].circle_m; прочие классы — probs_to_features.
  peaks   — li_decode.decode_li по умолчанию (пики курганов + точки gor/fort).
  gfpoint — D1: курганы legacy, gor/fort — точка в компоненте (decode_components, круг 1 м).
  kurdual — D2: курганы — кандидаты рельефа + доля классов в окне (dual 0,3), gor/fort legacy.
  hybrid  — D3: курганы как kurdual + gor/fort/прочие как decode_li.
  ch      — декодер centerhead (bot-2/centerhead) на картах курганов li: пики окна 3, min_score 0,05,
            NMS по расстоянию max(1 м, 0,8·min r) внутри класса (r = default_diam_m/2), оба класса,
            круг 0,5 м; карты приводятся f16→f32. gor/fort/arch — legacy. Голова НЕ обучалась на этих картах.
После декодирования — post.merge_within_markup + post.finalize по участку (как nn_oof).

Подмена в стеке (одинаково для всех вариантов): курганы участка пишутся, только если в стеке у участка есть
li-курганы (группа li); прочие классы — в группу li, если у участка в стеке есть li-объекты группы li,
иначе в li_coarse, если есть li_coarse; иначе участок не получает li-объектов (как в стеке).
Из стека удаляются все объекты markup li с группой li или li_coarse; остальные группы (li_dev, li_mosaic,
li_mask, li_fort, gor_phys, …) не трогаются.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LI_CLASSES = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]
KUR = LI_CLASSES[:2]
REPLACED = ("li", "li_coarse")


def group_of(props: dict) -> str:
    return props.get("threshold_group") or props.get("markup_type") or ""


def scan_stack(src: Path) -> dict:
    """→ {region: {"li_kur": bool, "li": bool, "li_coarse": bool}} по объектам markup li стека."""
    info: dict = defaultdict(lambda: {"li_kur": False, "li": False, "li_coarse": False})
    for fp in sorted(src.glob("*.jsonl")):
        with fp.open() as fh:
            for line in fh:
                if '"li"' not in line:
                    continue
                p = json.loads(line)["properties"]
                if p.get("markup_type") != "li":
                    continue
                g = group_of(p)
                if g == "li":
                    info[p["region_name"]]["li"] = True
                    if p["class_name"] in KUR:
                        info[p["region_name"]]["li_kur"] = True
                elif g == "li_coarse":
                    info[p["region_name"]]["li_coarse"] = True
    return dict(info)


def target_group(info: dict, region: str, cls: str) -> str | None:
    r = info.get(region)
    if not r:
        return None
    if cls in KUR:
        return "li" if r["li_kur"] else None
    if r["li"]:
        return "li"
    if r["li_coarse"]:
        return "li_coarse"
    return None


def chan(probs16: np.ndarray, cls: str) -> np.ndarray:
    return np.ascontiguousarray(probs16[LI_CLASSES.index(cls)], np.float32)


def legacy_kurgans(probs16, tr, crs, region, source, cand_p, cx, cy):
    from expds import candidates
    if cx is None or len(cx) == 0:
        return [], None
    c = {"x": np.asarray(cx, np.float64), "y": np.asarray(cy, np.float64)}
    kp = np.stack([chan(probs16, k) for k in KUR])
    s = candidates.sample_probs(c, kp, tr)                       # N×2 (povr, tsel)
    del kp
    scores = np.column_stack([np.zeros(len(s), np.float32), s])
    return candidates.to_features(c, scores, crs, region, cand_p, tuple(KUR), source), (c, s)


def legacy_rest(probs16, classes, tr, crs, res, region, source):
    from expds.infer_nn import probs_to_features
    out = []
    for cls in classes:
        p = chan(probs16, cls)
        out += probs_to_features(p[None], [cls], tr, crs, res, region, "li", source)
        del p
    return out


def dual_kurgans(probs16, tr, crs, region, source, cand_p, cx, cy, prm, res):
    """Кандидаты рельефа: уверенность S = max(p) в окне 5; класс по доле средних в окне; второй — при доле ≥ dual."""
    from expds import candidates
    from expds.li_decode import _circles, _window_mean
    if cx is None or len(cx) == 0:
        return []
    c = {"x": np.asarray(cx, np.float64), "y": np.asarray(cy, np.float64)}
    kp = np.stack([chan(probs16, k) for k in KUR])
    s = candidates.sample_probs(c, kp, tr)
    S = s.max(1)
    col = np.clip(np.floor((c["x"] - tr.c) / tr.a).astype(np.int64), 0, kp.shape[2] - 1)
    row = np.clip(np.floor((tr.f - c["y"]) / tr.a).astype(np.int64), 0, kp.shape[1] - 1)
    w = int(prm["share_win_px"])
    m = np.stack([_window_mean(kp[i], row, col, w) for i in range(2)], 1)
    del kp
    share = m / np.maximum(m.sum(1, keepdims=True), 1e-12)
    best = share.argmax(1)
    pts = []
    for i in range(len(S)):
        for k, cls in enumerate(KUR):
            if k == best[i] or share[i, k] >= float(prm["dual"]):
                pts.append((int(row[i]), int(col[i]), cls, float(S[i] * share[i, k]), i))
    # круг в самой точке кандидата (а не в центре пикселя): радиус [candidates].circle_m, как у legacy
    from shapely import buffer, points
    from expds.crs import WEB_MERCATOR, to_crs
    if not pts:
        return []
    idx = np.array([q[4] for q in pts])
    geoms = buffer(points(c["x"][idx], c["y"][idx]), float(cand_p.circle_m), quad_segs=4)
    if crs and crs != WEB_MERCATOR:
        geoms = to_crs(geoms, crs, WEB_MERCATOR)
    base = {"region_name": region, "markup_type": "li", "source": source, "original_crs": crs}
    return [{"geom": g, "class_name": q[2], "confidence": q[3], **base} for g, q in zip(geoms, pts)]


def peaks_kurgans(probs16, tr, crs, region, source, prm, res):
    from expds.li_decode import _circles, decode_kurgans
    kp = np.stack([chan(probs16, k) for k in KUR])
    pts = decode_kurgans(kp, KUR, res, prm)
    del kp
    rad = prm["radius_m"]
    base = {"region_name": region, "markup_type": "li", "source": source, "original_crs": crs}
    return [{"geom": g, "class_name": q[2], "confidence": q[3], **base}
            for g, q in zip(_circles(pts, tr, crs, lambda q: rad[q[2]]), pts)]


def point_components(probs16, classes, tr, crs, region, source, prm, res, use_area=True):
    from expds.li_decode import _circles, decode_components
    rad = prm["radius_m"]
    base = {"region_name": region, "markup_type": "li", "source": source, "original_crs": crs}
    out = []
    for cls in classes:
        p = chan(probs16, cls)
        pts = decode_components(p, res, float(prm["comp_thr"]),
                                float(prm["min_area_m2"].get(cls, 0.0)) if use_area else 0.0)
        del p
        out += [{"geom": g, "class_name": cls, "confidence": q[2], **base}
                for g, q in zip(_circles(pts, tr, crs, lambda q, c=cls: rad[c]), pts)]
    return out


def ch_kurgans(probs16, tr, crs, region, source, topk):
    from centerhead.config import HM_CLASSES, cfg_with
    from centerhead.decode import decode
    from centerhead.export import candidates_to_feats
    cfg = cfg_with(min_score=0.05, peak_pool=3, nms_min_m=1.0, nms_frac=0.8, nms_cross_class=False,
                   circle_r_m=0.5, topk=int(topk), markup_type="li")
    hm = np.stack([chan(probs16, k) for k in HM_CLASSES])      # канал 0 — целые, 1 — повреждённые; float32
    t = decode(hm, tr.c, tr.f, tr.a, None, None, None, cfg)
    del hm
    feats = candidates_to_feats(t, region, crs, cfg=cfg)
    for f in feats:
        f["source"] = source
    return feats, int(len(t["X"]))


def decode_file(fp: Path, mode: str, cand_p, prm, topk: int) -> tuple[list[dict], dict]:
    from expds import post
    d = np.load(fp, allow_pickle=False)
    probs16 = d["probs"]
    from affine import Affine
    tr = Affine(*map(float, d["transform"][:6]))
    crs = str(d["crs"]) if "crs" in d.files else None
    region = str(d["region"])
    source = str(d["source"]) if "source" in d.files else fp.stem
    cx = d["cand_x"] if "cand_x" in d.files else None
    cy = d["cand_y"] if "cand_y" in d.files else None
    res = abs(tr.a)
    gf = ["gorodishcha", "fortifikatsii"]
    meta = {"region": region, "shape": list(probs16.shape), "res": res, "n_cand": 0 if cx is None else int(len(cx))}
    feats: list[dict] = []
    if mode in ("legacy", "gfpoint", "ch"):
        if mode == "ch":
            kf, meta["ch_peaks"] = ch_kurgans(probs16, tr, crs, region, source, topk)
        else:
            kf, _ = legacy_kurgans(probs16, tr, crs, region, source, cand_p, cx, cy)
        feats += kf
        if mode == "gfpoint":
            feats += point_components(probs16, gf, tr, crs, region, source, prm, res, use_area=False)
            feats += legacy_rest(probs16, ["arkhitektury"], tr, crs, res, region, source)
        else:
            feats += legacy_rest(probs16, LI_CLASSES[2:], tr, crs, res, region, source)
    elif mode == "peaks":
        feats += peaks_kurgans(probs16, tr, crs, region, source, prm, res)
        feats += point_components(probs16, gf, tr, crs, region, source, prm, res, use_area=True)
        feats += legacy_rest(probs16, ["arkhitektury"], tr, crs, res, region, source)
    elif mode in ("kurdual", "hybrid"):
        feats += dual_kurgans(probs16, tr, crs, region, source, cand_p, cx, cy, prm, res)
        if mode == "hybrid":
            feats += point_components(probs16, gf, tr, crs, region, source, prm, res, use_area=True)
            feats += legacy_rest(probs16, ["arkhitektury"], tr, crs, res, region, source)
        else:
            feats += legacy_rest(probs16, LI_CLASSES[2:], tr, crs, res, region, source)
    else:
        raise SystemExit(f"unknown mode {mode}")
    del probs16, d
    gc.collect()
    feats = post.finalize(post.merge_within_markup(feats))
    meta["n_feats"] = len(feats)
    return feats, meta


def to_line(f: dict, fid: int, group: str) -> str:
    from shapely.geometry import mapping
    g = mapping(f["geom"])
    g["coordinates"] = [[[round(x, 2), round(y, 2)] for x, y, *_ in ring] for ring in g["coordinates"]]
    props = {"fid": fid, "class_name": f["class_name"], "confidence": round(float(f["confidence"]), 6),
             "region_name": f["region_name"], "crs": "EPSG:3857", "markup_type": "li",
             "original_crs": f.get("original_crs")}
    if group != "li":
        props["threshold_group"] = group
    return json.dumps({"type": "Feature", "geometry": g, "properties": props}, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--src", default=None, help="стек bot-10 work/fam (только чтение)")
    ap.add_argument("--out", default=None, help="папка нового стека (fam_*.jsonl)")
    ap.add_argument("--compact", default=None, help="npz: class/conf/centroid/area по объектам (для сверки)")
    ap.add_argument("--only", default=None, help="подстрока имени файла (дымовой прогон)")
    ap.add_argument("--topk", type=int, default=200000)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.toml"))
    a = ap.parse_args()

    import tomllib
    from expds.candidates import CandidateParams
    from expds.li_decode import params
    from expds.pipeline import _params
    cfg = tomllib.loads(Path(a.config).read_text())
    cand_p = _params(CandidateParams, cfg.get("candidates", {}))
    prm = params(cfg.get("li", {}))

    files = sorted(Path(a.probs).glob("*.npz"))
    if a.only:
        files = [f for f in files if a.only in f.name]
    info = scan_stack(Path(a.src)) if a.src else None
    stats = {"mode": a.mode, "cand_p": vars(cand_p), "prm": prm, "regions": {}, "written": Counter(),
             "dropped": Counter(), "removed": Counter()}
    by_region: dict[str, Path] = {}          # region → временный jsonl новых объектов (удаляется в конце)
    fid = 10_000_000
    comp = defaultdict(list)
    t0 = time.time()
    for fp in files:
        feats, meta = decode_file(fp, a.mode, cand_p, prm, a.topk)
        region = meta["region"]
        cnt = Counter(f["class_name"] for f in feats)
        meta["by_class"] = dict(cnt)
        stats["regions"][region] = meta
        if a.compact:
            for f in feats:
                g = f["geom"]
                c = g.centroid
                comp["region"].append(region); comp["cls"].append(LI_CLASSES.index(f["class_name"]))
                comp["conf"].append(float(f["confidence"])); comp["x"].append(c.x); comp["y"].append(c.y)
                comp["area"].append(g.area)
        if info is not None:
            tmp = Path(a.out) / "_tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            with (tmp / f"{len(by_region):03d}.jsonl").open("w") as fo:
                for f in feats:
                    grp = target_group(info, region, f["class_name"])
                    if grp is None:
                        stats["dropped"][f"{region}|{f['class_name']}"] += 1
                        continue
                    fo.write(to_line(f, fid, grp) + "\n")
                    fid += 1
                    stats["written"][f"{grp}|{f['class_name']}"] += 1
            by_region[region] = tmp / f"{len(by_region):03d}.jsonl"
        print(f"{time.time() - t0:7.1f}s {region} {meta['shape']} res={meta['res']:.2f} {dict(cnt)}", flush=True)
        del feats
        gc.collect()
    if a.compact:
        np.savez_compressed(a.compact, region=np.array(comp["region"]), cls=np.array(comp["cls"], np.int8),
                            conf=np.array(comp["conf"], np.float32), x=np.array(comp["x"]), y=np.array(comp["y"]),
                            area=np.array(comp["area"]))
    if info is not None and a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        src = Path(a.src)
        for fp in sorted(src.glob("*.jsonl")):
            regions_here = set()
            with fp.open() as fh, (out / fp.name).open("w") as fo:
                for line in fh:
                    p = json.loads(line)["properties"]
                    regions_here.add(p["region_name"])
                    if p.get("markup_type") == "li" and group_of(p) in REPLACED:
                        stats["removed"][f"{group_of(p)}|{p['class_name']}"] += 1
                        continue
                    fo.write(line if line.endswith("\n") else line + "\n")
                for region in sorted(regions_here & set(by_region)):
                    tp = by_region.pop(region)
                    with tp.open() as ft:
                        for line in ft:
                            fo.write(line)
                    tp.unlink()
        for region, tp in by_region.items():
            stats["dropped"][f"{region}|no_family_file"] += sum(1 for _ in tp.open())
            tp.unlink()
        (out / "_tmp").rmdir()
        cp = src / "_counts.json"
        if cp.exists():
            (out / "_counts.json").write_text(cp.read_text())
    stats["seconds"] = round(time.time() - t0, 1)
    for k in ("written", "dropped", "removed"):
        stats[k] = dict(stats[k])
    sp = (Path(a.out) / "_li_real_stats.json") if a.out else Path(str(a.compact or "li_real") + ".stats.json")
    sp.write_text(json.dumps(stats, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: stats[k] for k in ("written", "removed", "seconds")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
