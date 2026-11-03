"""bot-3: [ae_recall] выключен по умолчанию; целые копии мозаики и дедуп кадров D-FINE."""
from pathlib import Path

import tomllib
from shapely.geometry import Point

from expds import ae_recall, dfine_aeor, fusion, post
from test_fusion import X0, Y0, kurgan, write_ae


def test_flags_default_off():
    cfg = tomllib.loads(Path("configs/default.toml").read_text(encoding="utf-8"))
    assert ae_recall.mosaic_intact_on(cfg) is False
    assert ae_recall.dedup_m(cfg) == 0.0
    assert ae_recall.dedup_m({}) == 0.0
    assert fusion.FusionParams().mosaic_intact is False


def test_mosaic_intact_moves_only_whole_copies(tmp_path):
    ae = write_ae(tmp_path / "ae.tif")
    feats = kurgan(X0 + 35, Y0 + 60, 0.997, 0.001, group="li_mosaic")
    off = fusion.li_to_ae(feats, [ae])
    on = fusion.li_to_ae(feats, [ae], fusion.FusionParams(mosaic_intact=True))
    assert sorted(f["threshold_group"] for f in off) == ["li_ae", "li_ae"]
    by = {f["class_name"]: f["threshold_group"] for f in on}
    assert by[fusion.POVR] == "li_ae"
    assert by[fusion.TSEL] == "ae"
    taus = {"ae": {fusion.TSEL: 0.802255, fusion.POVR: 0.99053},
            "li_ae": {fusion.TSEL: 1.000001, fusion.POVR: 0.993061}}
    kept_off = post.apply_thresholds(off, {}, 0.15, taus)
    kept_on = post.apply_thresholds(on, {}, 0.15, taus)
    assert [f["class_name"] for f in kept_off] == [fusion.POVR]
    assert sorted(f["class_name"] for f in kept_on) == [fusion.POVR, fusion.TSEL]


def test_wbf_halves_a_single_view_nms_keeps_it():
    """OOF ядра koz3-dfine-ae склеивает flip через batched_nms и оставляет сырую уверенность.
    WBF при двух видах делит уверенность бокса, который был только на одном виде."""
    import numpy as np

    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    score = np.array([0.62])
    label = np.array([1])
    views = [(box, score, label), (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))]
    wb, ws, wl = dfine_aeor.fuse_views(views, {"fuse": "wbf", "fuse_iou": 0.55, "nms_iou": 0.5})
    nb, ns, nl = dfine_aeor.fuse_views(views, {"fuse": "nms", "fuse_iou": 0.55, "nms_iou": 0.5})
    assert len(ws) == 1 and abs(float(ws[0]) - 0.31) < 1e-9
    assert len(ns) == 1 and abs(float(ns[0]) - 0.62) < 1e-9
    assert int(wl[0]) == int(nl[0]) == 1
    assert wb.shape == nb.shape == (1, 4)


def test_cross_tif_dedup_keeps_the_higher_score_only_inside_dfine():
    def box(x, conf, group, cls="kurgany_povrezhdennye"):
        return {"geom": Point(x, 0).buffer(2), "class_name": cls, "confidence": conf,
                "region_name": "R", "threshold_group": group, "markup_type": "ae"}

    feats = [box(0, 0.4, "dfine_ae"), box(3, 0.8, "dfine_ae"), box(30, 0.2, "dfine_ae"),
             box(1, 0.9, "ae")]
    same = ae_recall.cross_tif_dedup(feats, 0)
    assert same is feats
    kept = ae_recall.cross_tif_dedup(feats, 4.0)
    confs = sorted(round(f["confidence"], 1) for f in kept)
    assert confs == [0.2, 0.8, 0.9]
    assert [f["threshold_group"] for f in kept if f["threshold_group"] == "ae"] == ["ae"]
