"""Слияние li→ae: копии только уверенных курганов нейросети и только в покрытии снимка."""
import numpy as np
import rasterio
from conftest import UTM, X0, Y0
from rasterio.transform import from_origin
from shapely.geometry import Point

from expds import fusion, post
from expds.crs import to_crs


def write_ae(path):
    a = np.zeros((200, 200), np.uint8)
    a[:, :100] = 128                                      # снимок покрывает только западную половину
    with rasterio.open(path, "w", driver="GTiff", height=200, width=200, count=1, dtype="uint8",
                       transform=from_origin(X0, Y0 + 100, 0.5, 0.5), crs=UTM) as ds:
        ds.write(a, 1)
    return path


def kurgan(x, y, pp, pt, group=None):
    g = to_crs(Point(x, y).buffer(2.0), UTM)
    out = [{"geom": g, "class_name": c, "confidence": s, "region_name": "R", "markup_type": "li", "source": "a.las"}
           for c, s in ((fusion.POVR, pp), (fusion.TSEL, pt))]
    return [dict(f, threshold_group=group) if group else f for f in out]


def test_copies_follow_gates_coverage_and_class_shift(tmp_path):
    ae = write_ae(tmp_path / "ae.tif")
    feats = (kurgan(X0 + 20, Y0 + 50, 0.995, 0.004)        # уверенный повреждённый в покрытии → обе копии
             + kurgan(X0 + 25, Y0 + 20, 0.5, 0.4)           # только «целый»: 0.5 + 0.4 ≥ 0.8
             + kurgan(X0 + 30, Y0 + 80, 0.3, 0.2)           # слабый — без копий
             + kurgan(X0 + 80, Y0 + 50, 0.999, 0.001)       # вне покрытия снимка
             + kurgan(X0 + 15, Y0 + 40, 0.996, 0.002, group="li_coarse")  # крупная сетка копируется
             + kurgan(X0 + 10, Y0 + 10, 0.999, 0.001, group="li_cpu"))   # запасной детектор не копируется
    out = fusion.li_to_ae(feats, [ae])
    got = sorted((f["class_name"], round(f["confidence"], 3)) for f in out)
    assert got == [(fusion.POVR, 0.995), (fusion.POVR, 0.996),
                   (fusion.TSEL, 0.9), (fusion.TSEL, 0.998), (fusion.TSEL, 0.999)]
    assert all(f["markup_type"] == "ae" and f["threshold_group"] == "li_ae" for f in out)
    kept = post.apply_thresholds(feats + out, {}, 0.15, {"li_ae": {fusion.POVR: 0.0, fusion.TSEL: 0.0}})
    assert sum(f["markup_type"] == "ae" for f in kept) == 5
    assert len(post.merge_within_markup(kept)) == len(kept)   # слои не схлопываются друг с другом
    assert fusion.li_to_ae(feats, [], fusion.FusionParams()) == []
