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
    assert all(f["markup_type"] == "ae" for f in out)
    # Копии мелкой сетки — группа ae, копия грубой сетки — li_ae, как в табло 0,353793.
    assert sorted(f["threshold_group"] for f in out) == ["ae", "ae", "ae", "li_ae", "li_ae"]
    zero = {fusion.POVR: 0.0, fusion.TSEL: 0.0}
    kept = post.apply_thresholds(feats + out, {}, 0.15, {"ae": zero, "li_ae": zero})
    assert sum(f["markup_type"] == "ae" for f in kept) == 5
    assert len(post.merge_within_markup(kept)) == len(kept)   # слои не схлопываются друг с другом
    assert fusion.li_to_ae(feats, [], fusion.FusionParams()) == []


def test_local_cs_photo_copies_the_same_kurgans_as_the_utm_photo(tmp_path):
    """Аэрофото с безымянной LOCAL_CS и метрами зоны даёт те же копии, что исходный снимок в UTM."""
    from rasterio.crs import CRS

    utm = write_ae(tmp_path / "utm.tif")
    local = 'LOCAL_CS["unnamed",UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    loc = tmp_path / "local.tif"
    with rasterio.open(utm) as ds:
        a, tr = ds.read(1), ds.transform
    with rasterio.open(loc, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype="uint8",
                       transform=tr, crs=CRS.from_wkt(local)) as out:
        out.write(a, 1)
    feats = kurgan(X0 + 20, Y0 + 50, 0.995, 0.004)
    ref = fusion.li_to_ae(feats, [utm])
    got = fusion.li_to_ae(feats, [loc], fallback_crs=UTM)
    assert [(f["class_name"], round(f["confidence"], 3), f["threshold_group"]) for f in got] == [
        (f["class_name"], round(f["confidence"], 3), f["threshold_group"]) for f in ref]
    assert ref and fusion.li_to_ae(feats, [loc]) == []


def test_mosaic_copies_keep_their_own_group(tmp_path):
    """Копия кургана мозаики — группа li_ae, копия мелкой сетки — ae. Без этого целые копии
    мелкой сетки уходили под порог li_ae 1,000001 и выпадали все."""
    ae = write_ae(tmp_path / "ae.tif")
    feats = (kurgan(X0 + 20, Y0 + 50, 0.995, 0.004)
             + kurgan(X0 + 35, Y0 + 60, 0.997, 0.001, group="li_mosaic"))
    out = fusion.li_to_ae(feats, [ae])
    by_group = sorted((f["threshold_group"], f["class_name"]) for f in out)
    assert by_group == [("ae", fusion.POVR), ("ae", fusion.TSEL),
                        ("li_ae", fusion.POVR), ("li_ae", fusion.TSEL)]
    taus = {"ae": {fusion.POVR: 0.99053, fusion.TSEL: 0.802255},
            "li_ae": {fusion.POVR: 0.993061, fusion.TSEL: 1.000001}}
    kept = post.apply_thresholds(out, {}, 0.15, taus)
    assert sorted((f["threshold_group"], f["class_name"]) for f in kept) == [
        ("ae", fusion.POVR), ("ae", fusion.TSEL), ("li_ae", fusion.POVR)]
