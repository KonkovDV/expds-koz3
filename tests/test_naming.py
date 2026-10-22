from expds.naming import class_of_annotation, markup_type_of, modality_of, site_family


def test_cyrillic_ae_is_recognized():
    # «Ае» набрано кириллицей — так в 7 файлах разметки трейна
    name = "Юшково_Ае_курганы_поврежденные.geojson"
    assert modality_of(name) == "Ae"
    assert markup_type_of(name) == "ae"
    assert class_of_annotation(name) == "kurgany_povrezhdennye"


def test_cyrillic_ae_inventory_is_not_dropped():
    # Опись files/sklad: эти участки теряются, если «Ае» не перевести в Ae.
    ae = "Ае"
    names = [
        f"Юшково_{ae}_курганы_поврежденные.geojson",
        f"Вышегжа_3_{ae}_курганы_целые.geojson",
        f"Миловидово_{ae}_курганы_поврежденные.geojson",
        f"Перынь_{ae}_архитектуры.geojson",
        f"Шниткино_{ae}_курганы_целые.geojson",
        f"Верхмарево_{ae}_курганы_поврежденные.geojson",
    ]
    assert all(markup_type_of(n) == "ae" for n in names)
    assert class_of_annotation(names[0]) == "kurgany_povrezhdennye"
    assert class_of_annotation(names[3]) == "arkhitektury"
    assert markup_type_of("01_Гнёздово_Нивленское_Li_c.tif") == "li"
    assert markup_type_of("Миловидово_SpOr_курганы_целые.geojson") == "sp_or"
    assert markup_type_of("05_Герите_Or_ортофото.tif") == "or"
    assert markup_type_of("x.las") == "li"
    assert class_of_annotation("Гнёздово_Усадьба_EtGr_профиль_14_ObjectPoly.geojson") == "object_poly"
    assert class_of_annotation("Судьбищи_2_MagGr_FindsPoints.geojson") == "finds_points"
    assert class_of_annotation("Миловидово_SpOr_курганы_целые.geojson") == "kurgany_tselye"
    assert class_of_annotation("Перынь_Ae_городища.geojson") == "gorodishcha"
    assert class_of_annotation("x_MagGr_StructurePoly.geojson") is None


def test_keywords_for_validation_names():
    assert markup_type_of("region_0003/немецкая_аэрофотосъёмка_1.tif") == "ae"
    assert markup_type_of("region_0003/спутник.tif") == "sp_or"
    assert markup_type_of("region_0003/магниторазведка.tif") == "maggr"


def test_scan_region_finds_boundaries_and_noise(tmp_path):
    from expds.catalog import scan_region

    reg = tmp_path / "200_Судьбищи_2"
    (reg / "extra").mkdir(parents=True)
    for name in ("Судьбищи_2_MagGr_границы.geojson", "Судьбищи_2_MagGr_шум.geojson"):
        (reg / "extra" / name).write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    r = scan_region(reg)
    assert [p.name for p in r.boundaries] == ["Судьбищи_2_MagGr_границы.geojson"]
    assert [p.name for p in r.noise] == ["Судьбищи_2_MagGr_шум.geojson"]
    assert not r.skipped


def test_site_named_like_noise_keeps_its_annotations(tmp_path):
    from expds.catalog import scan_region

    reg = tmp_path / "035_Шумгора"
    ann = reg / "06_Шумгора_разметка"
    ann.mkdir(parents=True)
    names = ("Шумгора_Li_курганы_целые.geojson", "Шумгора_SpOr_городища.geojson",
             "Гнёздово_Терраса_3_MagGr_цвет_2_шум.geojson", "Шумгора_EtGr_карта_01_0.2_границы.geojson")
    for name in names:
        (ann / name).write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    r = scan_region(reg)
    assert sorted(p.name for p, _, _ in r.annotations) == sorted(names[:2])
    assert [p.name for p in r.noise] == [names[2]]
    assert [p.name for p in r.boundaries] == [names[3]]


def test_site_family():
    assert site_family("209_Гнёздово_Терраса_1") == "Гнёздово"
    assert site_family("203_Семибратнее_Городище_3") == "Семибратнее"
    assert site_family("region_0001") == "region"
