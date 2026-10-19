from expds.naming import class_of_annotation, markup_type_of, modality_of, site_family


def test_cyrillic_ae_is_recognized():
    # «Ае» набрано кириллицей — так в 7 файлах разметки трейна
    name = "Юшково_Ае_курганы_поврежденные.geojson"
    assert modality_of(name) == "Ae"
    assert markup_type_of(name) == "ae"
    assert class_of_annotation(name) == "kurgany_povrezhdennye"


def test_codes_and_classes():
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


def test_site_family():
    assert site_family("209_Гнёздово_Терраса_1") == "Гнёздово"
    assert site_family("203_Семибратнее_Городище_3") == "Семибратнее"
    assert site_family("region_0001") == "region"
