# Gap breakdown (OOF, diagnostic). stack 0.321981; oracle_hit 0.690147; oracle_bip 0.690842

Definitions: see json 'definition'. Δ = weighted F1 change when only that loss is removed (others fixed); not additive.

| lever | global | lidar_old | 34new | optical |
|---|---|---|---|---|
| a_fn_nocand | +0.1728 | +0.0308 | +0.0120 | +0.1357 |
| c_fn_wrong_class | +0.0947 | +0.0117 | +0.0077 | +0.0778 |
| fp_pure | +0.1111 | +0.0208 | +0.0205 | +0.0450 |
| b1_fn_threshold | +0.1727 | +0.0776 | +0.0204 | +0.0826 |
| b2_fn_assign_dedup | +0.0159 | +0.0053 | +0.0019 | +0.0087 |
| c_dup_crossmod | +0.0046 | +0.0026 | +0.0016 | +0.0004 |
| c_dup_samemod | +0.0144 | +0.0022 | +0.0005 | +0.0117 |
| d_geometry | +0.0419 | +0.0283 | +0.0021 | +0.0118 |

## class × modality (Δ per lever)
| class (w) | mod | a_fn_nocand | c_fn_wrong_class | fp_pure | b1_fn_threshold | b2_fn_assign_dedup | c_dup_crossmod | c_dup_samemod | d_geometry |
|---|---|---|---|---|---|---|---|---|---|
| arkhitektury (10.0) | lidar | +0.0000 | +0.0002 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| arkhitektury (10.0) | 34new | +0.0000 | +0.0000 | +0.0018 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| arkhitektury (10.0) | optical | +0.0375 | +0.0092 | +0.0022 | +0.0151 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| finds_points (10.0) | lidar | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| finds_points (10.0) | 34new | +0.0011 | +0.0002 | +0.0019 | +0.0000 | +0.0005 | +0.0000 | +0.0000 | +0.0005 |
| finds_points (10.0) | optical | +0.0260 | +0.0022 | +0.0019 | +0.0225 | +0.0046 | +0.0000 | +0.0000 | +0.0078 |
| fortifikatsii (5.0) | lidar | +0.0021 | +0.0059 | +0.0008 | +0.0010 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| fortifikatsii (5.0) | 34new | +0.0031 | +0.0026 | +0.0004 | +0.0057 | +0.0001 | +0.0000 | +0.0000 | +0.0000 |
| fortifikatsii (5.0) | optical | +0.0036 | +0.0016 | +0.0032 | +0.0021 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| gorodishcha (15.0) | lidar | +0.0000 | +0.0028 | +0.0038 | +0.0028 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| gorodishcha (15.0) | 34new | +0.0000 | +0.0028 | +0.0096 | +0.0065 | +0.0009 | +0.0015 | +0.0000 | +0.0000 |
| gorodishcha (15.0) | optical | +0.0028 | +0.0422 | +0.0037 | +0.0118 | +0.0009 | +0.0000 | +0.0000 | +0.0000 |
| kurgany_povrezhdennye (30.0) | lidar | +0.0138 | +0.0021 | +0.0062 | +0.0377 | +0.0045 | +0.0018 | +0.0012 | +0.0180 |
| kurgany_povrezhdennye (30.0) | 34new | +0.0036 | +0.0014 | +0.0062 | +0.0042 | +0.0003 | +0.0001 | +0.0003 | +0.0005 |
| kurgany_povrezhdennye (30.0) | optical | +0.0308 | +0.0056 | +0.0262 | +0.0215 | +0.0000 | +0.0002 | +0.0000 | +0.0028 |
| kurgany_tselye (20.0) | lidar | +0.0148 | +0.0007 | +0.0100 | +0.0361 | +0.0008 | +0.0009 | +0.0010 | +0.0103 |
| kurgany_tselye (20.0) | 34new | +0.0043 | +0.0003 | +0.0006 | +0.0040 | +0.0000 | +0.0001 | +0.0002 | +0.0011 |
| kurgany_tselye (20.0) | optical | +0.0193 | +0.0083 | +0.0042 | +0.0081 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| object_poly (10.0) | lidar | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| object_poly (10.0) | 34new | +0.0000 | +0.0003 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| object_poly (10.0) | optical | +0.0158 | +0.0086 | +0.0036 | +0.0017 | +0.0032 | +0.0002 | +0.0116 | +0.0012 |

## counts class × modality
| class | mod | fn_wrong_kept | fn_wrong_low | tp | fp | ngt | fp_pure | fp_nonpoly | fp_dup_x | fp_dup_same | fp_geom | fn_thr | fn_assign | fn_geom | fn_nocand | or_hit | or_bip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arkhitektury | 34new | 0 | 0 | 0 | 551 | 0 | 551 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| arkhitektury | lidar | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| arkhitektury | optical | 23 | 28 | 25 | 638 | 406 | 636 | 0 | 0 | 2 | 0 | 86 | 0 | 0 | 244 | 111 | 111 |
| finds_points | 34new | 0 | 6 | 142 | 447 | 216 | 433 | 0 | 0 | 10 | 4 | 0 | 17 | 15 | 36 | 142 | 142 |
| finds_points | optical | 34 | 39 | 943 | 471 | 3262 | 420 | 0 | 0 | 6 | 45 | 841 | 151 | 256 | 998 | 1677 | 1678 |
| fortifikatsii | 34new | 18 | 36 | 62 | 296 | 306 | 290 | 0 | 0 | 6 | 0 | 125 | 3 | 0 | 65 | 182 | 182 |
| fortifikatsii | lidar | 2 | 126 | 10 | 512 | 207 | 493 | 0 | 0 | 19 | 0 | 21 | 0 | 0 | 45 | 34 | 34 |
| fortifikatsii | optical | 8 | 26 | 29 | 1487 | 183 | 1478 | 0 | 0 | 9 | 0 | 43 | 0 | 0 | 77 | 72 | 72 |
| gorodishcha | 34new | 1 | 2 | 25 | 151 | 36 | 127 | 0 | 24 | 0 | 0 | 7 | 1 | 0 | 0 | 31 | 31 |
| gorodishcha | lidar | 1 | 2 | 2 | 58 | 9 | 58 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 6 | 6 |
| gorodishcha | optical | 21 | 32 | 4 | 57 | 73 | 57 | 0 | 0 | 0 | 0 | 13 | 1 | 0 | 3 | 15 | 15 |
| kurgany_povrezhdennye | 34new | 0 | 22 | 63 | 1035 | 219 | 963 | 0 | 16 | 50 | 6 | 66 | 4 | 7 | 56 | 130 | 130 |
| kurgany_povrezhdennye | lidar | 2 | 31 | 836 | 1504 | 2076 | 966 | 0 | 289 | 194 | 55 | 632 | 71 | 284 | 221 | 1398 | 1407 |
| kurgany_povrezhdennye | optical | 3 | 85 | 336 | 3451 | 1325 | 3409 | 0 | 30 | 7 | 5 | 348 | 0 | 44 | 509 | 665 | 665 |
| kurgany_tselye | 34new | 0 | 3 | 35 | 51 | 119 | 36 | 0 | 3 | 10 | 2 | 35 | 0 | 9 | 37 | 70 | 70 |
| kurgany_tselye | lidar | 0 | 6 | 524 | 698 | 1103 | 540 | 0 | 51 | 59 | 48 | 350 | 7 | 83 | 133 | 867 | 868 |
| kurgany_tselye | optical | 19 | 54 | 35 | 241 | 355 | 241 | 0 | 0 | 0 | 0 | 71 | 0 | 0 | 176 | 106 | 106 |
| object_poly | 34new | 5 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| object_poly | optical | 86 | 52 | 584 | 1140 | 1078 | 291 | 0 | 15 | 814 | 20 | 26 | 49 | 14 | 267 | 609 | 609 |

## top-25 site cells by max lever Δ
| class | mod | site | tp | fp | ngt | best lever | Δ |
|---|---|---|---|---|---|---|---|
| kurgany_povrezhdennye | lidar | 101_Гнёздово_Днепровская | 9 | 1 | 263 | b1_fn_threshold | +0.0132 |
| finds_points | optical | 211_Гнёздово_Терраса_3 | 0 | 0 | 409 | a_fn_nocand | +0.0116 |
| arkhitektury | optical | 093_Старый_Барах | 0 | 0 | 69 | a_fn_nocand | +0.0106 |
| kurgany_povrezhdennye | lidar | 036_Тимерево | 336 | 237 | 777 | b1_fn_threshold | +0.0099 |
| arkhitektury | optical | 056_Фалхан | 0 | 0 | 53 | a_fn_nocand | +0.0084 |
| arkhitektury | optical | 008_Герите | 3 | 27 | 45 | a_fn_nocand | +0.0077 |
| object_poly | optical | 212_Гнёздово_Усадьба | 107 | 73 | 259 | a_fn_nocand | +0.0070 |
| kurgany_povrezhdennye | optical | 043_Гочево | 0 | 91 | 105 | a_fn_nocand | +0.0065 |
| kurgany_tselye | lidar | 180_Гнёздово_Лесная_2 | 148 | 159 | 254 | b1_fn_threshold | +0.0063 |
| kurgany_tselye | optical | 060_Сельцо | 0 | 0 | 54 | a_fn_nocand | +0.0062 |
| finds_points | optical | 202_Семибратнее_Городище_2 | 197 | 44 | 518 | b1_fn_threshold | +0.0059 |
| fortifikatsii | lidar | 004_Боровичи_1 | 3 | 9 | 148 | c_fn_wrong_class | +0.0056 |
| gorodishcha | optical | 185_Изборск | 0 | 0 | 6 | c_fn_wrong_class | +0.0056 |
| finds_points | optical | 207_Семибратнее_Городище_7 | 71 | 48 | 427 | a_fn_nocand | +0.0054 |
| object_poly | optical | 214_Гнёздово_Лесная_2 | 126 | 434 | 177 | c_dup_samemod | +0.0052 |
| kurgany_tselye | lidar | 030_Эсбю | 111 | 63 | 181 | b1_fn_threshold | +0.0049 |
| kurgany_povrezhdennye | optical | 185_Изборск | 3 | 30 | 89 | a_fn_nocand | +0.0045 |
| finds_points | optical | 205_Семибратнее_Городище_5 | 143 | 54 | 454 | b1_fn_threshold | +0.0044 |
| kurgany_tselye | lidar | 179_Гнёздово_Лесная_1 | 38 | 44 | 92 | b1_fn_threshold | +0.0043 |
| finds_points | optical | 203_Семибратнее_Городище_3 | 200 | 51 | 438 | b1_fn_threshold | +0.0042 |
| kurgany_povrezhdennye | lidar | 102_Гнёздово_Заольшанская | 0 | 0 | 136 | d_geometry | +0.0042 |
| arkhitektury | optical | 164_Кисловодская_котловина_1 | 0 | 0 | 49 | c_fn_wrong_class | +0.0041 |
| kurgany_tselye | lidar | 190_Муравлево | 44 | 11 | 80 | b1_fn_threshold | +0.0039 |
| finds_points | optical | 201_Семибратнее_Городище_1 | 4 | 2 | 155 | a_fn_nocand | +0.0039 |
| kurgany_tselye | lidar | 011_Эрентуна | 10 | 7 | 42 | b1_fn_threshold | +0.0037 |

## FN losses by class×modality (honest-10 wF1 points if fixed; a=no cand, b1=below thr, b2=assign/dedup, c=wrong class, d=geometry)
|cell|cat|pts|n GT|total GT|
|-|-|-|-|-|
|gorodishcha|optical|c_fn_wrong_class|+0.0422|53|73|
|kurgany_povrezhdennye|lidar|b1_fn_threshold|+0.0377|632|2076|
|arkhitektury|optical|a_fn_nocand|+0.0375|244|406|
|kurgany_tselye|lidar|b1_fn_threshold|+0.0361|350|1103|
|kurgany_povrezhdennye|optical|a_fn_nocand|+0.0308|509|1325|
|finds_points|optical|a_fn_nocand|+0.0260|998|3262|
|finds_points|optical|b1_fn_threshold|+0.0225|841|3262|
|kurgany_povrezhdennye|optical|b1_fn_threshold|+0.0215|348|1325|
|kurgany_tselye|optical|a_fn_nocand|+0.0193|176|355|
|kurgany_povrezhdennye|lidar|d_geometry|+0.0180|284|2076|
|object_poly|optical|a_fn_nocand|+0.0158|267|1078|
|arkhitektury|optical|b1_fn_threshold|+0.0151|86|406|
|kurgany_tselye|lidar|a_fn_nocand|+0.0148|133|1103|
|kurgany_povrezhdennye|lidar|a_fn_nocand|+0.0138|221|2076|
|gorodishcha|optical|b1_fn_threshold|+0.0118|13|73|
|kurgany_tselye|lidar|d_geometry|+0.0103|83|1103|
|arkhitektury|optical|c_fn_wrong_class|+0.0092|51|406|
|object_poly|optical|c_fn_wrong_class|+0.0086|138|1078|
|kurgany_tselye|optical|c_fn_wrong_class|+0.0083|73|355|
|kurgany_tselye|optical|b1_fn_threshold|+0.0081|71|355|
|finds_points|optical|d_geometry|+0.0078|256|3262|
|gorodishcha|34new|b1_fn_threshold|+0.0065|7|36|
|fortifikatsii|lidar|c_fn_wrong_class|+0.0059|128|207|
|fortifikatsii|34new|b1_fn_threshold|+0.0057|125|306|
|kurgany_povrezhdennye|optical|c_fn_wrong_class|+0.0056|88|1325|
|finds_points|optical|b2_fn_assign_dedup|+0.0046|151|3262|
|kurgany_povrezhdennye|lidar|b2_fn_assign_dedup|+0.0045|71|2076|
|kurgany_tselye|34new|a_fn_nocand|+0.0043|37|119|
|kurgany_povrezhdennye|34new|b1_fn_threshold|+0.0042|66|219|
|kurgany_tselye|34new|b1_fn_threshold|+0.0040|35|119|
|fortifikatsii|optical|a_fn_nocand|+0.0036|77|183|
|kurgany_povrezhdennye|34new|a_fn_nocand|+0.0036|56|219|
|object_poly|optical|b2_fn_assign_dedup|+0.0032|49|1078|
|fortifikatsii|34new|a_fn_nocand|+0.0031|65|306|
|kurgany_povrezhdennye|optical|d_geometry|+0.0028|44|1325|
|gorodishcha|lidar|b1_fn_threshold|+0.0028|3|9|
|gorodishcha|lidar|c_fn_wrong_class|+0.0028|3|9|
|gorodishcha|34new|c_fn_wrong_class|+0.0028|3|36|
|gorodishcha|optical|a_fn_nocand|+0.0028|3|73|
|fortifikatsii|34new|c_fn_wrong_class|+0.0026|54|306|
|finds_points|optical|c_fn_wrong_class|+0.0022|73|3262|
|fortifikatsii|lidar|a_fn_nocand|+0.0021|45|207|
|kurgany_povrezhdennye|lidar|c_fn_wrong_class|+0.0021|33|2076|
|fortifikatsii|optical|b1_fn_threshold|+0.0021|43|183|
|object_poly|optical|b1_fn_threshold|+0.0017|26|1078|
|fortifikatsii|optical|c_fn_wrong_class|+0.0016|34|183|
|kurgany_povrezhdennye|34new|c_fn_wrong_class|+0.0014|22|219|
|object_poly|optical|d_geometry|+0.0012|14|1078|
|finds_points|34new|a_fn_nocand|+0.0011|36|216|
|kurgany_tselye|34new|d_geometry|+0.0011|9|119|
|fortifikatsii|lidar|b1_fn_threshold|+0.0010|21|207|
|gorodishcha|34new|b2_fn_assign_dedup|+0.0009|1|36|
|gorodishcha|optical|b2_fn_assign_dedup|+0.0009|1|73|
|kurgany_tselye|lidar|b2_fn_assign_dedup|+0.0008|7|1103|
|kurgany_tselye|lidar|c_fn_wrong_class|+0.0007|6|1103|
|finds_points|34new|b2_fn_assign_dedup|+0.0005|17|216|
|kurgany_povrezhdennye|34new|d_geometry|+0.0005|7|219|
|finds_points|34new|d_geometry|+0.0005|15|216|
|kurgany_tselye|34new|c_fn_wrong_class|+0.0003|3|119|
|object_poly|34new|c_fn_wrong_class|+0.0003|5|5|
|kurgany_povrezhdennye|34new|b2_fn_assign_dedup|+0.0003|4|219|
|arkhitektury|lidar|c_fn_wrong_class|+0.0002|1|1|
|finds_points|34new|c_fn_wrong_class|+0.0002|6|216|
|fortifikatsii|34new|b2_fn_assign_dedup|+0.0001|3|306|

Note: c (wrong class) is an upper bound. For gor, the other-class ok-pair candidates are mostly kurgans/arch/forts inside the hillfort polygon (Counter g_x: PO 102, TS 86, arkh 50, fort 41), not a relabelled same object.
Context: stack A+M_rp+ORG3 (variants/NST/m0_O), honest-10 0.3220, not V3P.
