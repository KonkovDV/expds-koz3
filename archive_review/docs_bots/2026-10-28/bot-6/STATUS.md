# bot-6 STATUS — канал оптики / аэрофото 1940-х
- 19:2x МСК: прочитан COMMON.md; копия кода: bot-6/repo, ветка bot6-optics (от 24bd916).
- Источник «0,09 / 0,013»: docs/LB_LOG.md — справочный блок by_data_type «аэрофото» 0,090909 / 0,013333 (мин. целый набор 75 эталонов, 1 TP, 20 FP — scripts/lb_counts.py).
- Что выдаёт ae в сдаче: configs/default.toml [models] ae = ae_fort_mask_r18.pt (только fortifikatsii) + копии курганов лидара li_to_ae (группы ae / li_ae). Своей сети курганов на аэрофото нет.
- Растров в /workspace/data/train-expds-3 пока нет (только geojson разметки и UTM.json) — проверки растров невозможны; делаю проверку сдвига слоёв Ae vs Li по эталону.
- Сделано: work/ae_audit.json (курганы ae 1620, достижимо копией li 592, 545 на участках без li), work/layer_shift_gt*.json (сдвиг li→ae: устойчив у 4 из 11, 608→631 из 1276).
- Патч: ветка bot6-optics (5618d13), bot-6/bot6-optics.diff — scripts/ae_layer_audit.py. Итог: REPORT.md. ГОТОВО.
- 20:5x МСК: начаты проверки растров. Пришло 7 растров ae (001,005,012,015,017,019,020; zstd, xfer_state.json), ортофото нет. Скрипт work/raster/meta.py написан, прогон прерван — ПАУЗА по запросу бота «1» (аудит типов данных → TYPES.md). Продолжить с meta.py.
- 20:5x МСК: TYPES.md (частично, п.1–3) записан; дальше — размеры полигонов vs GT, потеря классов/типов, затем растры.
- 21:2x МСК: twins перепроверен честно (A=0,314615 воспроизведён). F +0,00285 [0,00061; 0,00581], П1–П4 да, П5 пакет. TYPES.md §5. Порядок дальше: COPY → LOCALIZE. copy_model.py запущен 21:23.
- 21:4x МСК: по запросу бота «1»: сверка с bot-7, аудит twins-only, ветка bot6-twins-only (ad8ae9b) + bot6-twins-only.diff, точный замер после порогов (work/twins/twins_exact.*) → TYPES.md §6. COPY: copy_model.py готов (work/copy/copy_model.log), COPY.md — следующим.
- 21:5x МСК: COPY.md (закрыто ботом «1»: модель −0,0020). LOCALIZE: localize.py запущен 21:47.
22:29 МСК: LOCALIZE.md готов. Точный пост-порог: лучший сверх bot-7 — fort 0,5 м +0,00026 (П1 нет, пакет); gor −0,0015; привязка Li/DEM отрицательна. Ветка bot6-localize 6b58d99, bot6-localize.diff.
22:4x МСК: (a) TYPES.md §7: потерь полей, классов и геометрии нет. Целиком недостижимые группы — каждая < 0,005 (arch ae 0,0040; сумма ≈ 0,0074). 211 (0 кандидатов): потолок 0,0106, реально ≈ 0,0007. Гнёздово etgr/terfl — слепая зона честного CV (unseen → drop), а не потеря сдачи. Файлы: work/lost/{geomcheck,lost,cand}.*.
22:4x МСК: (b) ae_layer_audit.py исправлен (cb17611: кириллица «Ае» + пустые фигуры), тест 1 passed, bot6-optics.diff 182 строки. REPORT.md §3/§4 на v3 (курганы ae 1749, достижимо li 595 = 34,0 %; сдвиг 611 → 634 из 1322) + таблица v1/v2/v3.
22:4x МСК: (c) REPORT.md §5: 36 снимков Ae, 32 участка (work/raster/ae_raster_check.*, 366 МБ). 2 снимка в 3857 (015, 029); 7 в int16 (у 106 растяжка ×3,04); у 017/036 чёрные поля не nodata; эталон ae на валидных пикселях 2170/2179. ВСЕ ТРИ ПУНКТА ГОТОВЫ.
22:56 МСК: п.1 GEO.md: честно (LOSO nested) +0,0190 к honest-10 drop [CI +0,0068; +0,0358] = keep-all +0,0186; порог 0,096635 бесполезен; pixgbm +0,0007…+0,0028 к стеку н.з. Далее п.2 (211), п.3 (растры).
23:00 МСК: bot6-twins-only 78add0e: починка масштаба twins; регресс-тест (старый код 0/20); на объектах bot-10 ae_twin 85, sp 113 (work/twinsfix/check.log). bot6-geo 67f84bd: [post] geo_convex_hull, bot6-geo.diff. EXPERIMENTS.md: запись о честной базе 0,333245.
