# PIXGBM на ПК — статус от bot-9 (28.10, 23:10 МСК)
1. **Пороги *_pix на всём трейне** — ГОТОВО (на коробке, ПК был занят; тот же M.sweep_groups, что в eval_pix.kept_mask, но curve_where по ВСЕМ участкам; вариант S, вход pix_lofo.geojson; global и site совпали):
   ```
   [thresholds_by_markup.maggr_pix]
   finds_points = 0.902915
   object_poly = 0.684725
   [thresholds_by_markup.etgr_pix]
   object_poly = 0.545825
   [thresholds_by_markup.terfl_pix]
   object_poly = 0.746421
   ```
   Скрипт pc/fit_thr.py, лог pc/fit_thr.log, json pc/thr_pix_full.json. Оговорка: пороги подобраны на LOFO-кандидатах (модели без своей семьи), а в сдаче — модель all (на трейне выше уверенность) → пороги, вероятно, чуть занижены для all.
2. **Модели** — models/all/{poly,finds}.txt скопированы в C:\Sloi\wt_pix\models\pixgbm\all (ветка bot3-pixgbm HEAD 7cdace5, включает union_name_re; по умолчанию выкл.).
3. **Тест-кандидаты** — тестовых данных на ПК и коробке нет (тест только в ядре Kaggle, его не трогаю). Вместо этого в очереди ПК (после lidar59, F: один читатель): geo_predict variant S (enabled, replace, model_dir all, пороги выше) по всему трейну F:\2\train-expds-3 → C:\Sloi\pix\geo_pix_S_train.geojson — проверка хука, время и пик RAM на всех гео-участках (на трейне это утечка, для скоринга НЕ использовать). Результаты допишу сюда и в bot-9/PC_RESULTS.md.
