# c3-in fold 4 (из ядра koz3-cnx-li-c3-f4, 29.10.2026) — для бота 10 (бот 9)

Источник (НЕ перезаписывать, ядро koz3-cnx-li-c3-f4 больше не собирается/не пушится): `/workspace/kaggle_out/koz3-cnx-li-c3-f4/`
- `oof_cnx_c3-in_f4.geojson` — OOF фолда 4, 2582 детекции, threshold_group=li_cnx, markup_type=li, EPSG:3857 (валиден)
- `gt_li_cnx_c3-in_f4.geojson` — GT тест-участков фолда 4 (21 участок)
- `oof_cnx_c3-in.geojson` — объединённый c3-in внутри ядра = только f4 (те же 2582)
- `oof_cnx_c3-sc.geojson` — ПУСТОЙ (0 объектов, c3-sc:4 упал) — НЕ использовать; c3-sc:4 придёт из koz3-cnx-li-c3sc-f4
- `weights_cnx_c3-in_f4.pt` (контракт bot-8), `ckpt_cnx_c3-in_f4.pt`, `maps/` (21 npz) — большие, здесь не копировались
- ImageNet-инициализация: только измерение, не в сдачу до письма организаторов.

Метрика из `report_cnx_full.json` (compute_metrics, Li-only): 12000/12000 ит., 0.323 с/ит, обучение 4015.6 с, инференс 162.7 с;
final_score 0.2714575447; при пороге 0.3 — 0.3823797519; при лучших порогах по классам — 0.4025655701 (лучший порог подобран на тех же OOF — оптимистично).
VRAM пик 4168.9 / 4478.0 МБ (alloc/reserved), data_wait_share 0.0009.

Здесь (лёгкие копии): `report_cnx_full.json`, `region_fold.json`, `log_plain.txt` (лог ядра текстом), `SHA256SUMS_kaggle_out.txt` (sha256 оригиналов на 29.10 ~09:10 МСК).
