## Связанные источники (не эти датасеты)
| Источник | Что | Зачем |
|---|---|---|
| `elenakonkova/koz3-cnx-li-prep` | рельеф `cnx_relief/<участок>.npy` (каналы lrm_s, lrm_l, slope, svf_r, opp_r, uint8) + `.json` (геопривязка), 119 файлов, 1,6 ГБ; ядро dmitrijkv/koz3-cnx-li-prep | вход всех c3-sc/r50c5 ядер и `infer_cpu.py` (c3 берёт первые 3 канала, r50c5 — все 5) |
| `aaaa0000004/koz3-w2-gaudit-las` | LAS 015/021/186/190 (1–6 ГБ каждый), `*_Li_{gorod,povr,tselye}.geojson`, `*_UTM.json`; 16 файлов | аудит классов земли (CI2-3/P-W2-8, закрыт) и DEM fill; доступен только аккаунту aaaa0000004 |
| S3 `train-expds-3` (`https://s3.ru-7.storage.selcloud.ru`) | все LAS, карты/разметка Li | первоисточник; ключи — приватный датасет elenakonkova/koz3-s3-cred (не печатать), загрузка `w2_tools/s3get.py` |
| `elenakonkova/koz3-w2-demfill` | (появится после P-W2-10) рельеф/карты/итог DEM fill | см. HANDOFF, п. 8 |
