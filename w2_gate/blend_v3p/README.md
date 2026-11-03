# P-W2-12: мягкий бленд карт lidar ConvNeXt на V3P в прод-контексте v3_on (02.11.2026 11:12 MSK)

**Вердикт: FAIL оба варианта** → линия бленда c3-sc закрыта, GPU не запускались (не нужны).

## Ключевые факты (до замера)
- Все карты c3-sc s0/s1/s2 и r50c5 уже с D8 TTA (tta=8, включает флипы) → «бленд + flip-TTA» == «бленд»; вместо дубля (b) = blend4 (+r50c5).
- Старый s012avg уже был усреднением КАРТ (seed_merge_cnx.py), не детекций; его FAIL (Δ−0,007) был в старом контексте стека A 0,314615.
  blend3 здесь воспроизводит его ровно (n 82996).

## Что измерено
- A = V3P_ctx 0,345852 (как P-W2-9 попр. 2; cfg configs/default.toml из koz3_v3on_b447649_6d4764f8_w6dfd2e07.zip — байт-в-байт = b505656). S0 V3P 0,3337867244 ok.
- B = A + детекции бленда, группа li_cnxblend (output_floors по умолчанию 0,15/0,20). Движок W4 v3_gate_lib / d3_gate TableEngine, honest-10 по семействам (nested), paired bootstrap 2000 по site_family, seed 0.
- Карты: среднее вероятностей hm/mask и сырых wh/off по моделям, valid = AND, потом один декод (копия kernel decode, peak_k 11, min_score 0,05, CIRCLE_R 0,5).

| вариант | n (ctx) | B | Δ | P | CI90 | new34 | nolidar | полы (povr/tselye/gor/fort) | |
|---|---|---|---|---|---|---|---|---|---|
| (a) blend3 s0+s1+s2 | 82996 (14693) | 0,342352 | −0,003500 | 0,2295 | [−0,0107, +0,0039] | −0,0042 | 0 | +0,0036 / **−0,0152** / **−0,0079** / **−0,0074** | FAIL |
| (b) blend4 +r50c5 | 62845 (10689) | 0,345546 | −0,000305 | 0,4765 | [−0,0089, +0,0083] | **−0,0153** | +0,00004 | +0,0081 / **−0,0157** / +0,0043 / −0,0048 | FAIL |
| справ. одиночный s2 | 94959 (16459) | 0,334894 | −0,010958 | 0,002 | | −0,0167 | −0,0002 | | FAIL |
| справ. blend3 vs s2 | | | +0,007458 | 0,9965 | | +0,0125 | | | |
| справ. blend4 vs s2 | | | +0,010652 | 0,988 | | +0,0013 | | | |

Вывод: бленд действительно даёт +0,007…+0,011 к одиночной модели (ожидание +0,01–0,02 в этом смысле подтверждается), но на V3P_ctx лидарный ConvNeXt-слой
ничего не добавляет сверх уже имеющегося li-слоя стека; стабильно ломается kurgany_tselye (−0,015). Пороги P≥0,95 (Бонферрони ×2) далеко.

## Валидация OOF (`validate_oof.py` → oof_validation.json): OK
5/5 фолдов, fold = region_fold (r50c5 = s2 ядра, 59 участков), семейства не разбиты, вне фолда 0; без детекций только 182_Красносельская_1.7км (как s0/s1/s2).
Контроль декодера: s2 из карт → 94959 = OOF ядра s2, совпадение по каждому участку; 92/94959 различий conf в 3-м знаке (округление f16/u8).

## Как повторить
```
W=/workspace/koz3/w2; PY=/workspace/koz3/venv/bin/python
nice -n 10 $PY w2_tools/blend_maps_cnx.py --report $W/oof/s2/koz3-w2-cnx-c3sc-s2f4/report_cnx_full.json --group li_cnxblend \
  --member s0="$W/maps/s0/**/maps_cnx_*.npz" --member s1="$W/maps/s1/**/maps_cnx_*.npz" --member s2="$W/maps/s2/**/maps_cnx_*.npz" \
  [--member r50c5="$W/blend/r50c5_maps/maps_cnx_*.npz"] --out $W/blend/blend3.geojson     # ~1,5 мин CPU
cd w2_gate/blend_v3p && nice -n 10 /workspace/koz3/w4venv/bin/python gate_blend_ctx.py 2000 gate_blend_ctx.json   # ~50 с, ~2,5 ГБ
```
`v3on_src/configs/default.toml` положить рядом (извлечь из архива zip выше; в git не кладётся). Точные glob'ы членов — в blend3.log/blend4.log на боксе.

## Где лежит
- Бокс: /workspace/koz3/w2/blend/ (blend3/blend4/check_s2 .geojson + .json sidecar, логи, r50c5_maps — симлинки f01 elenakonkova + f234 dmitrijkv, копия gate/README).
- PREREG: orch_bot-W2/PREREG_W2_cnx.md, P-W2-12 (d338318, до замера).
