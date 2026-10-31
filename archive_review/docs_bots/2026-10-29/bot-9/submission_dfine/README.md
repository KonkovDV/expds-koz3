# submission_dfine — самостоятельный модуль инференса D-FINE SpOr (бот 9)

Подробная инструкция на русском лежит в `../DFINE_SUBMISSION.md`. Ответ боту 7 — в `../FOR_BOT7_V3.md`.

- `dfine_infer.py`: модуль и CLI. Классы `DFineSpOr`, `run_package_regions`. Окна 640 px на 0,5 м в UTM, перекрытие 160, flip-TTA, NMS 0,5; круги R/C или эллипсы E; EPSG:3857, `threshold_group=sp_dfine`.
- `export_fp16.py`: копия весов с тензорами в fp16 (только для экономии архива).
- `wheels/`: офлайн-колёса transformers 5.21.0 и зависимости (`SHA256SUMS`, `requirements-dfine.txt`).
- `thr/`: подбор порогов sp_dfine (`run_thr.sh`, `sweep_R.py`, результат в `sweep.log` и `thr_sweep.json`).
- `smoke/`: CPU-смоук (`run_smoke.sh`, результат в `smoke.json`).

CLI: `python dfine_infer.py --ckpt DIR [--ckpt DIR2 …] --tif A.tif [--tif …] --region NAME --utm UTM.json --out out.geojson [--device cpu|cuda] [--vendor DIR] [--no-flip] [--max-windows N] [--threads N] [--output circles|ellipses]`
