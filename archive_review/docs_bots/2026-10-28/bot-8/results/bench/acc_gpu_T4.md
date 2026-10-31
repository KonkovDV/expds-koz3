# §7.12 (5) — замер на T4 (для бота 9; бот 8 его не запускал, GPU на box нет)

Нужны: `bench/bench_acc_gpu.py` и из ветки `bot8-cpu-a-speedup` (worktree `bot-8/wt-cpua`) папка `expds/li_a/`
(файлы `__init__.py hm.py hm_acc.py expert.py relief.py dtm.py decode.py`; код EhimenNathan — приватный, только в
приватный Kaggle Dataset, как в §7.10). torch с CUDA, numpy, scipy; для `--net unet` — segmentation_models_pytorch.

```bash
cd <dir with bench_acc_gpu.py>; R=<dir containing expds/>
nvidia-smi --query-gpu=name,clocks.sm,power.draw --format=csv -lms 500 > acc_clocks.csv & S=$!
# 1) только смешивание (сеть-заглушка), номинал 30 Мпкс и 75 Мпкс, TTA d8 и 1 вид
python bench_acc_gpu.py --repo $R --dev cuda --px 5480 5480 --net stub --tta d8 --out acc_t4_stub_30M_d8.json
python bench_acc_gpu.py --repo $R --dev cuda --px 5480 5480 --net stub --tta 1  --out acc_t4_stub_30M_1.json
python bench_acc_gpu.py --repo $R --dev cuda --px 8660 8660 --net stub --tta 1  --out acc_t4_stub_75M_1.json
# 2) с реальной по размеру сетью (случайные веса, resnet34-Unet), 30 Мпкс, 1 вид, 1 повтор — перекрытие с GPU
python bench_acc_gpu.py --repo $R --dev cuda --px 5480 5480 --net unet --tta 1 --reps 1 --out acc_t4_unet_30M_1.json
kill $S
```
Сверить в JSON: `modes.device_exact.byte_equal == true`; у `device` — `max_abs` ≤ 1e-6 и `flips` = 0; `x_wall`
(host / режим) и `peak_vram_mb` (ожидание [оц]: 30 Мпкс ≈ 1,0–1,5 ГБ сверх сети, 75 Мпкс ≈ 2,5–3,5 ГБ; хост-RAM
до ≈ 8 ГБ на 75 Мпкс из-за x9+x12 и np.stack в режиме host). Если OOM на 75M — код сам переходит на хост-аккумуляторы
(в логе «falling back to host accumulators»), это тоже результат.
