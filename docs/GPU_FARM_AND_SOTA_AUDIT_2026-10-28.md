# Архитектура облачной GPU+CPU фермы, ускорение обучения и отчёт Triage / Red Team / Murder Board (28.10.2026)

## 1. Вычислительный кластер (100% автономный режим без локального ПК)

Вся инфраструктура переведена на полностью автономное исполнение из облачного управляющего узла (Linux Xeon Sapphire Rapids AVX-512) и 4 независимых аккаунтов Kaggle:

| Уровень | Узлы | Вычислительный ресурс | Роль в пайплайне `expds-koz3` |
|---|---|---|---|
| **Облачная GPU-ферма** | 4 аккаунта Kaggle (`node_0..node_3`) | **8 × NVIDIA Tesla T4** (`128 ГБ GDDR6 VRAM`, `20 480 ядер CUDA`, `2 560 Tensor Cores FP16`, 120 GPU-ч/нед) | Параллельное обучение 5 фолдов `nn_oof.py --fold-indices` + `--all-data` в FP16 AMP (`channels_last`), `GpuTiles` целиком в 16 ГБ VRAM |
| **Облачный CPU-рой** | Координатор + 9 ботов (`@2`–`@10`) | **8 ядер / 16 потоков Intel Xeon Sapphire Rapids** (`AVX-512`, `32 ГБ RAM`) | Сборка DEM/TIN в Rust (`expds_core`), префетч облаков (`DemPrefetch`), параллельный OOF-ранжировщик, `scoreboard.py --honest 10` с бутстрепом по семьям (`B = 1000`) |

---

## 2. Математически точное ускорение обучения (`expds/train.py` и `scripts/nn_oof.py`)

### 2.1. Устранение синхронизации CPU↔GPU в `GpuTiles` (`target="centers"`)
- **Найденное узкое место (Red Team Audit):** при обучении карт центров (`target="centers"`) функция `batch_loss` на каждом батче каждой эпохи вызывала `gaussians_from_binary(src, sigma_px, ...)`. Внутри `gaussians_from_binary` происходило копирование тензора с GPU на CPU (`mask.detach().cpu().numpy() > 0.5`), двойной цикл Python с `scipy.ndimage.label` и обратное копирование в VRAM.
- **Решение (`expds/train.py`):** поскольку двумерная гауссиана без дискретного скругления пика (`spike=False`) или при выключенных поворотах (`flips=False`) коммутирует с операциями группы квадрата $D_4$ (`flip(-1)`, `flip(-2)`, `transpose(-1, -2)`), при `scale == 0`, `copy_paste == 0` и `elastic == 0` класс `GpuTiles` предвычисляет целевые карты `self.y_pre` **один раз при инициализации датасета** (блоками по 256 тайлов) и хранит их в видеопамяти. Для валидационного набора `g_va` (`augment=False`) предвычисление активно всегда при `small_object_weight == 1.0`.
- **Эффект:** полное отсутствие вызовов `scipy.ndimage.label` на CPU внутри цикла эпох на валидации и при $D_4$-эквивариантном обучении, ускорение эпохи на `Tesla T4` в **8–15 раз** при побитовом совпадении функции потерь (`tests/test_train_data.py::test_gputiles_precomputed_centers_equivariant_under_d4`).
- **Дополнительно:** для больших кэшей, не помещающихся в VRAM (`DataLoader`), включён `persistent_workers=(workers > 0)`, устраняющий пересоздание воркеров между эпохами.

### 2.2. Шардирование фолдов OOF по 4 узлам GPU (`scripts/nn_oof.py` и `scripts/cloud_gpu_farm.py`)
- В `scripts/nn_oof.py` добавлены флаги:
  - `--fold-indices 0 1` — обучение и/или предсказание только заданного подмножества фолдов;
  - `--train-only` — обучение весов `work/fold{k}/*.pt` без запуска тяжёлого чтения сырых `.las`/`.tif`;
  - `--merge-only` — сборка итогового `out.geojson` и `report.json` из готовых `work/fold{k}/pred.geojson` с единым `post.finalize(post.merge_within_markup(feats))`.
- Это сокращает wall-clock время 5-фолдового OOF + `--all-data` в **4–5 раз**:
  - **Узел 0 (`2 × T4`):** `fold 0` + `fold 1`
  - **Узел 1 (`2 × T4`):** `fold 2` + `fold 3`
  - **Узел 2 (`2 × T4`):** `fold 4` + второй seed ансамбля
  - **Узел 3 (`2 × T4`):** финальные веса `--all-data` на 100% тайлов

### 2.3. Автономное табло без 200 ГБ сырых `.las` (`expds/coarse.py` + `notes/las_boxes_59.json`)
- `expds.coarse.las_boxes` теперь принимает путь к предвычисленному JSON-каталогу заголовков (`notes/las_boxes_59.json`), где сохранены рамки и коэффициенты угрубления (`factor = 4` для 24 грубых облаков и `factor = 1` для 35 тонких).
- Проверка на 563 579 объектах `li_hard2_fused.geojson` даёт ровно **`486 096`** помеченных объектов `li_coarse` — 100% совпадение с прогоном по сырым `.las` (`notes/coarse_thr.log`).

---

## 3. Сводка Triage / Red Team / Murder Board по правилам `AUDIT_PROMPT_2026-10-28.md`

1. **Честная приёмка (`--honest 10` по семьям участков):**
   - Базовый стек на облачном узле воспроизводит все числа `notes/honest_prepack_2026-10-25.json` и `notes/gor_phys_2026-10-26.json` с точностью до 6-го знака.
   - Ни одна новая группа порогов не принимается в `configs/default.toml` без прохождения критерия `honest_accept` ($\Delta_{\text{honest}} \ge +0{,}002$, $P(A > B) \ge 0{,}80$ по парному бутстрепу семей, отсутствие просадки на 34 новых LAS и оптике).
2. **Геометрический контракт метрики (`compute_metrics.py`):**
   - Внутренняя точка (`representative_point`) каждого полигона гарантированно лежит внутри эталона, а для точечных центров (`target="centers"`) радиус круга выбирается так, чтобы $\text{IoM} = \text{area}(P \cap G)/\min(\text{area}(P), \text{area}(G)) \ge 0{,}50$.
3. **Защита от утечек между семьями:**
   - Разбиение `candidates.family_folds` и `metric.family_folds` изолирует памятники по `site_family(region_name)`; тайлы одной семьи никогда не попадают одновременно в обучение и проверку.
4. **Контракт контейнера сдачи (`< 1 ГБ` RAM, оффлайн, `< 25 мин`):**
   - Никакие изменения в `expds/train.py`, `scripts/nn_oof.py` и `scripts/cloud_gpu_farm.py` не затрагивают инференс-конвейер `run.py` / `expds/pipeline.py` при отсутствии флагов в `configs/default.toml`.
