# koz3-w2-weights — веса моделей W2, ~1,47 ГБ, 24 файла

**Что это:** пофолдовые веса c3-sc (каждые веса обучены БЕЗ своего фолда) → можно делать честный OOF-инференс без переобучения.

| Путь | Что | Ядро-источник |
|---|---|---|
| `s0/koz3-cnx-li-c3sc-f01/weights_cnx_c3-sc_f{0,1}.pt`, `s0/…-f23/…f{2,3}.pt`, `s0/…-f4-v2/…f4.pt` | сид 0, 5 фолдов | dmitrijkv/koz3-cnx-li-c3sc-* |
| `s1/koz3-cnx-li-c3sc-{s1f01,s1f23,s1f4}/weights_*.pt` | сид 1 | dmitrijkv / elenakonkova |
| `s2/koz3-w2-cnx-c3sc-{s2f01,s2f23,s2f4}/weights_*.pt` | сид 2 (+ `ckpt_*.pt` f2–f4 — чекпойнты с оптимизатором, для дообучения; `*.log`) | dmitrijkv / elenakonkova |
| `lihm/koz3-w2-lihm-c3sc-f01234/ckpt_cnx_c3-sc_f{0..4}.pt` | heatmap-голова (P-W2-7, FAIL); только ckpt, в метаданных нет описания новой головы | elenakonkova/koz3-w2-lihm-c3sc-f01234 |

Формат `.pt` (torch, `weights_only=True`): `{'state_dict', 'in_channels': 3, 'train': {'fold': K, ...}}`. Архитектура: `cnx_head.get_head('centerhead', cfg).build(3, 'tu-convnext_tiny', None)`, cfg — `report_cnx_full.json['cfg']` из koz3-w2-oof. Вход — рельеф (koz3-cnx-li-prep), каналы uint8, нормировка `(x-128)/127`.

## Инференс без обучения (CPU, бокс)
```
python w2_demfill/infer_cpu.py --relief <cnx_relief/XXX.npy> --meta <cnx_relief/XXX.json> \
  --weights s2/koz3-w2-cnx-c3sc-s2f23/weights_cnx_c3-sc_f2.pt --fold 2 --out maps_XXX.npz --threads 3
```
Правило: для участка фолда K брать только веса `_fK` (иначе утечка). Вендор-код — `/workspace/koz3/w2/vendor` (копия кода ядер; в ядрах — тот же код внутри .py). Карты → детекции: декодер ядра (`save_maps`/decode в `w2_kernels/*.py`), затем `merge_cnx.py`.
На Kaggle: `dataset_sources: ["elenakonkova/koz3-w2-weights","elenakonkova/koz3-cnx-li-prep"]`, пути `/kaggle/input/koz3-w2-weights/s2/...`.
