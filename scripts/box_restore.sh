#!/usr/bin/env bash
# Восстановить «бокс» команды (/workspace/koz3) для гейтов honest-10 в чистом контейнере (аудит 03.11).
#
#   KAGGLE_TOKEN_DMITRIJKV=... bash scripts/box_restore.sh [VENV_DIR]
#
# Токены — только из окружения, в файлы не пишутся. Источники:
#   - архив бокса R_results: приватный датасет dmitrijkv/koz3-handoff-20261102 (koz3bk_R_results.tar.xz, 101 МБ);
#     в нём orch/bot-W4/work/v3_gate_lib.py, пороги v3prod, разбиения; d3_gate.py — в wt/coord-orch/...;
#   - стек OOF базы A: dmitrijkv/koz3-gate-box-20261104 (stack_arch.geojson 1,1 ГБ, table.pkl, base_counts.pkl, gt_train_v2.geojson,
#     SHA256SUMS) — копия с бокса 04.11: аккаунта elenakonkova (и koz3-w2-gate на нём) с 04.11 нет;
#   - OOF D-FINE f0 (koz3-dfine-spor-full/-full-rest), D-FINE-2 (koz3-w2-dfine2-a/-b). OOF Faster R-CNN (elenakonkova) потерян.
# Чего нет ни в одном бэкапе: w4geo/eval/{A18,C19etonly_orget_s0}.geojson (геофизика базы). load_geo тогда отдаёт []:
# V3P = 0,334435 вместо 0,3337867 (+0,00065, только object_poly EtGr); Δ/P изменений только в классах ДЗЗ не меняются.
#
# Если Kaggle отвечает 429 и повторы не помогают: те же архивы лежат на Drive «2» → KOZ3_backup_2026-11-04
# (01_gate_base, 03_oof, 08_KOZ3-final_mirror/backup_2026-11-02/results; порядок — 00_README.md там же).
# 05.11 бокс восстановлен оттуда: суммы совпали, `gate_cnxin_ctx.py base` дал V3P 0,3344354 и V3P_ctx 0,3465004.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
K=/workspace/koz3
VENV="${1:-$K/venv312}"
T_D="${KAGGLE_TOKEN_DMITRIJKV:?set KAGGLE_TOKEN_DMITRIJKV}"
mkdir -p "$K"

# 1) Python 3.12 + зависимости сдачи (как CI)
if [ ! -x "$VENV/bin/python" ]; then
  uv venv -p python3.12 "$VENV" >/dev/null
  VIRTUAL_ENV="$VENV" uv pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
  VIRTUAL_ENV="$VENV" uv pip install -q -r "$REPO/requirements-dev.txt" -r "$REPO/requirements.lock"
fi
# w4venv — симлинк на каталог venv: Python находит pyvenv.cfg рядом с путём запуска. Симлинк только на bin/python
# запускал голый системный Python без пакетов venv (05.11).
mkdir -p "$K/wt"
if [ -d "$K/w4venv" ] && [ ! -L "$K/w4venv" ]; then rm -rf "$K/w4venv"; fi
ln -sfn "$VENV" "$K/w4venv"
ln -sfn "$REPO" "$K/wt/w1"

# Kaggle режет частые скачивания (HTTP 429): повторы с паузой, curl считает 429 временной ошибкой.
RETRY=(--retry 6 --retry-delay 30 --retry-max-time 1800)
ds() {  # owner slug file dst
  [ -s "$4" ] && return 0
  mkdir -p "$(dirname "$4")"
  curl -sSLf "${RETRY[@]}" -m 3600 -H "Authorization: Bearer $5" \
    "https://www.kaggle.com/api/v1/datasets/download/$1/$2/${3//\//%2F}" -o "$4.part" && mv "$4.part" "$4"
}
kout() {  # token user slug dst — только *.geojson и *report.json из корня вывода ядра
  mkdir -p "$4"
  curl -sSf "${RETRY[@]}" -m 120 -H "Authorization: Bearer $1" "https://www.kaggle.com/api/v1/kernels/output?userName=$2&kernelSlug=$3" |
    "$VENV/bin/python" -c "
import sys, json
for f in json.load(sys.stdin).get('files', []):
    n = f['fileName']
    if '/' not in n and (n.endswith('.geojson') or n.endswith('report.json')): print(n, f['url'])" |
    while read -r n u; do [ -s "$4/$n" ] || curl -sSLf -m 3600 "$u" -o "$4/$n"; done
}

# 2) Архив бокса и библиотеки гейта
if [ ! -f "$K/orch/bot-W4/work/v3_gate_lib.py" ]; then
  ds dmitrijkv koz3-handoff-20261102 koz3bk_R_results.tar.xz "$K/_dl/R.tar.xz" "$T_D"
  tar -xJf "$K/_dl/R.tar.xz" -C "$K"
fi
mkdir -p "$K/orch/bot-W1/work/d3_vendor"
cp -n "$K/wt/coord-orch/orch/bot-W1/work/d3_gate.py" "$K/orch/bot-W1/work/d3_gate.py"
git -C "$REPO" show cbd4804:expds/sp_dfine.py > "$K/orch/bot-W1/work/d3_vendor/sp_dfine_cbd4804.py"
"$VENV/bin/python" - "$K/orch/bot-W4/work/v3_gate_lib.py" <<'EOF'
import sys
p = sys.argv[1]; t = open(p).read()
old = 'def load_geo(p, pred):\n    return [f for f, _ in iter_features(Path(p)) if pred(f["properties"])]\n'
if old in t:
    t = t.replace(old, 'def load_geo(p, pred):\n    if not Path(p).exists():   # box_restore: C19/A18 not in any backup\n'
                       '        log(f"load_geo: {p} missing -> []")\n        return []\n'
                       '    return [f for f, _ in iter_features(Path(p)) if pred(f["properties"])]\n')
    open(p, "w").write(t)
EOF

# 3) Стек OOF базы A и эталон
GB=koz3-gate-box-20261104
for f in stack_arch.geojson gt_train_v2.geojson; do ds dmitrijkv "$GB" "$f" "$K/w5data/$f" "$T_D"; done
for f in table.pkl base_counts.pkl; do ds dmitrijkv "$GB" "$f" "$K/w5data/work/$f" "$T_D"; done
ds dmitrijkv "$GB" SHA256SUMS "$K/w5data/SHA256SUMS" "$T_D"
(cd "$K/w5data" && sha256sum -c SHA256SUMS)

# 4) OOF детекторов спутника
kout "$T_D" dmitrijkv koz3-dfine-spor-full "$K/w4geo/spor_oof/full"
kout "$T_D" dmitrijkv koz3-dfine-spor-full-rest "$K/w4geo/spor_oof/rest"
kout "$T_D" dmitrijkv koz3-w2-dfine2-a "$K/w2/oof/dfine2/dmitrijkv_koz3-w2-dfine2-a"
kout "$T_D" dmitrijkv koz3-w2-dfine2-b "$K/w2/oof/dfine2/dmitrijkv_koz3-w2-dfine2-b"

# 5) Конфиг v3_on для контекста гейта
mkdir -p "$K/w2/r50c5_gate/v3on_src/configs"
cp "$REPO/configs/v3_on.toml" "$K/w2/r50c5_gate/v3on_src/configs/default.toml"
echo "box restored at $K (venv $VENV)"
