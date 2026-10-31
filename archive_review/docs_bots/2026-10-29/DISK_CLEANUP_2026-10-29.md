# Disk cleanup 2026-10-29 ~14:50 MSK (approved by owner)
Free: 7.9G -> 22G.
- DELETED bot-8/.uvcache (6.7G uv cache; re-downloads on next `uv` run, UV_CACHE_DIR in bot-8/env.sh still points there).
- DELETED /workspace/xfer/p00..p66 (0.64G old v3 36af6b7 transfer parts; *.py scripts kept).
- HARDLINKED (not deleted) identical git pack pack-49020296...pack (754M) in 10 repos to /workspace/expds-koz3 copy (~7.4G saved). Packs are immutable; git gc/repack is safe.
- HARDLINKED identical model files: bot-9 dfine-large-coco snapshot model.safetensors + bot-5/tmp/hf blob 583dd94... (same sha256), bot-9/models/dfine_or/model.safetensors <-> kaggle_out/koz3-dfine-or/dfine_f1. Do NOT overwrite these in place (would change both); write new files instead.
- Removed 19 __pycache__/.pytest_cache dirs outside venvs (2M).
