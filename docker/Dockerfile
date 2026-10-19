# Окружение для запуска решения на платформе КОЗ №3.
#
# В образе ТОЛЬКО окружение: без кода и без весов. Образ обязан быть
# публичным на Docker Hub, и всё, что в нём лежит, видят конкуренты
# (слои читаются через API). Код, веса и expds_core*.so едут в архиве.
#
# Сборка и публикация (имя образа лучше нейтральное, без «koz3»):
#   docker build -t ВАШ_ЛОГИН/geo-runtime:2026-10 -f docker/Dockerfile .
#   docker push ВАШ_ЛОГИН/geo-runtime:2026-10
# Затем это имя — в metadata.json, поле image.
#
# Локальная проверка как на платформе (без сети):
#   docker run --rm --gpus all --network none -v %CD%:/app -w /app ^
#     -v D:/val:/data/input -v D:/out:/data/output ВАШ_ЛОГИН/geo-runtime:2026-10 ^
#     python inference/main.py /data/input /data/output/result.geojson

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 GDAL_PAM_ENABLED=NO OMP_NUM_THREADS=8

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch с CUDA 12.6 (колёса несут свои библиотеки CUDA; драйвер — на хосте платформы)
RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt \
    && python -c "import torch, rasterio, shapely, pyproj, laspy, scipy, segmentation_models_pytorch; print('ok')"

WORKDIR /app
