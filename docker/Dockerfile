# Окружение для запуска решения на платформе КОЗ №3.
#
# В образе ТОЛЬКО окружение: без кода и без весов. Образ обязан быть
# публичным на Docker Hub, и всё, что в нём лежит, видят конкуренты
# (слои читаются через API). Код, веса и expds_core*.so едут в архиве.
#
# Опубликован pantum/n7q:20261023d. Этот тег не переписывать.
# Новая сборка — новый тег без говорящего имени и без описания на Hub.
# Для локальной проверки без GPU: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
FROM python:3.12-slim-bookworm

# Для локальной проверки без GPU: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 GDAL_PAM_ENABLED=NO OMP_NUM_THREADS=8

# libexpat1: колесо rasterio (GDAL) без неё не импортируется в slim-образе
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CUDA 12.8: RTX 4090 (Ada) и более новые карты. Колёса несут свои библиотеки CUDA.
# Та же версия, на которой подбирались пороги (.venv, 22.10.2026).
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url ${TORCH_INDEX}

# Точные версии остальных пакетов; requirements.txt — только нижние границы для разработки.
COPY requirements.lock /tmp/requirements.lock
RUN pip install -r /tmp/requirements.lock && rm /tmp/requirements.lock \
    && python -c "import torch, rasterio, shapely, pyproj, laspy, scipy, segmentation_models_pytorch; print('ok')"

WORKDIR /app
