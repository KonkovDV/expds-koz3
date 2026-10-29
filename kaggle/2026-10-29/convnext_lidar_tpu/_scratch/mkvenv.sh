set -e
cd /workspace/koz3_shared/bot-9/kaggle/convnext_lidar_tpu
export UV_CONCURRENT_INSTALLS=2 UV_CONCURRENT_DOWNLOADS=2
nice -n 10 uv venv -p 3.12 _scratch/venv
nice -n 10 uv pip install -p _scratch/venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchvision==0.23.0+cpu
nice -n 10 uv pip install -p _scratch/venv/bin/python torch_xla==2.8.0 timm==1.0.29 segmentation-models-pytorch==0.6.0 numpy==2.1.3 rasterio==1.4.3 pyproj==3.7.1 shapely==2.1.1 boto3==1.40.61 scipy==1.16.3 laspy==2.7.0
nice -n 10 uv pip install -p _scratch/venv/bin/python --no-deps rvt-py==2.2.1
_scratch/venv/bin/python -m pip --version || true
uv pip freeze -p _scratch/venv/bin/python
echo DONE
