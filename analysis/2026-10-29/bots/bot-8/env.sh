export B8=/workspace/koz3_shared/bot-8
export RUSTUP_HOME=$B8/.rustup CARGO_HOME=$B8/.cargo CARGO_TARGET_DIR=$B8/target CARGO_BUILD_JOBS=2
export PATH=$B8/.cargo/bin:$B8/venv/bin:$PATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2
export UV_CACHE_DIR=$B8/.uvcache
export EXPDS_THREADS=2
export EXPDS_METRIC_THREADS=2
export EXPDS_CPU_THREADS=2   # torch_all_cores() on the box: 2-thread cap
