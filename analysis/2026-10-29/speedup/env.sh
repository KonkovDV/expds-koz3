export SP=/workspace/koz3_shared/speedup
export B8=/workspace/koz3_shared/bot-8
export RUSTUP_HOME=$B8/.rustup CARGO_HOME=$B8/.cargo CARGO_TARGET_DIR=$SP/target CARGO_BUILD_JOBS=2
export PATH=$B8/.cargo/bin:$B8/venv/bin:$PATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 EXPDS_THREADS=2 TORCH_THREADS=2
