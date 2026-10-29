# SPEEDUP_PLAN (bot-1, 2026-10-29). Plan only, no code rewritten.
Target: the 4090 submission container. Rule: every change must be bit-identical, or within tolerance (with a written rationale),
and must re-pass honest-10 (keep >= A 0.333237) before merge.

Ranked by expected gain / effort:
1. Optics I/O-compute overlap (est. -30..50% of optics wall, effort S). Optics is 65% of the v2 time, at 0.5 CPU thread and 30% GPU.
   Prefetch the next raster/tile with a ThreadPool (2-4 workers; GDAL releases the GIL) while the GPU infers. Use pinned memory with non_blocking copies.
   Bit-identical: same tensors, only the order of work changes.
2. GDAL env (effort XS, bit-identical): GDAL_CACHEMAX=1024, GDAL_NUM_THREADS=ALL_CPUS (decode), VSI_CACHE=TRUE, GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR.
3. Thread env = physical cores (XS): OMP/MKL/OPENBLAS/NUMEXPR=physical cores; LightGBM num_threads=physical cores (not HT).
   PIXGBM feature stage is single-threaded (54.7 s on region 093): parallelize per tile with a process or thread pool (S). Deterministic.
4. Morph features (bot-2, 0.87 ms/candidate, about 1057 s at train scale; effort S-M):
   a) First vectorize with numpy/scipy.ndimage, batched per tile, not per candidate.
   b) numba @njit(cache=True, parallel=True); the numba wheel must be in the offline wheelhouse.
   c) Rust via PyO3/maturin in the existing expds_core (already ships AVX2, offline wheel, MIT/Apache deps). Best if (a) is not enough. Expected 10-50x, bit-identical if integer/float order is kept.
5. torch inference (4090): channels_last + fp16 autocast (TF32 is not available on the PC, so verify on box/Kaggle);
   torch.backends.cudnn.benchmark=True for fixed tile size. fp16 is NOT bit-identical: require tolerance + honest-10.
   torch.compile / CUDA graphs are optional (compile time counts toward the time limit).
6. ONNX Runtime / TensorRT EP (M-L): about 1.3-2x on the GPU part only, but adds wheels and risk. Lowest priority, because GPU is not the bottleneck.
7. lidar: big-LAS regions dominate (M030 65 s). Read LAS with laspy chunked iteration + Rust gridding (already used), and overlap reading with gridding (S).

PC-side only (does not affect the 4090): power plan High performance (reversible, powercfg /setactive SCHEME_MIN);
F: HDD is the PC bottleneck, but C: has only 4.5 GB free, so staging is impossible until space is freed.
GTX 1650 has no tensor cores, so fp16/cudnn.benchmark speedups measured here understate the 4090 gains. Verify ratios on Kaggle/box.
