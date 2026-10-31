# PC hardware profile (DESKTOP), read-only, 2026-10-29 MSK
- CPU: i5-13600KF Raptor Lake, 6P+8E = 14C/20T, 3.5 GHz base. AVX2 yes, AVX-512 no.
- RAM: 2x16 GB DDR5-6000. WSL2/Docker sees 15.5 GiB. v2 mock_val peak 4.6 GiB.
- GPU: GTX 1650 TU117, 4 GB, 75 W, PCIe3 x16, driver 610.88. No tensor cores, no TF32. fp16 gives only about 2x.
- C: KC3000 1 TB NVMe, only 4.5 GB free. F: ST2000VX008 2 TB HDD, 23-49 MB/s measured.
- OS: Win11 Pro WS 26200, WSL2 6.6.87.2, Docker 29.8.2. Power plan: Balanced.
- venv: Py 3.12.10, torch 2.11.0+cu128, cuDNN 91900, numpy 2.5.4, lightgbm 4.7.0. Rust expds_core 0.1.0 avx2.

## v2 eb268d6 mock_val Docker sim: utilization per phase (logs in C:\Sloi\pc_measure_v2, copies in v2prof/)
| Phase | s | GPU avg/max | CPU avg (of 20 thr) | RAM pk | Bottleneck |
|---|---|---|---|---|---|
| geophys (PIXGBM 31 s) | 37 | 8%/9% | 4.6 thr | 1.2 GiB | CPU; PIXGBM features single-threaded |
| lidar | 160 | 18%/99% | 1.9 thr | 4.7 GiB | serial CPU + LAS read (M030 65 s, M032 46 s) |
| optics | 421 | 30%/100% | 0.5 thr | 3.3 GiB | I/O and serialization: CPU and GPU mostly idle |
| sigma | 17 | 32% | 1.3 thr | 3.1 GiB | small |
Total 643.5 s. Optics is 65% of the time with both processors idle, so reads are not overlapped with inference.
