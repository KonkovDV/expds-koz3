"""LiDAR expert, scenario (a): clean-room implementation (bot-8, branch bot8-cpu-a-clean).

Written from the prose spec bot-8/CLEANROOM_SPEC_li_a.md (algorithms, parameters, float operation order); contains
no third-party code. Modules: ground (LAS → DTM), terrain (relief stack), tiling (tiled inference + blending),
runner (site procedure / ensemble). Bit-equality with the reference implementation is checked by
tests/test_lidar_a_oracle.py, which loads the reference from a local read-only copy at test time and skips without it.
"""
