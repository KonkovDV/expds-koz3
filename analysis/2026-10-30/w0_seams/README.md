# W0 seams: rp-anchor micro-disk (30.10)
- m1_anchor.py: per pred, disk r=5 cm at rp / centroid-in-P / polylabel vs same-class GT (stack_arch OOF, W5 table.pkl). -> m1_anchor.json, w5data/work/w0_disk_pairs.pkl
- g1_disk_gate.py: honest-10 vs the stack: A_rp +0.00521 P 0.9955; C_cen +0.00158 P 0.69; B_pl +0.00267 P 0.79.
- g2_decomp.py: vs B' (kurgans already micro 0.5 as on main): NK_rp +0.00437 P 0.9885 new34 +0.00225 opt +0.00852 -> PASS (d>=0.003, P>=0.9, Bonferroni x3 P>=0.967 too).
  fort +0.00205 P 0.982; gor +0.00248 P 0.867 (new34 -0.0012); arch/object_poly ~0; kurgans rp vs centroid -0.00034.
- Theory: the metric checks rp(P) in G and cov>=0.5; a disk of r<<G at rp(P) has cov=1, so it is a hit whenever P is.
- Code: post.micro_circles(spec.anchor="rp"); the default (centroid) is byte-identical. The config is commented in default.toml.
