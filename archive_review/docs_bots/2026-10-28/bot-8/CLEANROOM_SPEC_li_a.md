# Clean-room spec: LiDAR expert, scenario (a) — DTM, relief stack, tiled inference, ensemble

bot-8, 28.10.2026 (МСК). Written from reading the behaviour of EhimenNathan-EXpedition @d434e56 (`koz3/dtm.py`,
`koz3/relief.py`, `koz3/hm.py:predict_tiled`, `inference/koz3_infer/experts/las.py:predict`). The spec describes
algorithms, data and parameters in prose only — no code text. The implementation (`expds/lidar_a/`, branch
`bot8-cpu-a-clean`) is written from this spec with its own structure and names. The original is used only as a test
oracle loaded at test time from the read-only competitor copy (tests skip when it is absent); it is not vendored.
Caveat: spec and implementation have the same author (one agent), so this is a documented clean-room procedure,
not a two-team one. Bit-exactness against the oracle requires the same floating-point operation order; where that
matters it is stated below as a requirement on the arithmetic (not as code).

## 1. Ground grid from a LAS/LAZ file
Inputs: point cloud path. Output: elevation grid `dtm` (float32, H×W, no NaN), `valid` (bool H×W), origin
`(x_left, y_top, cell)`.

1. Grid geometry from the header bounding box only. Start with cell 0.25 m; while (extent_x/cell)·(extent_y/cell)
   exceeds 250 million, double the cell. Left edge = floor(min x); top edge = ceil(max y). Columns =
   ceil((max x − left)/cell) + 1, rows = ceil((top − min y)/cell) + 1. Row 0 is the top (north) row.
2. Ground selection mode, decided on the first block of 2,000,000 points as read:
   - "class" mode if the share of points with classification 2 is at least 5 %;
   - otherwise "ground-only" mode if that block looks like bare earth: bin its points into 1 m squares measured
     from the block's own min x / min y, keep squares with ≥ 3 points, take max−min of z per square; bare earth if
     there is at least one such square and the 95th percentile (linear interpolation) of those spreads is < 0.6 m;
   - otherwise "lowest-return" mode.
3. Rasterisation, streaming the file in blocks of 5,000,000 points (block size is part of the arithmetic: per-cell
   sums are formed per block and added to the running total block by block, in file order). Elevation is taken as
   float32; x, y as float64. Column index = (x − left)/cell truncated toward zero, row index = (top − y)/cell
   truncated; points outside the grid are dropped. Class mode keeps only class 2; the other two modes drop classes
   7 and 18 (noise). Class and ground-only modes accumulate a float64 sum of the float32 elevations and an int32
   count per cell; the cell value is sum / max(count, 1) in float64, cast to float32; empty cells are NaN.
   Lowest-return mode keeps the per-cell float32 minimum (empty = NaN) and then applies the morphological filter.
4. Progressive morphological filter (Zhang et al. 2003) on the minimum grid: if everything is empty, return as is.
   Fill empty cells with the value of the nearest non-empty cell (Euclidean nearest, as given by scipy's exact EDT
   with indices) to get a working surface; ground flag starts as "non-empty". For window sizes 1, 2, 4, 8, 16 m in
   order: odd window side in cells = the rounded value of size/cell with its lowest bit set, at least 3; open the
   surface with a flat square structuring element of that side (grey opening, scipy default boundary); height
   threshold = 0.3 m for the first window, then min(3.0, 0.3 + 0.3·(size − previous size)); a cell stays ground only
   if surface − opened ≤ threshold; the opened surface becomes the working surface. Non-ground cells of the minimum
   grid become NaN.
5. Hole filling ("nearest ground"): every cell takes the value of its nearest ground cell (scipy exact EDT indices).
   If any cell was filled, compute a Gaussian blur (σ = 1 cell, scipy defaults: truncate 4, reflect boundary) of the
   nearest-filled grid and use the blurred value for filled cells only; ground cells keep their own value.
   `valid` = (EDT distance in cells) × cell ≤ 3.0 m. Output cast to float32.

### 1a. Faster but identical fill ("exact_fast", default) — our design
The EDT is run for indices only. Validity is decided on the integer squared distance d² = Δrow² + Δcol²: the
largest integer d²_max with sqrt(d²_max)·cell ≤ 3.0 (float64) is found once; the float test is monotone in the
integer d², so `d² ≤ d²_max` gives the same mask. Gather and blur run in row blocks on a thread pool; each block is
blurred with a 4-row halo (the σ = 1 kernel reaches 4 rows), so each kept row sees the same arithmetic → bit-equal.

### 1b. Banded fill ("band", opt-in)
Tiles (multiple of 32 cells) are processed on threads; each tile runs an indices-only EDT on its window extended by
a band B = max(ceil(80 m / cell), ceil(3 m / cell) + 2). A cell whose nearest ground found in the window is within
B is exact; farther cells (necessarily invalid) take the first ground cell of the nearest ground-bearing 32×32 block
(block-level EDT). 80 m exceeds the receptive field of every relief channel from a valid pixel (§2), so the network
input is equal in value (only the sign of value×0 on invalid pixels may differ).

## 2. Relief stack (network input), on a torch device
Inputs: dtm, valid, source cell s. Output: C×h×w float32 at 0.5 m (C = 9, or 12 with openness), valid h×w bool.

1. Resampling to 0.5 m, factor f = 0.5/s, output h = round(H/f), w = round(W/f). If f is an integer > 1: pad the
   dtm at bottom/right by edge replication to a multiple of f (valid padded with False), average-pool f×f, crop to
   h×w (both dtm and valid-as-float). If f = 1: nothing. Otherwise interpolate to (h, w): dtm "area" when f > 1
   else bilinear (no corner alignment); valid "area" when f > 1 else nearest.
2. Separable Gaussian G(σ px): taps k = (integer part of 3σ) with lowest bit set, at least 3; tap offsets
   −k//2…k//2 in the tensor dtype; weights exp(−offset²/(2σ²)) divided by their sum; horizontal pass first (edge
   replicate pad k//2 left and right, 1×k correlation), then vertical (k×1).
   Wide Gaussian WG(σ): if σ ≤ 12 it is G(σ); else factor q = floor(σ/6): edge-pad right/bottom to multiples of q,
   average-pool q×q, G(σ/q), bilinear upsample by scale q (no corner alignment), crop to the input size.
3. Channels, r = 0.5 m, z = resampled dtm (1×1×h×w):
   - LRM at 2, 5, 10, 25 m: (z − WG(scale/r)) divided by 0.5, clipped to [−4, 4] (variant "unclip": asinh instead
     of clipping).
   - Slope: zs = G(0.75)(z); gx = (zs[col+1] − zs[col−1]) / (2r) on interior columns, border columns replicated;
     gy likewise along rows. Slope angle = atan(sqrt(gx² + gy²)); channel = slope / 0.5.
   - Hillshade at azimuth 315° and 45°, altitude 35°: aspect = atan2(−gx, gy); value = sin(alt)·cos(slope) +
     cos(alt)·sin(slope)·cos(azimuth − aspect) (azimuth, altitude in radians, evaluated as float64 scalars, products
     left to right); channel = 2·value − 1.
   - Laplacian: G(1/r) of z, edge-pad 1, 3×3 4-neighbour Laplacian stencil (centre −4), divided by r²; channel =
     (lap / 0.2) clipped to [−4, 4].
   - (12-ch only) Openness and sky-view, radius 10 m: K = max(2, round(10/r)) = 20 px; z edge-padded by K. For 8
     directions with angle (i·2·π)/8 (float64, multiply before divide): upward and downward horizon angles start at
     −π/2; for step lengths k = 1, 1+t, 1+2t, … ≤ K with t = max(1, K//10): offset (round(k·sin), round(k·cos)) in
     (row, col), horizontal distance = hypot(offset)·r, skipped if 0; elevation angle = atan((z_shifted − z)/dist);
     up = max(up, angle), down = max(down, −angle). After each direction add (π/2 − up) to the positive sum, (π/2 −
     down) to the negative sum, sin(max(up, 0)) to the sky sum (sums start at the integer 0, directions in order).
     Means = sums / 8, sky-view = 1 − sky_sum/8. Channels: ((pos − π/2)/0.1), ((neg − π/2)/0.1), ((svf − 0.97)/0.03),
     each clipped to [−4, 4].
   - Valid: vb = 1 where resampled valid > 0.5 else 0; channel = 2·vb − 1.
   Order: LRM2, LRM5, LRM10, LRM25, slope, hs315, hs45, laplacian, [pos-open, neg-open, svf], valid. All channels are
   multiplied by vb. The 9-channel stack is therefore channels 0–7 and 11 of the 12-channel one (our item (2)).
4. Windowed evaluation for big sites: if h·w ≤ 3072², evaluate at once. Else walk 3072×3072 output windows in row-
   major order with a 384 px margin (clipped to the grid); the source window starts at round(start·f) and is
   round(size·f) long, edge-padded (valid False) if it runs past the dtm; evaluate and copy the window's core.

## 3. Tiled network inference with flip/rotation TTA
Inputs: model, C×H×W stack, n_out, tile T, overlap O, TTA mode, device. Output n_out×H×W float32 in [0, 1].
- Tile origins along each axis: 0, T−O, 2(T−O), … while < max(1, size − O); tile end = min(size, origin + T).
- Each tile is padded bottom/right to a multiple of 32: reflect padding if min(tile h, w) > max(pad h, pad w), else
  zeros; moved to the device as a float32 batch of 1.
- Views: TTA "d8" on square padded tiles → 8 views: rotations k = 0..3 (90° steps in the H×W plane), each without
  then with a horizontal (last axis) flip; otherwise TTA on → identity, flip of the last axis, flip of the row axis;
  TTA off → identity. Each view: apply rotation then flip, run the model inside CUDA bfloat16 autocast, cast output
  to float32, sigmoid, undo the flip, rotate back; sum views starting from 0 in view order, divide by the view count,
  crop to the tile size.
- Blending: base ramp (float64) = min(index+1, O)/O over T entries. Per axis: if the tile does not cover the whole
  axis, weight = min(ramp, reversed ramp) over the tile length, else ones; then the first O entries are set to 1 if
  the tile starts at 0 and the last O entries to 1 if it ends at the axis end. Tile weight = outer product (float64)
  cast to float32. Accumulate out += prediction × weight (product first, then add), count += weight; result =
  out / max(count, 1e-6), float32.
- OOM ladder: tile 1024, 512, 256 (overlap = min(192, T//4)); on CUDA OOM empty the cache and retry smaller.

### 3a. Device accumulation (our item (5))
"device_exact" (default): out/count live on the model's device, weights built on the host exactly as above and
cached per tile shape on the device; one transfer per model at the end, then the upstream host ensemble → bit-equal.
"device": the ensemble is a running weighted sum on the device (≤ 1 ulp from the host tensordot). If the
site-sized buffers do not fit, accumulation falls back to host torch tensors (same arithmetic).

## 4. Site procedure and ensemble
- Models = list of (net, kind, weight); kind "" uses the 9-channel stack, "x" the 12-channel one, "u" the 12-channel
  stack with asinh LRMs. If < 120 s remain before the deadline, only the first model is used; if TTA is "d8" and
  < 300 s remain, TTA falls back to the 3-view flips.
- Ensemble: weights as float32, divided by their float32 sum; map = Σ weight·map via a single tensordot over the
  stacked maps (host, float32).
- Decoding of peaks/polygons is NOT part of this spec (platform glue passes a decoder callable).

## 5. Parallel DTMs (our item (6)) — our design
DTMs of all files of a run are built in worker subprocesses (no fork after CUDA): jobs ordered by point count
(largest first), started while the estimated RAM of running jobs fits a budget (estimate from the LAS header:
64 B/cell + 84 B/point for up to 5 M points + 80 MiB; a job always starts if none is running); results come back
as .npy files in /dev/shm; the consumer takes them in completion order and the run output is reassembled in the
original order.
