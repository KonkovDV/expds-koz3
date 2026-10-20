//! Приведение DEM к рабочему шагу модели.
//!
//! Целый множитель — среднее блоков k×k (так училась сеть: 0,1 м → 0,5 м, k = 5).
//! Иначе каждая ячейка 0,5 м собирает все ячейки исходного шага, которые в неё
//! попадают. Формулы индекса и порядок суммирования совпадают с NumPy в
//! `candidates.to_work_grid`, поэтому результат тот же.

use rayon::prelude::*;

/// Округление как `round` в Python 3: половинки к ближайшему чётному.
pub fn py_round_pos(x: f64) -> i64 {
    let floor = x.floor();
    let frac = x - floor;
    if (frac - 0.5).abs() < 1e-9 {
        let i = floor as i64;
        if i % 2 == 0 { i } else { i + 1 }
    } else {
        x.round() as i64
    }
}

/// Среднее блоков k×k. Хвост, не кратный k, отбрасывается — как у NumPy.
pub fn downsample_mean(z: &[f32], h: usize, w: usize, k: usize) -> (Vec<f32>, usize, usize) {
    if k <= 1 {
        return (z.to_vec(), h, w);
    }
    let (oh, ow) = (h / k, w / k);
    let mut out = vec![f32::NAN; oh.saturating_mul(ow)];
    if oh == 0 || ow == 0 {
        return (out, oh, ow);
    }
    out.par_chunks_mut(ow).enumerate().for_each(|(bi, row)| {
        for bj in 0..ow {
            let mut sum = 0f32;
            let mut n = 0u32;
            for ki in 0..k {
                let base = (bi * k + ki) * w + bj * k;
                for kj in 0..k {
                    let v = z[base + kj];
                    if v.is_finite() {
                        sum += v;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                row[bj] = sum / n as f32;
            }
        }
    });
    (out, oh, ow)
}

fn dest_index(i: usize, src_res: f64, dst_res: f64, n_out: usize) -> usize {
    let v = ((i as f64) * src_res / dst_res).floor() as usize;
    v.min(n_out - 1)
}

/// Первая исходная строка, у которой номер строки назначения ≥ `oi`.
fn first_source_row(oi: usize, h: usize, src_res: f64, dst_res: f64, nh: usize) -> usize {
    if oi == 0 {
        return 0;
    }
    let mut lo = 0usize;
    let mut hi = h;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if dest_index(mid, src_res, dst_res, nh) < oi {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Среднее по ячейкам сетки `dst_res`. Суммы — f64, в порядке строк исходного DEM.
pub fn bin_mean(z: &[f32], h: usize, w: usize, src_res: f64, dst_res: f64) -> (Vec<f32>, usize, usize) {
    let nh = ((h as f64) * src_res / dst_res).floor().max(1.0) as usize;
    let nw = ((w as f64) * src_res / dst_res).floor().max(1.0) as usize;
    if h == 0 || w == 0 {
        return (vec![f32::NAN; nh * nw], nh, nw);
    }
    let cols: Vec<usize> = (0..w).map(|j| dest_index(j, src_res, dst_res, nw)).collect();
    let mut sums = vec![0f64; nh * nw];
    let mut cnts = vec![0u32; nh * nw];
    sums.par_chunks_mut(nw)
        .zip(cnts.par_chunks_mut(nw))
        .enumerate()
        .for_each(|(oi, (srow, crow))| {
            let r0 = first_source_row(oi, h, src_res, dst_res, nh);
            let r1 = if oi + 1 == nh { h } else { first_source_row(oi + 1, h, src_res, dst_res, nh) };
            for i in r0..r1 {
                let row = &z[i * w..(i + 1) * w];
                for (j, &v) in row.iter().enumerate() {
                    if v.is_finite() {
                        let c = cols[j];
                        srow[c] += v as f64;
                        crow[c] += 1;
                    }
                }
            }
        });
    let out = sums
        .par_iter()
        .zip(cnts.par_iter())
        .map(|(&s, &c)| if c > 0 { (s / c as f64) as f32 } else { f32::NAN })
        .collect();
    (out, nh, nw)
}

/// DEM к рабочему шагу. Возвращает (пиксели, высота, ширина, фактический шаг).
pub fn work_grid(z: &[f32], h: usize, w: usize, src_res: f64, dst_res: f64) -> Result<(Vec<f32>, usize, usize, f64), String> {
    if !(src_res > 0.0 && dst_res > 0.0) {
        return Err(format!("шаг сетки должен быть > 0, получено {src_res}, {dst_res}"));
    }
    let ratio = dst_res / src_res;
    if ratio <= 1.02 {
        return Ok((z.to_vec(), h, w, src_res));
    }
    let k = py_round_pos(ratio);
    if k >= 1 && (src_res * k as f64 - dst_res).abs() <= 0.02 * dst_res {
        let (out, oh, ow) = downsample_mean(z, h, w, k as usize);
        return Ok((out, oh, ow, src_res * k as f64));
    }
    let (out, nh, nw) = bin_mean(z, h, w, src_res, dst_res);
    Ok((out, nh, nw, dst_res))
}
