//! Локальный рельеф и признаки кандидатов.
//!
//! * `box_mean` — среднее в окне (2r+1)×(2r+1) по интегральному изображению,
//!   на краях — по попавшим в растр пикселям; NaN игнорируются.
//! * `local_relief` = DEM − box_mean: курган даёт положительный «холм».
//! * `ring_stats` — статистика диска и кольца вокруг кандидата.

use rayon::prelude::*;

/// Среднее в квадратном окне радиуса `r` (NaN не учитываются).
pub fn box_mean(z: &[f32], h: usize, w: usize, r: usize) -> Vec<f32> {
    let (h1, w1) = (h + 1, w + 1);
    // Горизонтальные префиксы строк независимы. Вертикаль потом идёт сверху вниз,
    // в том же порядке, что и однопоточный проход, поэтому суммы f64 те же.
    let mut row_s = vec![0f64; h * w1];
    let mut row_n = vec![0u32; h * w1];
    row_s.par_chunks_mut(w1).zip(row_n.par_chunks_mut(w1)).enumerate().for_each(|(i, (rs, rn))| {
        let (mut s, mut c) = (0f64, 0u32);
        let src = &z[i * w..(i + 1) * w];
        for (j, &v) in src.iter().enumerate() {
            if v.is_finite() {
                s += v as f64;
                c += 1;
            }
            rs[j + 1] = s;
            rn[j + 1] = c;
        }
    });
    let mut s = vec![0f64; h1 * w1];
    let mut n = vec![0u32; h1 * w1];
    // Столбцы независимы. Полоса в 64 столбца читается подряд по строке,
    // сумма в каждом столбце идёт сверху вниз — тот же порядок f64, что и в одном потоке.
    let tile = 64usize;
    // Адреса, а не указатели: иначе замыкание rayon не Send. Полосы пишут разные столбцы.
    let s_addr = s.as_mut_ptr() as usize;
    let n_addr = n.as_mut_ptr() as usize;
    (0..w1.div_ceil(tile)).into_par_iter().for_each(|t| {
        let j0 = t * tile;
        let j1 = (j0 + tile).min(w1);
        let width = j1 - j0;
        let mut acc_s = vec![0f64; width];
        let mut acc_n = vec![0u32; width];
        for i in 0..h {
            let src = i * w1;
            let dst = (i + 1) * w1;
            for (k, j) in (j0..j1).enumerate() {
                acc_s[k] += row_s[src + j];
                acc_n[k] += row_n[src + j];
                unsafe {
                    *(s_addr as *mut f64).add(dst + j) = acc_s[k];
                    *(n_addr as *mut u32).add(dst + j) = acc_n[k];
                }
            }
        }
    });
    let mut out = vec![f32::NAN; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(i, row)| {
        let r0 = i.saturating_sub(r);
        let r1 = (i + r + 1).min(h);
        for (j, o) in row.iter_mut().enumerate() {
            let c0 = j.saturating_sub(r);
            let c1 = (j + r + 1).min(w);
            let sum = s[r1 * w1 + c1] - s[r0 * w1 + c1] - s[r1 * w1 + c0] + s[r0 * w1 + c0];
            let cnt = n[r1 * w1 + c1] + n[r0 * w1 + c0] - n[r0 * w1 + c1] - n[r1 * w1 + c0];
            if cnt > 0 {
                *o = (sum / cnt as f64) as f32;
            }
        }
    });
    out
}

/// Уклон в градусах, ядро Хорна 3×3 (как ArcGIS и `lidar.slope_deg`). Край повторяется.
pub fn slope_deg(z: &[f32], h: usize, w: usize, res: f64) -> Vec<f32> {
    if h == 0 || w == 0 {
        return Vec::new();
    }
    let denom = 8.0 * res;
    let mut out = vec![0f32; h * w];
    // Сначала float32, как NumPy на массиве float32, и только деление — в float64.
    let at = |y: isize, x: isize| -> f32 {
        let y = y.clamp(0, h as isize - 1) as usize;
        let x = x.clamp(0, w as isize - 1) as usize;
        z[y * w + x]
    };
    out.par_chunks_mut(w).enumerate().for_each(|(i, row)| {
        let y = i as isize;
        for (j, o) in row.iter_mut().enumerate() {
            let x = j as isize;
            let dx = (at(y - 1, x + 1) + 2.0f32 * at(y, x + 1) + at(y + 1, x + 1)
                - (at(y - 1, x - 1) + 2.0f32 * at(y, x - 1) + at(y + 1, x - 1))) as f64
                / denom;
            let dy = (at(y - 1, x - 1) + 2.0f32 * at(y - 1, x) + at(y - 1, x + 1)
                - (at(y + 1, x - 1) + 2.0f32 * at(y + 1, x) + at(y + 1, x + 1))) as f64
                / denom;
            *o = dx.hypot(dy).atan().to_degrees() as f32;
        }
    });
    out
}

/// Локальный рельеф: высота минус среднее окрестности радиуса `r` пикселей.
pub fn local_relief(z: &[f32], h: usize, w: usize, r: usize) -> Vec<f32> {
    let m = box_mean(z, h, w, r);
    z.par_iter().zip(m.par_iter()).map(|(&a, &b)| a - b).collect()
}

/// Признаки кандидата: [среднее диска, среднее кольца, std кольца,
/// выраженность = диск − кольцо, min кольца, max кольца]; радиусы в пикселях.
pub fn ring_stats(z: &[f32], h: usize, w: usize, rows: &[i64], cols: &[i64],
                  r_in: f64, r_out: f64) -> Vec<[f32; 6]> {
    let ro = r_out.ceil() as i64;
    rows.par_iter()
        .zip(cols.par_iter())
        .map(|(&r, &c)| {
            let (mut ds, mut dn) = (0f64, 0u64);
            let (mut rs, mut rq, mut rn) = (0f64, 0f64, 0u64);
            let (mut rmin, mut rmax) = (f64::INFINITY, f64::NEG_INFINITY);
            for dr in -ro..=ro {
                let y = r + dr;
                if y < 0 || y >= h as i64 {
                    continue;
                }
                for dc in -ro..=ro {
                    let x = c + dc;
                    if x < 0 || x >= w as i64 {
                        continue;
                    }
                    let d = ((dr * dr + dc * dc) as f64).sqrt();
                    let v = z[y as usize * w + x as usize];
                    if !v.is_finite() {
                        continue;
                    }
                    let v = v as f64;
                    if d <= r_in {
                        ds += v;
                        dn += 1;
                    } else if d <= r_out {
                        rs += v;
                        rq += v * v;
                        rn += 1;
                        rmin = rmin.min(v);
                        rmax = rmax.max(v);
                    }
                }
            }
            let nan = f32::NAN;
            let disk = if dn > 0 { ds / dn as f64 } else { f64::NAN };
            if rn == 0 {
                return [disk as f32, nan, nan, nan, nan, nan];
            }
            let mean = rs / rn as f64;
            let var = (rq / rn as f64 - mean * mean).max(0.0);
            [disk as f32, mean as f32, var.sqrt() as f32, (disk - mean) as f32, rmin as f32, rmax as f32]
        })
        .collect()
}
