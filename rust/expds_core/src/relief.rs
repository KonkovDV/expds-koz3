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
    let mut s = vec![0f64; h1 * w1];
    let mut n = vec![0u32; h1 * w1];
    for i in 0..h {
        let (mut rs, mut rn) = (0f64, 0u32);
        for j in 0..w {
            let v = z[i * w + j];
            if v.is_finite() {
                rs += v as f64;
                rn += 1;
            }
            s[(i + 1) * w1 + j + 1] = s[i * w1 + j + 1] + rs;
            n[(i + 1) * w1 + j + 1] = n[i * w1 + j + 1] + rn;
        }
    }
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
