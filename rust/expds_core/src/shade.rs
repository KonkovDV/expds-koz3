//! Отмывки рельефа в формате датасета (`g`, `i`, `c`, `ch`).
//!
//! * `hillshade_u8` — формула ESRI/ArcGIS (азимут 315°, высота 45°, Z = 1),
//!   градиенты как `numpy.gradient` (центральные разности, на краях —
//!   односторонние), вычисления в f32, округление к чётному.
//! * `color_relief` — палитра «Hot to cold» (TerraModeler):
//!   hue = frac(0.75 − cycles·t), S = 1, V = 209/255; вне съёмки — 0.
//!
//! Результат совпадает с numpy-реализацией бейзлайна с точностью ±1 уровень
//! яркости на единичных пикселях (разные реализации sin/cos/atan в f32).

use rayon::prelude::*;

#[inline]
fn grad_at(z: &[f32], n: usize, stride: usize, base: usize, i: usize, d: f32) -> f32 {
    if n < 2 {
        return 0.0;
    }
    if i == 0 {
        (z[base + stride] - z[base]) / d
    } else if i == n - 1 {
        (z[base + i * stride] - z[base + (i - 1) * stride]) / d
    } else {
        (z[base + (i + 1) * stride] - z[base + (i - 1) * stride]) / (2.0 * d)
    }
}

/// Серая отмывка 0..255 (uint8). `z` — заполненный DEM без NaN, строки сверху вниз.
#[allow(clippy::too_many_arguments)]
pub fn hillshade_u8(z: &[f32], h: usize, w: usize, res_x: f64, res_y: f64,
                    azimuth_deg: f64, altitude_deg: f64, z_factor: f64) -> Vec<u8> {
    let zenith = (90.0 - altitude_deg).to_radians();
    let cos_zen = zenith.cos() as f32;
    let sin_zen = zenith.sin() as f32;
    let az = azimuth_deg.to_radians() as f32;
    let (rx, ry, zf) = (res_x as f32, res_y as f32, z_factor as f32);
    let mut out = vec![0u8; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(r, row)| {
        for (c, px) in row.iter_mut().enumerate() {
            // ось 0 — строки (вниз = юг), ось 1 — столбцы (вправо = восток)
            let gy_north = -grad_at(z, h, w, c, r, ry);
            let gx_east = grad_at(z, w, 1, r * w, c, rx);
            let slope = (gx_east.hypot(gy_north) * zf).atan();
            let aspect = (-gx_east).atan2(-gy_north);
            let shade = 255.0f32 * (cos_zen * slope.cos() + sin_zen * slope.sin() * (az - aspect).cos());
            *px = shade.clamp(0.0, 255.0).round_ties_even() as u8;
        }
    });
    out
}

/// Остаток от деления с семантикой Python/NumPy (знак делителя).
#[inline]
fn pymod(a: f32, b: f32) -> f32 {
    let m = a % b;
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) { m + b } else { m }
}

/// Цветная отмывка «Hot to cold»: возвращает 3×H×W (поканально), uint8.
pub fn color_relief(z: &[f32], valid: &[bool], h: usize, w: usize,
                    z_min: f64, z_range: f64, cycles: f64) -> Vec<u8> {
    const PHASE: f32 = 0.75;
    let value: f32 = (209.0f64 / 255.0) as f32;
    let (zmin, zr, cyc) = (z_min as f32, z_range as f32, cycles as f32);
    let n = h * w;
    let mut out = vec![0u8; 3 * n];
    let (rb, rest) = out.split_at_mut(n);
    let (gb, bb) = rest.split_at_mut(n);
    rb.par_iter_mut()
        .zip(gb.par_iter_mut())
        .zip(bb.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((r, g), b))| {
            if !valid[i] {
                return;
            }
            let t = (z[i] - zmin) / zr;
            let hue = pymod(PHASE - cyc * t, 1.0);
            let h6 = pymod(hue, 1.0) * 6.0;
            let k = (h6.floor() as i32).rem_euclid(6);
            let f = h6 - h6.floor();
            let v = value;
            // S = 1, поэтому p = V·(1 − S) = 0, q = V·(1 − f), t = V·(1 − (1 − f))
            let p = 0.0f32;
            let q = v * (1.0 - f);
            let tt = v * (1.0 - (1.0 - f));
            let (rr, gg, bv) = match k {
                0 => (v, tt, p),
                1 => (q, v, p),
                2 => (p, v, tt),
                3 => (p, q, v),
                4 => (tt, p, v),
                _ => (v, p, q),
            };
            let cv = |x: f32| (x * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
            *r = cv(rr);
            *g = cv(gg);
            *b = cv(bv);
        });
    out
}
