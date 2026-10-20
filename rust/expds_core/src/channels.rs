//! Каналы модели лидара одним вызовом: рабочая сетка, локальный рельеф, уклон, uint8.
//!
//! Формулы те же, что у `lidar.relief_channels` (NumPy): пустые ячейки — 0,
//! остальные — round(1 + unit·254) как `np.rint` (половинки к чётному).

use rayon::prelude::*;

use crate::relief::{local_relief, slope_deg};
use crate::resample::{py_round_pos, work_grid};
use crate::shade::hillshade_u8;

fn np_rint_u8(x: f32) -> u8 {
    if !x.is_finite() || x <= 0.0 {
        return 0;
    }
    let base = x.floor();
    let frac = x - base;
    let n = if (frac - 0.5).abs() <= 1.0e-5 {
        let i = base as i64;
        if i.rem_euclid(2) == 0 { i } else { i + 1 }
    } else {
        x.round() as i64
    };
    n.clamp(0, 255) as u8
}

fn radius_px(meters: f64, step: f64) -> usize {
    py_round_pos(meters / step).max(1) as usize
}

/// `tags` — `lrm_s`, `lrm_l`, `slope`, `g` в нужном порядке.
/// Возвращает (пиксели C×H×W, каналы, высота, ширина, шаг).
pub fn relief_channels(
    dem: &[f32],
    h: usize,
    w: usize,
    src_res: f64,
    dst_res: f64,
    tags: &[String],
    lrm_s_m: f64,
    lrm_s_clip: f64,
    lrm_l_m: f64,
    lrm_l_clip: f64,
    slope_max: f64,
) -> Result<(Vec<u8>, usize, usize, usize, f64), String> {
    let (z, oh, ow, step) = work_grid(dem, h, w, src_res, dst_res)?;
    let n = oh * ow;
    let mut sum = 0f64;
    let mut nfin = 0u64;
    for &v in &z {
        if v.is_finite() {
            sum += v as f64;
            nfin += 1;
        }
    }
    let mean = if nfin > 0 { (sum / nfin as f64) as f32 } else { 0.0 };
    let mut filled = z.clone();
    for v in &mut filled {
        if !v.is_finite() {
            *v = mean;
        }
    }
    let mut planes: Vec<Vec<f32>> = Vec::with_capacity(tags.len());
    for tag in tags {
        let unit = match tag.as_str() {
            "lrm_s" | "lrm_l" => {
                let (meters, clip) = if tag == "lrm_s" {
                    (lrm_s_m, lrm_s_clip)
                } else {
                    (lrm_l_m, lrm_l_clip)
                };
                let lr = local_relief(&z, oh, ow, radius_px(meters, step));
                let clip = clip as f32;
                lr.into_iter()
                    .map(|v| {
                        let u = if v.is_finite() { v } else { 0.0 };
                        (u / clip).clamp(-1.0, 1.0) * 0.5 + 0.5
                    })
                    .collect()
            }
            "slope" => {
                let sl = slope_deg(&filled, oh, ow, step);
                let scale = slope_max as f32;
                sl.into_iter().map(|v| (v / scale).clamp(0.0, 1.0)).collect()
            }
            "g" => {
                let g = hillshade_u8(&filled, oh, ow, step, step, 315.0, 45.0, 1.0);
                g.into_iter().map(|v| v as f32 / 255.0).collect()
            }
            other => return Err(format!("неизвестный канал li: {other}")),
        };
        planes.push(unit);
    }
    let nc = planes.len();
    let mut out = vec![0u8; nc * n];
    out.par_chunks_mut(n).enumerate().for_each(|(ci, plane)| {
        let unit = &planes[ci];
        for i in 0..n {
            if z[i].is_finite() {
                plane[i] = np_rint_u8(1.0f32 + unit[i] * 254.0f32);
            }
        }
    });
    Ok((out, nc, oh, ow, step))
}
