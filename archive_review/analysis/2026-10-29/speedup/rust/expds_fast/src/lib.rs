//! LAS → регулярная сетка: накопление по ячейкам за один кусок точек.
//!
//! Семантика (и побитный результат) та же, что у numpy-кода
//!   x = X*sx + ox (float64), y = …, z = float32(Z*sz + oz)
//!   ix = int64((x - x0) / res), iy = int64((y1 - y) / res), в сетке W×H
//!   режим 0/1: zsum += bincount(k, z), cnt += bincount(k)   (сумма куска с 0.0 по порядку точек)
//!   режим 2:   minimum.at(zmin, k, z)
//! Фильтр классов: режим 0 — только класс 2; режимы 1, 2 — без классов 7 и 18.
//! Порядок сложения внутри ячейки — порядок точек, поэтому float64-суммы совпадают до бита.
use numpy::{PyArray1, PyReadonlyArray1, PyReadwriteArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline]
fn keep(mode: u8, c: u8) -> bool {
    if mode == 0 { c == 2 } else { c != 7 && c != 18 }
}

/// Общее ядро: pt(i) → (X, Y, Z, класс) точки i куска.
#[allow(clippy::too_many_arguments)]
fn accumulate_core<P: Fn(usize) -> (i32, i32, i32, u8) + Sync>(
    n: usize,
    pt: P,
    (sx, sy, sz): (f64, f64, f64),
    (ox, oy, oz): (f64, f64, f64),
    x0: f64,
    y1: f64,
    res: f64,
    w: i64,
    h: i64,
    mode: u8,
    zsum: &mut [f64],
    cnt: &mut [i32],
    zmin: &mut [f32],
) -> usize {
    let cells = (w as u64) * (h as u64);
        // 1) ключ ячейки каждой точки (u32::MAX — отброшена), параллельно
    let mut keys = vec![0u32; n];
    keys.par_chunks_mut(1 << 15).enumerate().for_each(|(b, kk)| {
        let base = b << 15;
        for (j, kv) in kk.iter_mut().enumerate() {
            let i = base + j;
            *kv = u32::MAX;
            let (xi, yi, _, ci) = pt(i);
            if !keep(mode, ci) {
            continue;
            }
            let x = (xi as f64) * sx + ox;
            let y = (yi as f64) * sy + oy;
            let ix = ((x - x0) / res) as i64;
            let iy = ((y1 - y) / res) as i64;
            if ix >= 0 && ix < w && iy >= 0 && iy < h {
            *kv = (iy * w + ix) as u32;
            }
        }
    });
    let used = keys.iter().filter(|&&k| k != u32::MAX).count();
    let zf = |i: usize| ((pt(i).2 as f64) * sz + oz) as f32;
    // 2) разбиение ячеек на полосы; каждая полоса проходит все точки по порядку → порядок сложения
    //    внутри ячейки совпадает с bincount, а потоки пишут в непересекающиеся куски
    let nt = rayon::current_num_threads().max(1);
    let band = ((cells as usize) + nt - 1) / nt;
    let band = band.max(1);
    if mode == 2 {
        zmin.par_chunks_mut(band).enumerate().for_each(|(b, zm)| {
            let lo = (b * band) as u32;
            let hi = lo + zm.len() as u32;
            for (i, &k) in keys.iter().enumerate() {
            if k >= lo && k < hi {
                let z = zf(i);
                let c = &mut zm[(k - lo) as usize];
                if !(*c < z) {
                    *c = z;
                }
            }
            }
        });
        return used;
    }
    // Сумма куска по ячейке считается с 0.0 и лишь затем прибавляется к zsum (как zsum += bincount).
    // Если до этого куска в ячейке точек не было (cnt == 0), то 0.0 + частичная сумма = сама частичная
    // сумма, и копим прямо в zsum. Иначе (ячейка на стыке кусков) — отдельная частичная сумма.
    zsum.par_chunks_mut(band).zip(cnt.par_chunks_mut(band)).enumerate().for_each(|(b, (zsb, cb))| {
        let lo = (b * band) as u32;
        let hi = lo + zsb.len() as u32;
        let mut seen = vec![0u64; (zsb.len() + 63) / 64];
        let mut part: std::collections::HashMap<u32, (f64, i32)> = std::collections::HashMap::new();
        for (i, &k) in keys.iter().enumerate() {
            if k < lo || k >= hi {
            continue;
            }
            let j = (k - lo) as usize;
            let z = zf(i) as f64;
            let (wi, bit) = (j >> 6, 1u64 << (j & 63));
            if seen[wi] & bit == 0 {
            seen[wi] |= bit;
            if cb[j] != 0 {
                part.insert(k, (0.0 + z, 1));
                continue;
            }
            } else if !part.is_empty() {
            if let Some(e) = part.get_mut(&k) {
                e.0 += z;
                e.1 += 1;
                continue;
            }
            }
            zsb[j] += z;
            cb[j] += 1;
        }
        for (k, (ps, pc)) in part {
            let j = (k - lo) as usize;
            zsb[j] += ps;
            cb[j] += pc;
        }
    });
    used
}

fn check(n: usize, w: i64, h: i64, mode: u8, lz: usize, lc: usize, lm: usize) -> PyResult<()> {
    let cells = (w as u64) * (h as u64);
    if mode > 2 || w <= 0 || h <= 0 || cells >= (u32::MAX as u64) || (n as u64) >= (1u64 << 32) {
        return Err(PyValueError::new_err("режим 0..2, W*H и число точек < 2^32"));
    }
    // неиспользуемые режимом массивы можно передать пустыми
    let need = |len: usize, used: bool| len as u64 == cells || (!used && len == 0);
    if !need(lz, mode < 2) || !need(lc, mode < 2) || !need(lm, mode == 2) {
        return Err(PyValueError::new_err("zsum, cnt (режимы 0, 1) и zmin (режим 2) должны иметь W*H элементов"));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (xr, yr, zr, cls, scale, offset, x0, y1, res, w, h, mode, zsum, cnt, zmin))]
#[allow(clippy::too_many_arguments)]
fn grid_accumulate<'py>(
    py: Python<'py>,
    xr: PyReadonlyArray1<'py, i32>,
    yr: PyReadonlyArray1<'py, i32>,
    zr: PyReadonlyArray1<'py, i32>,
    cls: PyReadonlyArray1<'py, u8>,
    scale: (f64, f64, f64),
    offset: (f64, f64, f64),
    x0: f64,
    y1: f64,
    res: f64,
    w: i64,
    h: i64,
    mode: u8,
    mut zsum: PyReadwriteArray1<'py, f64>,
    mut cnt: PyReadwriteArray1<'py, i32>,
    mut zmin: PyReadwriteArray1<'py, f32>,
) -> PyResult<usize> {
    let n = xr.len();
    if yr.len() != n || zr.len() != n || cls.len() != n {
        return Err(PyValueError::new_err("длины X, Y, Z, classification различаются"));
    }
    check(n, w, h, mode, zsum.len(), cnt.len(), zmin.len())?;
    let (xs, ys, zs, cs) = (xr.as_slice()?, yr.as_slice()?, zr.as_slice()?, cls.as_slice()?);
    let (zsum, cnt, zmin) = (zsum.as_slice_mut()?, cnt.as_slice_mut()?, zmin.as_slice_mut()?);
    Ok(py.detach(move || {
        accumulate_core(n, |i| (xs[i], ys[i], zs[i], cs[i]), scale, offset, x0, y1, res, w, h, mode, zsum, cnt, zmin)
    }))
}

/// То же по сырым записям точек LAS (байтовый буфер куска, rec_len байт на точку): X, Y, Z — int32 LE
/// по смещениям 0, 4, 8; класс — байт cls_off & cls_mask (форматы 0–5: 15 и 0x1F, 6–10: 16 и 0xFF).
/// Без копирования полей на стороне Python.
#[pyfunction]
#[pyo3(signature = (buf, rec_len, cls_off, cls_mask, scale, offset, x0, y1, res, w, h, mode, zsum, cnt, zmin))]
#[allow(clippy::too_many_arguments)]
fn grid_accumulate_raw<'py>(
    py: Python<'py>,
    buf: PyReadonlyArray1<'py, u8>,
    rec_len: usize,
    cls_off: usize,
    cls_mask: u8,
    scale: (f64, f64, f64),
    offset: (f64, f64, f64),
    x0: f64,
    y1: f64,
    res: f64,
    w: i64,
    h: i64,
    mode: u8,
    mut zsum: PyReadwriteArray1<'py, f64>,
    mut cnt: PyReadwriteArray1<'py, i32>,
    mut zmin: PyReadwriteArray1<'py, f32>,
) -> PyResult<usize> {
    if rec_len < 12 || cls_off >= rec_len || buf.len() % rec_len != 0 {
        return Err(PyValueError::new_err("rec_len ≥ 12, cls_off < rec_len, длина буфера кратна rec_len"));
    }
    let n = buf.len() / rec_len;
    check(n, w, h, mode, zsum.len(), cnt.len(), zmin.len())?;
    let b = buf.as_slice()?;
    let (zsum, cnt, zmin) = (zsum.as_slice_mut()?, cnt.as_slice_mut()?, zmin.as_slice_mut()?);
    let rd = move |o: usize| i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    Ok(py.detach(move || {
        accumulate_core(
            n,
            |i| {
                let o = i * rec_len;
                (rd(o), rd(o + 4), rd(o + 8), b[o + cls_off] & cls_mask)
            },
            scale, offset, x0, y1, res, w, h, mode, zsum, cnt, zmin,
        )
    }))
}

/// g = float32(zsum / max(cnt, 1)) там, где cnt > 0, иначе NaN — побитно как
/// np.where(cnt > 0, zsum / np.maximum(cnt, 1), np.nan).astype(np.float32).
#[pyfunction]
fn grid_mean<'py>(
    py: Python<'py>,
    zsum: PyReadonlyArray1<'py, f64>,
    cnt: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if zsum.len() != cnt.len() {
        return Err(PyValueError::new_err("длины zsum и cnt различаются"));
    }
    let (s, c) = (zsum.as_slice()?, cnt.as_slice()?);
    let out: Vec<f32> = py.detach(|| {
        let mut out = vec![0f32; s.len()];
        out.par_chunks_mut(1 << 16).enumerate().for_each(|(b, o)| {
            let base = b << 16;
            for (j, v) in o.iter_mut().enumerate() {
                let i = base + j;
                *v = if c[i] > 0 { (s[i] / (c[i] as f64)) as f32 } else { f32::NAN };
            }
        });
        out
    });
    Ok(PyArray1::from_vec(py, out))
}

#[pymodule]
fn expds_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(grid_accumulate, m)?)?;
    m.add_function(wrap_pyfunction!(grid_mean, m)?)?;
    m.add_function(wrap_pyfunction!(grid_accumulate_raw, m)?)?;
    Ok(())
}
