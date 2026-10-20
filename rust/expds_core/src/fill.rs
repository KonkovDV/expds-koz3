//! Заполнение пустот DEM.
//!
//! * `fill_pyramid` — многомасштабное заполнение NaN: усреднение 2×2 по
//!   конечным значениям до полного покрытия, затем билинейное увеличение
//!   (как `cv2.resize(INTER_LINEAR)`) с подстановкой только в пустые ячейки.
//!   Повторяет `_fill_nan_pyramid` бейзлайна.
//! * `convex_hull_mask` — маска выпуклой оболочки занятых ячеек: внутри неё
//!   пустоты заполняются, снаружи остаётся nodata.

use std::sync::atomic::{AtomicU32, Ordering};

use rayon::prelude::*;

/// Билинейное изменение размера с семантикой OpenCV INTER_LINEAR
/// (центры пикселей, зажим индексов на краях). Вход без NaN.
pub fn resize_linear(src: &[f32], sh: usize, sw: usize, dh: usize, dw: usize) -> Vec<f32> {
    // Таблицы индексов и весов по каждой оси.
    let axis = |s: usize, d: usize| -> Vec<(usize, usize, f32)> {
        let scale = 1.0f64 / (d as f64 / s as f64);
        (0..d)
            .map(|i| {
                let f = ((i as f64 + 0.5) * scale - 0.5) as f32;
                let mut i0 = f.floor() as i64;
                let mut frac = f - i0 as f32;
                if i0 < 0 {
                    i0 = 0;
                    frac = 0.0;
                }
                if i0 >= s as i64 - 1 {
                    i0 = s as i64 - 1;
                    frac = 0.0;
                }
                let i0 = i0 as usize;
                let i1 = (i0 + 1).min(s - 1);
                (i0, i1, frac)
            })
            .collect()
    };
    let xs = axis(sw, dw);
    let ys = axis(sh, dh);
    // Строки независимы: сначала растяжение по X, затем по Y.
    let mut rows_h = vec![0f32; sh * dw];
    rows_h.par_chunks_mut(dw).enumerate().for_each(|(r, drow)| {
        let srow = &src[r * sw..(r + 1) * sw];
        for (j, &(x0, x1, fx)) in xs.iter().enumerate() {
            drow[j] = srow[x0] * (1.0 - fx) + srow[x1] * fx;
        }
    });
    let mut out = vec![0f32; dh * dw];
    out.par_chunks_mut(dw).enumerate().for_each(|(i, o)| {
        let &(y0, y1, fy) = &ys[i];
        let a = &rows_h[y0 * dw..(y0 + 1) * dw];
        let b = &rows_h[y1 * dw..(y1 + 1) * dw];
        for j in 0..dw {
            o[j] = a[j] * (1.0 - fy) + b[j] * fy;
        }
    });
    out
}

fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// Уменьшение в 2 раза: среднее конечных значений блока 2×2 (NaN, если их нет).
fn halve(cur: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let (ph, pw) = (h + h % 2, w + w % 2);
    let (nh, nw) = (ph / 2, pw / 2);
    let get = |r: usize, c: usize| -> f32 { if r < h && c < w { cur[r * w + c] } else { f32::NAN } };
    let mut out = vec![f32::NAN; nh * nw];
    for r in 0..nh {
        for c in 0..nw {
            let mut sum = 0f32;
            let mut cnt = 0u32;
            for (dr, dc) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let v = get(2 * r + dr, 2 * c + dc);
                if v.is_finite() {
                    sum += v;
                    cnt += 1;
                }
            }
            if cnt > 0 {
                out[r * nw + c] = (sum as f64 / cnt as f64) as f32;
            }
        }
    }
    (out, nh, nw)
}

/// Заполняет все NaN многомасштабной пирамидой (см. описание модуля).
pub fn fill_pyramid(dem: &[f32], h: usize, w: usize, max_levels: usize) -> Vec<f32> {
    let mut levels: Vec<(Vec<f32>, usize, usize)> = Vec::new();
    let (mut cur, mut ch, mut cw) = (dem.to_vec(), h, w);
    for _ in 0..max_levels {
        if all_finite(&cur) || ch.min(cw) < 2 {
            break;
        }
        levels.push((cur.clone(), ch, cw));
        let (n, nh, nw) = halve(&cur, ch, cw);
        cur = n;
        ch = nh;
        cw = nw;
    }
    if !all_finite(&cur) {
        let (mut s, mut n) = (0f64, 0u64);
        for v in cur.iter().filter(|v| v.is_finite()) {
            s += *v as f64;
            n += 1;
        }
        let mean = if n > 0 { (s / n as f64) as f32 } else { 0.0 };
        for v in cur.iter_mut() {
            if !v.is_finite() {
                *v = mean;
            }
        }
    }
    for (level, lh, lw) in levels.into_iter().rev() {
        let up = resize_linear(&cur, ch, cw, lh, lw);
        cur = level.iter().zip(up.iter()).map(|(&l, &u)| if l.is_finite() { l } else { u }).collect();
        ch = lh;
        cw = lw;
    }
    cur
}

fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Выпуклая оболочка (алгоритм Эндрю), вершины против часовой стрелки.
pub fn convex_hull(mut pts: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let mut lower: Vec<(f64, f64)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(f64, f64)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Линейная интерполяция пустых ячеек по триангуляции Делоне занятых (TIN), как
/// `scipy.interpolate.LinearNDInterpolator`: вершины — центры занятых ячеек,
/// пустая ячейка получает значение треугольника, в который попадает; вне
/// выпуклой оболочки остаётся NaN. Каждый треугольник растеризуется по своей
/// рамке, поэтому время ~ площади съёмки, а не числу пар «точка — треугольник».
pub fn fill_tin(dem: &[f32], h: usize, w: usize) -> Vec<f32> {
    use delaunator::{triangulate, Point};

    let mut out = dem.to_vec();
    let (idx, pts): (Vec<usize>, Vec<Point>) = dem
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, _)| (i, Point { x: (i % w) as f64, y: (i / w) as f64 }))
        .unzip();
    if pts.len() < 3 || pts.len() == h * w {
        return out;
    }
    let tri = triangulate(&pts);
    let nt = tri.triangles.len() / 3;
    if nt >= u32::MAX as usize {
        return fill_tin_sequential(dem, h, w, &idx, &pts, &tri.triangles);
    }
    // Пустую ячейку получает первый по порядку Делоне треугольник, чей тест её принял,
    // как в последовательном обходе. Проход 1 параллельно ищет этот номер (atomic min),
    // проход 2 считает высоту по тем же формулам. Результат побитово тот же.
    let owner: Vec<AtomicU32> = (0..h * w).into_par_iter().map(|_| AtomicU32::new(u32::MAX)).collect();
    (0..nt).into_par_iter().with_min_len(1024).for_each(|ti| {
        let t = &tri.triangles[3 * ti..3 * ti + 3];
        let Some(bb) = tin_bbox(&pts[t[0]], &pts[t[1]], &pts[t[2]], h, w) else {
            return;
        };
        let (a, b, c) = (&pts[t[0]], &pts[t[1]], &pts[t[2]]);
        for y in bb.y0..=bb.y1 {
            let py = y as f64;
            let (sx0, sx1) = scan_span(a, b, c, py, bb.x0, bb.x1);
            let row = y * w;
            for x in sx0..=sx1 {
                let i = row + x;
                if dem[i].is_finite() {
                    continue;
                }
                if barycentric(a, b, c, bb.det, x as f64, py).is_some() {
                    owner[i].fetch_min(ti as u32, Ordering::Relaxed);
                }
            }
        }
    });
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let py = y as f64;
        for (x, cell) in row.iter_mut().enumerate() {
            let ti = owner[y * w + x].load(Ordering::Relaxed);
            if ti == u32::MAX {
                continue;
            }
            let t = &tri.triangles[3 * ti as usize..3 * ti as usize + 3];
            let (a, b, c) = (&pts[t[0]], &pts[t[1]], &pts[t[2]]);
            let det = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
            if let Some((l1, l2, l3)) = barycentric(a, b, c, det, x as f64, py) {
                let (za, zb, zc) = (dem[idx[t[0]]] as f64, dem[idx[t[1]]] as f64, dem[idx[t[2]]] as f64);
                *cell = (l1 * za + l2 * zb + l3 * zc) as f32;
            }
        }
    });
    out
}

struct TinBox {
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    det: f64,
}

/// Рамка треугольника на сетке и определитель; None — вырожденный или вне сетки.
fn tin_bbox(a: &delaunator::Point, b: &delaunator::Point, c: &delaunator::Point, h: usize, w: usize)
            -> Option<TinBox> {
    let det = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    if det.abs() < 1e-12 {
        return None;
    }
    let x0 = a.x.min(b.x).min(c.x).ceil().max(0.0) as usize;
    let x1 = (a.x.max(b.x).max(c.x).floor() as usize).min(w - 1);
    let y0 = a.y.min(b.y).min(c.y).ceil().max(0.0) as usize;
    let y1 = (a.y.max(b.y).max(c.y).floor() as usize).min(h - 1);
    if x0 > x1 || y0 > y1 {
        return None;
    }
    Some(TinBox { x0, x1, y0, y1, det })
}

/// Барицентрические координаты ячейки (px, py); None — ячейка вне треугольника.
#[inline]
fn barycentric(a: &delaunator::Point, b: &delaunator::Point, c: &delaunator::Point, det: f64, px: f64, py: f64)
               -> Option<(f64, f64, f64)> {
    let l1 = ((b.y - c.y) * (px - c.x) + (c.x - b.x) * (py - c.y)) / det;
    let l2 = ((c.y - a.y) * (px - c.x) + (a.x - c.x) * (py - c.y)) / det;
    let l3 = 1.0 - l1 - l2;
    if l1 >= -1e-9 && l2 >= -1e-9 && l3 >= -1e-9 { Some((l1, l2, l3)) } else { None }
}

/// Столбцы строки py, которые может задеть треугольник, с запасом в 1 ячейку.
/// Сужает обход рамки; принимает ячейку всё равно только barycentric.
fn scan_span(a: &delaunator::Point, b: &delaunator::Point, c: &delaunator::Point, py: f64, bx0: usize,
             bx1: usize) -> (usize, usize) {
    let v = [(a.x, a.y), (b.x, b.y), (c.x, c.y)];
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..3 {
        let (x1, y1) = v[i];
        let (x2, y2) = v[(i + 1) % 3];
        if (y1 - py).abs() <= 1e-8 {
            lo = lo.min(x1);
            hi = hi.max(x1);
        }
        if (y1 - py) * (y2 - py) < 0.0 {
            let x = x1 + (py - y1) * (x2 - x1) / (y2 - y1);
            lo = lo.min(x);
            hi = hi.max(x);
        }
    }
    if !lo.is_finite() {
        return (bx0, bx1);
    }
    let x0 = ((lo - 1.0).floor() as isize).clamp(bx0 as isize, bx1 as isize) as usize;
    let x1 = ((hi + 1.0).ceil() as isize).clamp(bx0 as isize, bx1 as isize) as usize;
    if x0 > x1 { (bx0, bx1) } else { (x0, x1) }
}

/// Тот же обход в одном потоке: для триангуляций больше u32::MAX треугольников.
fn fill_tin_sequential(dem: &[f32], h: usize, w: usize, idx: &[usize], pts: &[delaunator::Point],
                       triangles: &[usize]) -> Vec<f32> {
    let mut out = dem.to_vec();
    for t in triangles.chunks_exact(3) {
        let (a, b, c) = (&pts[t[0]], &pts[t[1]], &pts[t[2]]);
        let Some(bb) = tin_bbox(a, b, c, h, w) else { continue };
        let (za, zb, zc) = (dem[idx[t[0]]] as f64, dem[idx[t[1]]] as f64, dem[idx[t[2]]] as f64);
        for y in bb.y0..=bb.y1 {
            for x in bb.x0..=bb.x1 {
                let i = y * w + x;
                if out[i].is_finite() {
                    continue;
                }
                if let Some((l1, l2, l3)) = barycentric(a, b, c, bb.det, x as f64, y as f64) {
                    out[i] = (l1 * za + l2 * zb + l3 * zc) as f32;
                }
            }
        }
    }
    out
}

/// Маска выпуклой оболочки занятых ячеек (центры пикселей, границы включительно).
pub fn convex_hull_mask(valid: &[bool], h: usize, w: usize) -> Vec<bool> {
    // Для оболочки достаточно крайних занятых ячеек каждой строки.
    let mut pts = Vec::new();
    for r in 0..h {
        let row = &valid[r * w..(r + 1) * w];
        if let Some(c0) = row.iter().position(|&v| v) {
            let c1 = row.iter().rposition(|&v| v).unwrap();
            pts.push((c0 as f64, r as f64));
            pts.push((c1 as f64, r as f64));
        }
    }
    if pts.len() < 3 {
        return valid.to_vec();
    }
    let hull = convex_hull(pts);
    if hull.len() < 3 {
        return valid.to_vec();
    }
    let mut mask = vec![false; h * w];
    let ymin = hull.iter().map(|p| p.1).fold(f64::INFINITY, f64::min).max(0.0) as usize;
    let ymax = (hull.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max) as usize).min(h - 1);
    let n = hull.len();
    for y in ymin..=ymax {
        let yf = y as f64;
        let (mut xl, mut xr) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..n {
            let (a, b) = (hull[i], hull[(i + 1) % n]);
            let (lo, hi) = if a.1 <= b.1 { (a, b) } else { (b, a) };
            if yf < lo.1 - 1e-9 || yf > hi.1 + 1e-9 {
                continue;
            }
            if (hi.1 - lo.1).abs() < 1e-12 {
                xl = xl.min(lo.0.min(hi.0));
                xr = xr.max(lo.0.max(hi.0));
            } else {
                let x = lo.0 + (yf - lo.1) * (hi.0 - lo.0) / (hi.1 - lo.1);
                xl = xl.min(x);
                xr = xr.max(x);
            }
        }
        if xl > xr {
            continue;
        }
        let c0 = ((xl - 1e-9).ceil().max(0.0)) as usize;
        let c1 = ((xr + 1e-9).floor() as i64).min(w as i64 - 1);
        if c1 < 0 {
            continue;
        }
        for c in c0..=(c1 as usize) {
            mask[y * w + c] = true;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn parallel_tin_is_bitwise_sequential() {
        // Ячейки решётки дают много вырожденных четвёрок на одной окружности.
        for (h, w, hole, seed0) in [(60usize, 80usize, 0.8, 1u64), (120, 90, 0.4, 7), (40, 300, 0.95, 3)] {
            let mut seed = seed0;
            let mut dem = vec![f32::NAN; h * w];
            for (i, v) in dem.iter_mut().enumerate() {
                if lcg(&mut seed) >= hole {
                    *v = (10.0 + (i % w) as f64 * 0.01 + lcg(&mut seed) * 0.3) as f32;
                }
            }
            for r in 20..30.min(h) {
                for c in 10..40.min(w) {
                    dem[r * w + c] = f32::NAN;
                }
            }
            let par = fill_tin(&dem, h, w);
            let (idx, pts): (Vec<usize>, Vec<delaunator::Point>) = dem
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(i, _)| (i, delaunator::Point { x: (i % w) as f64, y: (i / w) as f64 }))
                .unzip();
            let tri = delaunator::triangulate(&pts);
            let seq = fill_tin_sequential(&dem, h, w, &idx, &pts, &tri.triangles);
            let filled = par.iter().zip(&dem).filter(|(p, d)| p.is_finite() && !d.is_finite()).count();
            assert!(filled > 100);
            for (i, (a, b)) in par.iter().zip(&seq).enumerate() {
                assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "ячейка {i}: {a} != {b}");
            }
        }
    }
}
