//! Поиск локальных максимумов с подавлением соседей.
//!
//! Пиксель — кандидат, если он равен максимуму своего окна (2r+1)×(2r+1)
//! и не меньше `min_value`. Кандидаты сортируются по убыванию значения
//! (при равенстве — по строке, затем по столбцу) и принимаются жадно:
//! отбрасывается всё, что ближе `r` пикселей к уже принятому пику.
//! Используется для центров курганов (по локальному рельефу или карте
//! вероятностей) и для точечных аномалий магниторазведки.

use rayon::prelude::*;

const NEG: f32 = f32::NEG_INFINITY;

/// Максимум окна 2r+1 вдоль строки за O(1) на элемент (ван Херк — Гиль — Верман).
///
/// Строка дополняется −∞ на r с обеих сторон; g — максимум от начала блока
/// длины 2r+1, h — до конца блока; окно [j, j+2r] задевает не больше двух
/// блоков, поэтому его максимум = max(h[j], g[j+2r]).
fn sliding_max_row(src: &[f32], r: usize, g: &mut [f32], h: &mut [f32], out: &mut [f32]) {
    let n = src.len();
    let k = 2 * r + 1;
    let l = n + 2 * r;
    let at = |i: usize| if i >= r && i < r + n { src[i - r] } else { NEG };
    for i in 0..l {
        let v = at(i);
        g[i] = if i % k == 0 { v } else { g[i - 1].max(v) };
    }
    for i in (0..l).rev() {
        let v = at(i);
        h[i] = if i % k == k - 1 || i == l - 1 { v } else { h[i + 1].max(v) };
    }
    for (j, o) in out.iter_mut().enumerate() {
        *o = h[j].max(g[j + 2 * r]);
    }
}

/// Тот же приём по столбцам для полосы строк [r0, r0+rows): префиксы и суффиксы
/// блоков считаются целыми строками, поэтому доступ к памяти последовательный.
fn sliding_max_cols(src: &[f32], hgt: usize, w: usize, r: usize, r0: usize, out: &mut [f32]) {
    let rows = out.len() / w;
    let k = 2 * r + 1;
    let l = rows + 2 * r;
    let neg_row = vec![NEG; w];
    let row_at = |i: usize| -> &[f32] {
        let y = (r0 + i) as isize - r as isize;
        if y >= 0 && (y as usize) < hgt { &src[y as usize * w..(y as usize + 1) * w] } else { &neg_row }
    };
    let mut g = vec![NEG; l * w];
    let mut h = vec![NEG; l * w];
    for i in 0..l {
        let row = row_at(i);
        if i % k == 0 {
            g[i * w..(i + 1) * w].copy_from_slice(row);
        } else {
            let (prev, cur) = g.split_at_mut(i * w);
            for ((o, &a), &b) in cur[..w].iter_mut().zip(&prev[(i - 1) * w..]).zip(row) {
                *o = a.max(b);
            }
        }
    }
    for i in (0..l).rev() {
        let row = row_at(i);
        if i % k == k - 1 || i == l - 1 {
            h[i * w..(i + 1) * w].copy_from_slice(row);
        } else {
            let (cur, next) = h.split_at_mut((i + 1) * w);
            for ((o, &a), &b) in cur[i * w..].iter_mut().zip(&next[..w]).zip(row) {
                *o = a.max(b);
            }
        }
    }
    for (jj, orow) in out.chunks_exact_mut(w).enumerate() {
        let (hs, ge) = (&h[jj * w..(jj + 1) * w], &g[(jj + 2 * r) * w..(jj + 2 * r + 1) * w]);
        for ((o, &a), &b) in orow.iter_mut().zip(hs).zip(ge) {
            *o = a.max(b);
        }
    }
}

/// Максимальный фильтр окна (2r+1)², NaN/±∞ не участвуют (как scipy.ndimage.maximum_filter, mode="nearest").
pub fn max_filter(img: &[f32], h: usize, w: usize, r: usize) -> Vec<f32> {
    if h == 0 || w == 0 {
        return Vec::new();
    }
    let mut tmp = vec![NEG; h * w];
    tmp.par_chunks_mut(w).enumerate().for_each_init(
        || (vec![NEG; w + 2 * r], vec![NEG; w + 2 * r], vec![NEG; w]),
        |(g, hb, clean), (i, row)| {
            for (c, &v) in clean.iter_mut().zip(&img[i * w..(i + 1) * w]) {
                *c = if v.is_finite() { v } else { NEG };
            }
            sliding_max_row(clean, r, g, hb, row);
        },
    );
    let band = (2 * (2 * r + 1)).max(64).min(h);
    let mut out = vec![NEG; h * w];
    out.par_chunks_mut(band * w).enumerate().for_each(|(b, chunk)| {
        sliding_max_cols(&tmp, h, w, r, b * band, chunk);
    });
    out
}

/// Возвращает (строка, столбец, значение) принятых пиков.
pub fn find_peaks(img: &[f32], h: usize, w: usize, r: usize, min_value: f32,
                  max_peaks: usize) -> Vec<(i64, i64, f32)> {
    let mf = max_filter(img, h, w, r);
    let mut cand: Vec<(i64, i64, f32)> = (0..h * w)
        .into_par_iter()
        .filter_map(|i| {
            let v = img[i];
            if v.is_finite() && v >= min_value && v == mf[i] {
                Some(((i / w) as i64, (i % w) as i64, v))
            } else {
                None
            }
        })
        .collect();
    cand.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap().then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    let cell = r.max(1) as i64;
    let r2 = (r * r) as i64;
    let mut grid: std::collections::HashMap<(i64, i64), Vec<(i64, i64)>> = std::collections::HashMap::new();
    let mut out = Vec::new();
    for (pr, pc, v) in cand {
        if out.len() >= max_peaks {
            break;
        }
        let (gr, gc) = (pr / cell, pc / cell);
        let mut close = false;
        'outer: for dr in -1..=1 {
            for dc in -1..=1 {
                if let Some(list) = grid.get(&(gr + dr, gc + dc)) {
                    for &(qr, qc) in list {
                        let d2 = (qr - pr) * (qr - pr) + (qc - pc) * (qc - pc);
                        if d2 <= r2 {
                            close = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
        if !close {
            grid.entry((gr, gc)).or_default().push((pr, pc));
            out.push((pr, pc, v));
        }
    }
    out
}

/// Индекс как у `scipy.ndimage` mode='reflect': отражение относительно края пикселя.
fn reflect_index(i: isize, n: isize) -> isize {
    if n <= 1 {
        return 0;
    }
    let m = n * 2 - 2;
    let mut ii = i.rem_euclid(m);
    if ii >= n {
        ii = m - ii;
    }
    ii
}

/// Максимум окна `size`×`size` в точках (rows, cols). Совпадает с maximum_filter(size)[row, col].
pub fn sample_window_max(img: &[f32], h: usize, w: usize, rows: &[i64], cols: &[i64], size: usize) -> Vec<f32> {
    let rad = (size / 2) as isize;
    let hi = size as isize - rad;
    rows.par_iter().zip(cols.par_iter()).map(|(&r, &c)| {
        let mut best = f32::NEG_INFINITY;
        for dr in -rad..hi {
            let y = reflect_index(r as isize + dr, h as isize) as usize;
            for dc in -rad..hi {
                let x = reflect_index(c as isize + dc, w as isize) as usize;
                best = best.max(img[y * w + x]);
            }
        }
        if best.is_finite() { best } else { f32::NAN }
    }).collect()
}
