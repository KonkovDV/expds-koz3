//! Поиск локальных максимумов с подавлением соседей.
//!
//! Пиксель — кандидат, если он равен максимуму своего окна (2r+1)×(2r+1)
//! и не меньше `min_value`. Кандидаты сортируются по убыванию значения
//! (при равенстве — по строке, затем по столбцу) и принимаются жадно:
//! отбрасывается всё, что ближе `r` пикселей к уже принятому пику.
//! Используется для центров курганов (по локальному рельефу или карте
//! вероятностей) и для точечных аномалий магниторазведки.

use rayon::prelude::*;

fn max_filter(img: &[f32], h: usize, w: usize, r: usize) -> Vec<f32> {
    let clean = |v: f32| if v.is_finite() { v } else { f32::NEG_INFINITY };
    let mut tmp = vec![f32::NEG_INFINITY; h * w];
    tmp.par_chunks_mut(w).enumerate().for_each(|(i, row)| {
        let src = &img[i * w..(i + 1) * w];
        for (j, px) in row.iter_mut().enumerate() {
            let (a, b) = (j.saturating_sub(r), (j + r + 1).min(w));
            *px = src[a..b].iter().fold(f32::NEG_INFINITY, |m, &v| m.max(clean(v)));
        }
    });
    let mut out = vec![f32::NEG_INFINITY; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(i, row)| {
        let (a, b) = (i.saturating_sub(r), (i + r + 1).min(h));
        for j in 0..w {
            let mut m = f32::NEG_INFINITY;
            for k in a..b {
                m = m.max(tmp[k * w + j]);
            }
            row[j] = m;
        }
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
