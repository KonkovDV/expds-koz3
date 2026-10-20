//! Робастная нормировка растра: (x − медиана) / (1.4826 · MAD).
//!
//! Медиана — выбор k-го элемента (`select_nth_unstable`), а не полная сортировка.
//! Для чётного числа значений — среднее двух центральных, как `numpy.median`.
//! NaN не входят в оценку и остаются NaN на выходе.

fn order(a: &f32, b: &f32) -> std::cmp::Ordering {
    a.total_cmp(b)
}

/// Медиана. `v` переставляется. Пустой срез — NaN.
fn median(v: &mut [f32]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    v.select_nth_unstable_by(mid, order);
    if n % 2 == 1 {
        return v[mid] as f64;
    }
    let hi = v[mid] as f64;
    let lo = v[..mid].iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    (lo + hi) / 2.0
}

pub fn robust_z(a: &[f32]) -> Vec<f32> {
    let mut fin: Vec<f32> = a.iter().copied().filter(|v| v.is_finite()).collect();
    if fin.is_empty() {
        return vec![f32::NAN; a.len()];
    }
    let med = median(&mut fin);
    for v in &mut fin {
        *v = (*v as f64 - med).abs() as f32;
    }
    let mut mad = median(&mut fin) * 1.4826;
    if mad == 0.0 {
        mad = 1.0;
    }
    a.iter()
        .map(|&v| if v.is_finite() { ((v as f64 - med) / mad) as f32 } else { f32::NAN })
        .collect()
}
