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

/// Максимум значения по номеру области: out[l] = max(0, max values[i] при labels[i] == l).
/// Как `np.maximum.at(zeros(n + 1), labels, values)` для конечных values; номера вне 0..=n пропускаются.
pub fn label_max(labels: &[i32], values: &[f32], n: usize) -> Vec<f32> {
    use rayon::prelude::*;

    let chunk = (labels.len() / (rayon::current_num_threads() * 4)).max(1 << 16);
    labels
        .par_chunks(chunk)
        .zip(values.par_chunks(chunk))
        .fold(
            || vec![0f32; n + 1],
            |mut acc, (ls, vs)| {
                for (&l, &v) in ls.iter().zip(vs) {
                    if l >= 0 && (l as usize) <= n && v > acc[l as usize] {
                        acc[l as usize] = v;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0f32; n + 1],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    if y > *x {
                        *x = y;
                    }
                }
                a
            },
        )
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
