//! `expds_core` — Rust-ускоритель для пайплайна КОЗ №3.
//!
//! Все функции имеют эквиваленты на NumPy в `expds/accel.py`; Python-код
//! выбирает Rust, если модуль собран, и NumPy в остальных случаях.
//! Тяжёлые вычисления выполняются без GIL и параллельно (rayon).

mod channels;
mod cpu;
mod delaunay;
mod fill;
mod geo;
mod jsonfilter;
mod grid;
mod peaks;
mod relief;
mod resample;
mod shade;
mod stats;

use std::borrow::Cow;

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
            PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Данные массива без копии, если он непрерывный (C-порядок); иначе — копия.
fn to_vec_f32<'a>(a: &'a PyReadonlyArray2<'_, f32>) -> (Cow<'a, [f32]>, usize, usize) {
    let (h, w) = a.as_array().dim();
    match a.as_slice() {
        Ok(s) => (Cow::Borrowed(s), h, w),
        Err(_) => (Cow::Owned(a.as_array().iter().copied().collect()), h, w),
    }
}

fn to_vec_bool<'a>(a: &'a PyReadonlyArray2<'_, bool>) -> (Cow<'a, [bool]>, usize, usize) {
    let (h, w) = a.as_array().dim();
    match a.as_slice() {
        Ok(s) => (Cow::Borrowed(s), h, w),
        Err(_) => (Cow::Owned(a.as_array().iter().copied().collect()), h, w),
    }
}

/// Версия модуля и как собран пул потоков (для логов и проверки сборки).
#[pyfunction]
fn version() -> String {
    format!("{} {}", env!("CARGO_PKG_VERSION"), cpu::describe())
}

/// LAS/LAZ → DEM (среднее z точек `ground_class`, NaN в пустых ячейках).
///
/// Возвращает (dem float32 H×W, info: dict с left, top, res, width, height,
/// factor, points_used, points_total, used_all, classes).
#[pyfunction]
#[pyo3(signature = (path, res=0.1, ground_class=2, max_pixels=60_000_000, use_all=false, keep_prob=1.0))]
fn las_grid<'py>(py: Python<'py>, path: String, res: f64, ground_class: u8, max_pixels: u64,
                 use_all: bool, keep_prob: f64) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyDict>)> {
    if res.is_nan() || res <= 0.0 {
        return Err(PyValueError::new_err("res должен быть > 0"));
    }
    let result = py
        .detach(|| grid::las_grid(&path, res, ground_class, max_pixels, use_all, keep_prob))
        .map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?;
    let s = result.spec;
    let dem = Array2::from_shape_vec((s.height, s.width), result.dem)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let info = PyDict::new(py);
    info.set_item("left", s.left)?;
    info.set_item("top", s.top)?;
    info.set_item("res", s.res)?;
    info.set_item("width", s.width)?;
    info.set_item("height", s.height)?;
    info.set_item("factor", s.factor)?;
    info.set_item("points_used", result.points_used)?;
    info.set_item("points_total", result.points_total)?;
    info.set_item("used_all", result.used_all)?;
    let classes = PyDict::new(py);
    for (c, n) in result.classes {
        classes.set_item(c, n)?;
    }
    info.set_item("classes", classes)?;
    Ok((dem.into_pyarray(py), info))
}

/// Заполняет NaN многомасштабной пирамидой (как `_fill_nan_pyramid` бейзлайна).
#[pyfunction]
#[pyo3(signature = (dem, max_levels=16))]
fn fill_pyramid<'py>(py: Python<'py>, dem: PyReadonlyArray2<'py, f32>, max_levels: usize)
                     -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (v, h, w) = to_vec_f32(&dem);
    let out = py.detach(|| fill::fill_pyramid(&v, h, w, max_levels));
    Ok(Array2::from_shape_vec((h, w), out).unwrap().into_pyarray(py))
}

/// Линейная интерполяция пустых ячеек по Делоне занятых (TIN); вне выпуклой оболочки — NaN.
#[pyfunction]
fn fill_tin<'py>(py: Python<'py>, dem: PyReadonlyArray2<'py, f32>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (v, h, w) = to_vec_f32(&dem);
    let out = py.detach(|| fill::fill_tin(&v, h, w));
    Ok(Array2::from_shape_vec((h, w), out).unwrap().into_pyarray(py))
}

/// Маска выпуклой оболочки занятых ячеек.
#[pyfunction]
fn convex_hull_mask<'py>(py: Python<'py>, valid: PyReadonlyArray2<'py, bool>)
                         -> PyResult<Bound<'py, PyArray2<bool>>> {
    let (v, h, w) = to_vec_bool(&valid);
    let out = py.detach(|| fill::convex_hull_mask(&v, h, w));
    Ok(Array2::from_shape_vec((h, w), out).unwrap().into_pyarray(py))
}

/// Билинейное изменение размера (семантика OpenCV INTER_LINEAR), вход без NaN.
#[pyfunction]
fn resize_linear<'py>(py: Python<'py>, src: PyReadonlyArray2<'py, f32>, height: usize, width: usize)
                      -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (v, h, w) = to_vec_f32(&src);
    if h == 0 || w == 0 || height == 0 || width == 0 {
        return Err(PyValueError::new_err("пустой размер"));
    }
    let out = py.detach(|| fill::resize_linear(&v, h, w, height, width));
    Ok(Array2::from_shape_vec((height, width), out).unwrap().into_pyarray(py))
}

/// Серая отмывка ESRI (uint8). `filled` — DEM без NaN.
#[pyfunction]
#[pyo3(signature = (filled, res_x, res_y, azimuth=315.0, altitude=45.0, z_factor=1.0))]
fn hillshade<'py>(py: Python<'py>, filled: PyReadonlyArray2<'py, f32>, res_x: f64, res_y: f64,
                  azimuth: f64, altitude: f64, z_factor: f64) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let (v, h, w) = to_vec_f32(&filled);
    let out = py.detach(|| shade::hillshade_u8(&v, h, w, res_x, res_y, azimuth, altitude, z_factor));
    Ok(Array2::from_shape_vec((h, w), out).unwrap().into_pyarray(py))
}

/// Цветная отмывка «Hot to cold» (uint8, 3×H×W).
#[pyfunction]
fn color_relief<'py>(py: Python<'py>, filled: PyReadonlyArray2<'py, f32>, valid: PyReadonlyArray2<'py, bool>,
                     z_min: f64, z_range: f64, cycles: f64) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let (v, h, w) = to_vec_f32(&filled);
    let (m, mh, mw) = to_vec_bool(&valid);
    if (h, w) != (mh, mw) {
        return Err(PyValueError::new_err("размеры filled и valid не совпадают"));
    }
    let out = py.detach(|| shade::color_relief(&v, &m, h, w, z_min, z_range, cycles));
    Ok(Array3::from_shape_vec((3, h, w), out).unwrap().into_pyarray(py))
}

/// (x − медиана) / (1.4826·MAD) по конечным пикселям; NaN остаются NaN.
#[pyfunction]
fn robust_z<'py>(py: Python<'py>, a: PyReadonlyArray2<'py, f32>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (v, h, w) = to_vec_f32(&a);
    let out = py.detach(|| stats::robust_z(&v));
    Ok(Array2::from_shape_vec((h, w), out).unwrap().into_pyarray(py))
}

/// Максимум значений по номеру области (0..=n), начиная с 0: как np.maximum.at.
#[pyfunction]
fn label_max<'py>(py: Python<'py>, labels: PyReadonlyArray2<'py, i32>, values: PyReadonlyArray2<'py, f32>, n: usize)
                  -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (lh, lw) = labels.as_array().dim();
    let (vh, vw) = values.as_array().dim();
    if (lh, lw) != (vh, vw) {
        return Err(PyValueError::new_err("labels и values разного размера"));
    }
    let l: Cow<[i32]> = match labels.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(labels.as_array().iter().copied().collect()),
    };
    let (v, _, _) = to_vec_f32(&values);
    let out = py.detach(|| stats::label_max(&l, &v, n));
    Ok(Array1::from(out).into_pyarray(py))
}

/// Среднее в окне (2r+1)² без учёта NaN.
#[pyfunction]
fn box_mean<'py>(py: Python<'py>, z: PyReadonlyArray2<'py, f32>, radius: usize)
                 -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (v, h, w) = to_vec_f32(&z);
    let out = py.detach(|| relief::box_mean(&v, h, w, radius));
    Ok(Array2::from_shape_vec((h, w), out).unwrap().into_pyarray(py))
}

/// DEM к рабочему шагу: (растр, фактический шаг в метрах).
#[pyfunction]
fn work_grid<'py>(py: Python<'py>, dem: PyReadonlyArray2<'py, f32>, src_res: f64, dst_res: f64)
                  -> PyResult<(Bound<'py, PyArray2<f32>>, f64)> {
    let (v, h, w) = to_vec_f32(&dem);
    let (pix, oh, ow, step) = py
        .detach(|| resample::work_grid(&v, h, w, src_res, dst_res))
        .map_err(PyValueError::new_err)?;
    Ok((Array2::from_shape_vec((oh, ow), pix).unwrap().into_pyarray(py), step))
}

/// Уклон в градусах (Хорн 3×3).
#[pyfunction]
fn slope_deg<'py>(py: Python<'py>, z: PyReadonlyArray2<'py, f32>, res: f64) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if !(res > 0.0) {
        return Err(PyValueError::new_err("шаг должен быть > 0"));
    }
    let (v, h, w) = to_vec_f32(&z);
    let out = py.detach(|| relief::slope_deg(&v, h, w, res));
    Ok(Array2::from_shape_vec((h, w), out).unwrap().into_pyarray(py))
}

/// Положительная открытость (градусы) и SVF; пустые ячейки — NaN.
#[pyfunction]
#[pyo3(signature = (z, res, radius_px, n_dir=16))]
fn openness_svf<'py>(py: Python<'py>, z: PyReadonlyArray2<'py, f32>, res: f64, radius_px: usize, n_dir: usize)
                     -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    if !(res > 0.0) || radius_px == 0 || n_dir == 0 {
        return Err(PyValueError::new_err("res, radius_px и n_dir должны быть > 0"));
    }
    let (v, h, w) = to_vec_f32(&z);
    let (o, s) = py.detach(|| relief::openness_svf(&v, h, w, res, radius_px, n_dir));
    Ok((Array2::from_shape_vec((h, w), o).unwrap().into_pyarray(py),
        Array2::from_shape_vec((h, w), s).unwrap().into_pyarray(py)))
}

/// Локальный рельеф: z − среднее окна радиуса `radius` пикселей.
#[pyfunction]
fn local_relief<'py>(py: Python<'py>, z: PyReadonlyArray2<'py, f32>, radius: usize)
                     -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (v, h, w) = to_vec_f32(&z);
    let out = py.detach(|| relief::local_relief(&v, h, w, radius));
    Ok(Array2::from_shape_vec((h, w), out).unwrap().into_pyarray(py))
}

/// Каналы модели li (uint8 C×H×W) и фактический шаг сетки.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn relief_channels<'py>(py: Python<'py>, dem: PyReadonlyArray2<'py, f32>, src_res: f64, dst_res: f64,
                        tags: Vec<String>, lrm_s_m: f64, lrm_s_clip: f64, lrm_l_m: f64, lrm_l_clip: f64,
                        slope_max: f64) -> PyResult<(Bound<'py, PyArray3<u8>>, f64)> {
    let (v, h, w) = to_vec_f32(&dem);
    let (pix, nc, oh, ow, step) = py.detach(|| {
        channels::relief_channels(&v, h, w, src_res, dst_res, &tags, lrm_s_m, lrm_s_clip, lrm_l_m,
                                  lrm_l_clip, slope_max)
    }).map_err(PyValueError::new_err)?;
    Ok((Array3::from_shape_vec((nc, oh, ow), pix).unwrap().into_pyarray(py), step))
}

/// Максимум окна в заданных пикселях: (K, H, W) → (N, K). Как maximum_filter(size)[rows, cols].
#[pyfunction]
fn sample_window_max<'py>(py: Python<'py>, img: PyReadonlyArray3<'py, f32>, rows: PyReadonlyArray1<'py, i64>,
                          cols: PyReadonlyArray1<'py, i64>, size: usize) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if rows.len() != cols.len() {
        return Err(PyValueError::new_err("rows и cols разной длины"));
    }
    if size == 0 {
        return Err(PyValueError::new_err("size должен быть > 0"));
    }
    let view = img.as_array();
    let (k, h, w) = view.dim();
    let flat: Vec<f32> = if let Some(s) = view.as_slice() {
        s.to_vec()
    } else {
        view.iter().copied().collect()
    };
    let rr: Vec<i64> = rows.as_array().iter().copied().collect();
    let cc: Vec<i64> = cols.as_array().iter().copied().collect();
    let mut out = Vec::with_capacity(rr.len() * k);
    // по каналам, затем транспонируем в N×K
    let mut per_k: Vec<Vec<f32>> = Vec::with_capacity(k);
    py.detach(|| {
        for ki in 0..k {
            let plane = &flat[ki * h * w..(ki + 1) * h * w];
            per_k.push(peaks::sample_window_max(plane, h, w, &rr, &cc, size));
        }
    });
    let n = rr.len();
    for i in 0..n {
        for plane in &per_k {
            out.push(plane[i]);
        }
    }
    Ok(Array2::from_shape_vec((n, k), out).unwrap().into_pyarray(py))
}

/// Локальные максимумы с подавлением соседей: (rows int64, cols int64, values float32).
#[pyfunction]
#[pyo3(signature = (img, radius, min_value, max_peaks=1_000_000))]
#[allow(clippy::type_complexity)]
fn find_peaks<'py>(py: Python<'py>, img: PyReadonlyArray2<'py, f32>, radius: usize, min_value: f32,
                   max_peaks: usize)
                   -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
    let (v, h, w) = to_vec_f32(&img);
    let peaks = py.detach(|| peaks::find_peaks(&v, h, w, radius, min_value, max_peaks));
    let rows: Vec<i64> = peaks.iter().map(|p| p.0).collect();
    let cols: Vec<i64> = peaks.iter().map(|p| p.1).collect();
    let vals: Vec<f32> = peaks.iter().map(|p| p.2).collect();
    Ok((Array1::from(rows).into_pyarray(py), Array1::from(cols).into_pyarray(py),
        Array1::from(vals).into_pyarray(py)))
}

/// Признаки кандидатов (N×6): диск, кольцо, std кольца, выраженность, min и max кольца.
#[pyfunction]
fn ring_stats<'py>(py: Python<'py>, z: PyReadonlyArray2<'py, f32>, rows: PyReadonlyArray1<'py, i64>,
                   cols: PyReadonlyArray1<'py, i64>, r_in: f64, r_out: f64)
                   -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (v, h, w) = to_vec_f32(&z);
    if rows.len() != cols.len() {
        return Err(PyValueError::new_err("rows и cols разной длины"));
    }
    let rr: Vec<i64> = rows.as_array().iter().copied().collect();
    let cc: Vec<i64> = cols.as_array().iter().copied().collect();
    let stats = py.detach(|| relief::ring_stats(&v, h, w, &rr, &cc, r_in, r_out));
    let n = stats.len();
    let flat: Vec<f32> = stats.into_iter().flatten().collect();
    Ok(Array2::from_shape_vec((n, 6), flat).unwrap().into_pyarray(py))
}

/// GeoJSON-словари геометрий → (coords float64 N×2, ring_offsets int64, geom_offsets int64, taken int64).
///
/// Вход для `shapely.from_ragged_array(POLYGON, coords, (ring_offsets, geom_offsets))`.
/// `taken` — индексы входа, взятые быстрым путём (простые Polygon); остальные
/// Python разбирает официальной `_extract_geometry`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn flatten_polygons<'py>(py: Python<'py>, geoms: &Bound<'py, PyList>)
    -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>,
                 Bound<'py, PyArray1<i64>>)> {
    let (coords, ring_off, geom_off, taken) = geo::flatten(geoms);
    let n = coords.len() / 2;
    Ok((Array2::from_shape_vec((n, 2), coords).unwrap().into_pyarray(py),
        Array1::from(ring_off).into_pyarray(py), Array1::from(geom_off).into_pyarray(py),
        Array1::from(taken).into_pyarray(py)))
}

/// Есть ли в байтах `n` и более ASCII-цифр подряд (проверка перед orjson: длинное целое
/// orjson превратил бы во float, а json — в int; тогда scoreboard читает стандартным json).
#[pyfunction]
fn has_digit_run(data: &[u8], n: usize) -> bool {
    if n == 0 {
        return true;
    }
    let mut run = 0usize;
    for &b in data {
        if b.is_ascii_digit() {
            run += 1;
            if run >= n {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

/// Потоковый фильтр GeoJSON: объекты с confidence (или score, иначе 1.0) >= thr → dst.
/// Возвращает (всего, оставлено). GIL отпускается; разбор properties параллельно (rayon).
#[pyfunction]
#[pyo3(signature = (src, dst, thr, batch=20000))]
fn filter_geojson_conf(py: Python<'_>, src: String, dst: String, thr: f64, batch: usize) -> PyResult<(u64, u64)> {
    let st = py.detach(|| jsonfilter::filter_file(&src, &dst, thr, batch.max(1))).map_err(PyValueError::new_err)?;
    Ok((st.total, st.kept))
}

#[pymodule]
fn expds_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    cpu::init();
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(las_grid, m)?)?;
    m.add_function(wrap_pyfunction!(fill_pyramid, m)?)?;
    m.add_function(wrap_pyfunction!(fill_tin, m)?)?;
    m.add_function(wrap_pyfunction!(convex_hull_mask, m)?)?;
    m.add_function(wrap_pyfunction!(resize_linear, m)?)?;
    m.add_function(wrap_pyfunction!(hillshade, m)?)?;
    m.add_function(wrap_pyfunction!(color_relief, m)?)?;
    m.add_function(wrap_pyfunction!(box_mean, m)?)?;
    m.add_function(wrap_pyfunction!(robust_z, m)?)?;
    m.add_function(wrap_pyfunction!(label_max, m)?)?;
    m.add_function(wrap_pyfunction!(local_relief, m)?)?;
    m.add_function(wrap_pyfunction!(work_grid, m)?)?;
    m.add_function(wrap_pyfunction!(slope_deg, m)?)?;
    m.add_function(wrap_pyfunction!(openness_svf, m)?)?;
    m.add_function(wrap_pyfunction!(find_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(relief_channels, m)?)?;
    m.add_function(wrap_pyfunction!(sample_window_max, m)?)?;
    m.add_function(wrap_pyfunction!(ring_stats, m)?)?;
    m.add_function(wrap_pyfunction!(flatten_polygons, m)?)?;
    m.add_function(wrap_pyfunction!(has_digit_run, m)?)?;
    m.add_function(wrap_pyfunction!(filter_geojson_conf, m)?)?;
    Ok(())
}
