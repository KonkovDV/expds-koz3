//! `expds_core` — Rust-ускоритель для пайплайна КОЗ №3.
//!
//! Все функции имеют эквиваленты на NumPy в `expds/accel.py`; Python-код
//! выбирает Rust, если модуль собран, и NumPy в остальных случаях.
//! Тяжёлые вычисления выполняются без GIL и параллельно (rayon).

mod fill;
mod grid;
mod peaks;
mod relief;
mod shade;

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn to_vec_f32(a: &PyReadonlyArray2<'_, f32>) -> (Vec<f32>, usize, usize) {
    let v = a.as_array();
    let (h, w) = v.dim();
    (v.iter().copied().collect(), h, w)
}

fn to_vec_bool(a: &PyReadonlyArray2<'_, bool>) -> (Vec<bool>, usize, usize) {
    let v = a.as_array();
    let (h, w) = v.dim();
    (v.iter().copied().collect(), h, w)
}

/// Версия модуля (для логов и проверки сборки).
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// LAS/LAZ → DEM (среднее z точек `ground_class`, NaN в пустых ячейках).
///
/// Возвращает (dem float32 H×W, info: dict с left, top, res, width, height,
/// factor, points_used, points_total, used_all, classes).
#[pyfunction]
#[pyo3(signature = (path, res=0.1, ground_class=2, max_pixels=60_000_000, use_all=false))]
fn las_grid<'py>(py: Python<'py>, path: String, res: f64, ground_class: u8, max_pixels: u64,
                 use_all: bool) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyDict>)> {
    if res.is_nan() || res <= 0.0 {
        return Err(PyValueError::new_err("res должен быть > 0"));
    }
    let result = py
        .detach(|| grid::las_grid(&path, res, ground_class, max_pixels, use_all))
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

/// Среднее в окне (2r+1)² без учёта NaN.
#[pyfunction]
fn box_mean<'py>(py: Python<'py>, z: PyReadonlyArray2<'py, f32>, radius: usize)
                 -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (v, h, w) = to_vec_f32(&z);
    let out = py.detach(|| relief::box_mean(&v, h, w, radius));
    Ok(Array2::from_shape_vec((h, w), out).unwrap().into_pyarray(py))
}

/// Локальный рельеф: z − среднее окна радиуса `radius` пикселей.
#[pyfunction]
fn local_relief<'py>(py: Python<'py>, z: PyReadonlyArray2<'py, f32>, radius: usize)
                     -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (v, h, w) = to_vec_f32(&z);
    let out = py.detach(|| relief::local_relief(&v, h, w, radius));
    Ok(Array2::from_shape_vec((h, w), out).unwrap().into_pyarray(py))
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

#[pymodule]
fn expds_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(las_grid, m)?)?;
    m.add_function(wrap_pyfunction!(fill_pyramid, m)?)?;
    m.add_function(wrap_pyfunction!(convex_hull_mask, m)?)?;
    m.add_function(wrap_pyfunction!(resize_linear, m)?)?;
    m.add_function(wrap_pyfunction!(hillshade, m)?)?;
    m.add_function(wrap_pyfunction!(color_relief, m)?)?;
    m.add_function(wrap_pyfunction!(box_mean, m)?)?;
    m.add_function(wrap_pyfunction!(local_relief, m)?)?;
    m.add_function(wrap_pyfunction!(find_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(ring_stats, m)?)?;
    Ok(())
}
