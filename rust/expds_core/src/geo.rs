//! GeoJSON-полигоны → плоские массивы для `shapely.from_ragged_array`.
//!
//! Обход вложенных списков Python — самое дорогое место `metric._collect`
//! (конструктор `shapely.Polygon` на каждый объект ≈ 40 мкс). Здесь координаты
//! читаются напрямую из PyList/PyFloat. Быстрый путь берёт только «простые»
//! полигоны: тип ровно "Polygon", каждое кольцо — список пар чисел (float или int,
//! не bool, конечные), после замыкания ≥ 4 точек. Всё остальное (MultiPolygon, Point, 3D,
//! пустые и вырожденные кольца, странные типы) помечается как «не взято», и Python
//! обрабатывает такой объект официальной `_extract_geometry`. Кольцо замыкается так
//! же, как в shapely/GEOS: если первая точка != последней (сравнение f64, NaN != NaN),
//! в конец дописывается первая.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt, PyList, PyTuple};

/// Только конечные числа: NaN/inf уходят в официальную ветку (GEOS по-разному
/// трактует замыкание кольца с NaN, быстрый путь такие объекты не берёт).
fn num(o: &Bound<'_, PyAny>) -> Option<f64> {
    let v = if let Ok(f) = o.cast::<PyFloat>() {
        f.value()
    } else if o.is_exact_instance_of::<PyInt>() {
        o.extract::<f64>().ok()?
    } else {
        return None;
    };
    if v.is_finite() { Some(v) } else { None }
}

fn seq_items<'py>(o: &Bound<'py, PyAny>) -> Option<Vec<Bound<'py, PyAny>>> {
    if let Ok(l) = o.cast::<PyList>() {
        return Some(l.iter().collect());
    }
    if let Ok(t) = o.cast::<PyTuple>() {
        return Some(t.iter().collect());
    }
    None
}

fn pair(o: &Bound<'_, PyAny>) -> Option<(f64, f64)> {
    if let Ok(l) = o.cast::<PyList>() {
        if l.len() != 2 {
            return None;
        }
        return Some((num(&l.get_item(0).ok()?)?, num(&l.get_item(1).ok()?)?));
    }
    if let Ok(t) = o.cast::<PyTuple>() {
        if t.len() != 2 {
            return None;
        }
        return Some((num(&t.get_item(0).ok()?)?, num(&t.get_item(1).ok()?)?));
    }
    None
}

/// Один полигон: дописывает кольца в coords/ring_off. None — не быстрый путь (ничего не дописано).
fn polygon(g: &Bound<'_, PyAny>, coords: &mut Vec<f64>, ring_off: &mut Vec<i64>) -> Option<()> {
    let d = g.cast::<PyDict>().ok()?;
    let t = d.get_item("type").ok()??;
    if t.extract::<&str>().ok()? != "Polygon" {
        return None;
    }
    let rings = seq_items(&d.get_item("coordinates").ok()??)?;
    if rings.is_empty() {
        return None;
    }
    let c0 = coords.len();
    let r0 = ring_off.len();
    let mut ok = true;
    'outer: for ring in &rings {
        let pts = match seq_items(ring) {
            Some(p) => p,
            None => { ok = false; break; }
        };
        if pts.is_empty() {
            ok = false;
            break;
        }
        let start = coords.len();
        for p in &pts {
            match pair(p) {
                Some((x, y)) => { coords.push(x); coords.push(y); }
                None => { ok = false; break 'outer; }
            }
        }
        let (fx, fy) = (coords[start], coords[start + 1]);
        let (lx, ly) = (coords[coords.len() - 2], coords[coords.len() - 1]);
        if fx != lx || fy != ly {
            coords.push(fx);
            coords.push(fy);
        }
        if (coords.len() - start) / 2 < 4 {
            ok = false;
            break;
        }
        ring_off.push((coords.len() / 2) as i64);
    }
    if !ok {
        coords.truncate(c0);
        ring_off.truncate(r0);
        return None;
    }
    Some(())
}

/// (coords N×2 плоско, ring_offsets, geom_offsets, индексы взятых объектов).
pub fn flatten(geoms: &Bound<'_, PyList>) -> (Vec<f64>, Vec<i64>, Vec<i64>, Vec<i64>) {
    let mut coords = Vec::with_capacity(geoms.len() * 64);
    let mut ring_off = vec![0i64];
    let mut geom_off = vec![0i64];
    let mut taken = Vec::with_capacity(geoms.len());
    for (i, g) in geoms.iter().enumerate() {
        if polygon(&g, &mut coords, &mut ring_off).is_some() {
            geom_off.push((ring_off.len() - 1) as i64);
            taken.push(i as i64);
        }
    }
    (coords, ring_off, geom_off, taken)
}
