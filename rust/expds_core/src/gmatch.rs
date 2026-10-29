//! Консервативный префильтр пар «кандидат ↔ эталон» для жадного сопоставления метрики КОЗ-3.
//!
//! R-дерево (rstar) по рамкам эталонов + площадь пересечения geo (i_overlay), параллельно (rayon).
//! Возвращает только пары, которые МОГУТ дать hit официального `_match_quality`
//! (cov ≥ thr − eps_cov для эталона-полигона, dist(centroid, pt) ≤ r + eps_d для эталона-точки).
//! Сам hit/score считает Python официальной функцией только на этих парах, поэтому результат
//! побайтно равен полному перебору при условии, что ошибка geo ≪ eps (проверяется тестом).
use ::geo::{Area, BooleanOps, BoundingRect, Centroid, Coord, LineString, MultiPolygon, Polygon, Rect};
use rayon::prelude::*;
use rstar::primitives::{GeomWithData, Rectangle};
use rstar::{RTree, AABB};

pub struct Ragged<'a> {
    pub xy: &'a [f64],   // N×2
    pub ring: &'a [i64], // смещения колец в координатах
    pub part: &'a [i64], // смещения полигонов в кольцах
    pub geom: &'a [i64], // смещения мультиполигонов в полигонах
}

impl Ragged<'_> {
    pub fn len(&self) -> usize { self.geom.len().saturating_sub(1) }
    pub fn get(&self, i: usize) -> MultiPolygon<f64> {
        let mut polys = Vec::new();
        for p in self.geom[i] as usize..self.geom[i + 1] as usize {
            let mut rings: Vec<LineString<f64>> = Vec::new();
            for r in self.part[p] as usize..self.part[p + 1] as usize {
                let (a, b) = (self.ring[r] as usize, self.ring[r + 1] as usize);
                rings.push(LineString::from((a..b).map(|k| Coord { x: self.xy[2 * k], y: self.xy[2 * k + 1] })
                    .collect::<Vec<_>>()));
            }
            if rings.is_empty() { continue; }
            let ext = rings.remove(0);
            polys.push(Polygon::new(ext, rings));
        }
        MultiPolygon::new(polys)
    }
}

type Node = GeomWithData<Rectangle<[f64; 2]>, (bool, usize)>; // (точка?, индекс в своём массиве)

fn rect_of(r: &Rect<f64>) -> Rectangle<[f64; 2]> {
    Rectangle::from_corners([r.min().x, r.min().y], [r.max().x, r.max().y])
}

/// Пары (pred, gt_id) после префильтра; gt_id — исходный индекс эталона. Отсортированы по (pred, gt_id).
#[allow(clippy::too_many_arguments)]
pub fn prefilter(pred: &Ragged, gpoly: &Ragged, gpoly_id: &[i64], gpt: &[f64], gpt_id: &[i64],
                 r_tol: f64, thr: f64, eps_cov: f64, eps_d: f64) -> (Vec<i64>, Vec<i64>) {
    let gp: Vec<MultiPolygon<f64>> = (0..gpoly.len()).into_par_iter().map(|i| gpoly.get(i)).collect();
    let ga: Vec<f64> = gp.par_iter().map(|g| g.unsigned_area()).collect();
    let mut nodes: Vec<Node> = Vec::with_capacity(gp.len() + gpt_id.len());
    for (i, g) in gp.iter().enumerate() {
        if let Some(r) = g.bounding_rect() { nodes.push(GeomWithData::new(rect_of(&r), (false, i))); }
    }
    let rr = r_tol * (1.0 + 1e-9) + 1e-6;
    for i in 0..gpt_id.len() {
        let (x, y) = (gpt[2 * i], gpt[2 * i + 1]);
        nodes.push(GeomWithData::new(Rectangle::from_corners([x - rr, y - rr], [x + rr, y + rr]), (true, i)));
    }
    let tree = RTree::bulk_load(nodes);
    let per: Vec<Vec<i64>> = (0..pred.len()).into_par_iter().map(|i| {
        let p = pred.get(i);
        let Some(pb) = p.bounding_rect() else { return Vec::new() };
        let env = AABB::from_corners([pb.min().x, pb.min().y], [pb.max().x, pb.max().y]);
        let mut pa: Option<f64> = None;
        let mut cen: Option<Option<Coord<f64>>> = None;
        let mut out = Vec::new();
        for n in tree.locate_in_envelope_intersecting(&env) {
            let (is_pt, k) = n.data;
            if is_pt {
                let c = *cen.get_or_insert_with(|| p.centroid().map(|c| c.0));
                // нет центроида (пустая фигура) — оставляем пару, решит официальный код
                let keep = match c {
                    Some(c) => (c.x - gpt[2 * k]).hypot(c.y - gpt[2 * k + 1]) <= r_tol + eps_d,
                    None => true,
                };
                if keep { out.push(gpt_id[k]); }
            } else {
                let a = *pa.get_or_insert_with(|| p.unsigned_area());
                let m = a.min(ga[k]);
                if !(m > 0.0) { continue; } // официально: min_area ≤ 0 → cov = 0
                let gb = n.geom();
                let (lo, hi) = (gb.lower(), gb.upper());
                let iw = pb.max().x.min(hi[0]) - pb.min().x.max(lo[0]);
                let ih = pb.max().y.min(hi[1]) - pb.min().y.max(lo[1]);
                // площадь пересечения ≤ площади пересечения рамок
                if iw.max(0.0) * ih.max(0.0) < (thr - eps_cov) * m * (1.0 - 1e-9) { continue; }
                let g = &gp[k];
                let inter = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.intersection(g).unsigned_area()));
                let keep = match inter {
                    Ok(v) if v.is_finite() => v / m >= thr - eps_cov,
                    _ => true, // сбой geo — консервативно оставить
                };
                if keep { out.push(gpoly_id[k]); }
            }
        }
        out.sort_unstable();
        out
    }).collect();
    let n: usize = per.iter().map(|v| v.len()).sum();
    let (mut qp, mut qg) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for (i, v) in per.into_iter().enumerate() {
        for g in v { qp.push(i as i64); qg.push(g); }
    }
    (qp, qg)
}
