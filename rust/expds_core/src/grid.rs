//! LAS/LAZ → регулярная сетка высот (DEM) по точкам земли.
//!
//! Логика повторяет бейзлайн организаторов (`build_dem_from_las`):
//! * сетка: left = floor(min_x/res)*res, bottom = floor(min_y/res)*res,
//!   width = ceil((max_x-left)/res)+1, height = ceil((max_y-bottom)/res)+1;
//! * если ячеек больше `max_pixels`, шаг увеличивается в ceil(sqrt(N/max)) раз;
//! * значение ячейки — среднее z точек класса `ground_class` (обычно 2);
//! * если таких точек нет, берутся все точки;
//! * пустые ячейки = NaN (их заполняет отдельный шаг, см. fill.rs).

use las::{PointDataBuilder, Reader};

/// Параметры сетки DEM.
#[derive(Clone, Copy, Debug)]
pub struct GridSpec {
    pub left: f64,
    pub top: f64,
    pub res: f64,
    pub width: usize,
    pub height: usize,
    /// Во сколько раз увеличен шаг из-за ограничения `max_pixels`.
    pub factor: u32,
}

/// Строит параметры сетки по габаритам облака (как бейзлайн).
pub fn grid_spec(min_x: f64, min_y: f64, max_x: f64, max_y: f64, res: f64, max_pixels: u64) -> GridSpec {
    let extent = |step: f64| {
        let left = (min_x / step).floor() * step;
        let bottom = (min_y / step).floor() * step;
        let width = ((max_x - left) / step).ceil() as usize + 1;
        let height = ((max_y - bottom) / step).ceil() as usize + 1;
        (left, bottom, width, height)
    };
    let (mut left, mut bottom, mut width, mut height) = extent(res);
    let mut step = res;
    let mut factor = 1u32;
    let cells = (width as u64) * (height as u64);
    if max_pixels > 0 && cells > max_pixels {
        factor = ((cells as f64) / (max_pixels as f64)).sqrt().ceil() as u32;
        step = res * factor as f64;
        let e = extent(step);
        left = e.0;
        bottom = e.1;
        width = e.2;
        height = e.3;
    }
    GridSpec { left, top: bottom + height as f64 * step, res: step, width, height, factor }
}

/// Накопитель сумм и счётчиков по ячейкам.
pub struct Accumulator {
    pub spec: GridSpec,
    pub sums: Vec<f64>,
    pub counts: Vec<u32>,
    pub used: u64,
}

impl Accumulator {
    pub fn new(spec: GridSpec) -> Self {
        let n = spec.width * spec.height;
        Accumulator { spec, sums: vec![0.0; n], counts: vec![0; n], used: 0 }
    }

    /// Добавляет точку; точки вне сетки отбрасываются.
    #[inline]
    pub fn add(&mut self, x: f64, y: f64, z: f64) {
        let s = &self.spec;
        let col = ((x - s.left) / s.res).floor();
        let row = ((s.top - y) / s.res).floor();
        if col < 0.0 || row < 0.0 {
            return;
        }
        let (col, row) = (col as usize, row as usize);
        if col >= s.width || row >= s.height {
            return;
        }
        let idx = row * s.width + col;
        self.sums[idx] += z;
        self.counts[idx] += 1;
        self.used += 1;
    }

    /// Средняя высота по ячейкам; пустые ячейки — NaN.
    pub fn mean_dem(&self) -> Vec<f32> {
        self.sums
            .iter()
            .zip(self.counts.iter())
            .map(|(s, &c)| if c > 0 { (s / c as f64) as f32 } else { f32::NAN })
            .collect()
    }
}

/// Результат построения DEM из LAS.
pub struct GridResult {
    pub dem: Vec<f32>,
    pub spec: GridSpec,
    pub points_used: u64,
    pub points_total: u64,
    pub classes: Vec<(u8, u64)>,
    pub used_all: bool,
}

fn accumulate(path: &str, spec: GridSpec, ground_class: u8, use_all: bool,
              histogram: &mut Option<[u64; 256]>, batch: u64) -> Result<(Accumulator, u64), las::Error> {
    let mut reader = Reader::from_path(path)?;
    let mut acc = Accumulator::new(spec);
    let mut pd = PointDataBuilder::new().for_header(reader.header()).build();
    let mut total = 0u64;
    loop {
        let n = reader.fill_points(batch, &mut pd)?;
        if n == 0 {
            break;
        }
        total += n;
        for (((x, y), z), c) in pd.x().zip(pd.y()).zip(pd.z()).zip(pd.classification()) {
            if let Some(h) = histogram.as_mut() {
                h[c as usize] += 1;
            }
            if use_all || c == ground_class {
                acc.add(x, y, z);
            }
        }
    }
    Ok((acc, total))
}

/// Читает LAS/LAZ потоково и строит DEM (средняя высота точек земли).
pub fn las_grid(path: &str, res: f64, ground_class: u8, max_pixels: u64, use_all: bool) -> Result<GridResult, las::Error> {
    let reader = Reader::from_path(path)?;
    let b = reader.header().bounds();
    drop(reader);
    let spec = grid_spec(b.min.x, b.min.y, b.max.x, b.max.y, res, max_pixels);
    let batch = 2_000_000u64;
    let mut hist: Option<[u64; 256]> = Some([0u64; 256]);
    let (mut acc, total) = accumulate(path, spec, ground_class, use_all, &mut hist, batch)?;
    let mut used_all = use_all;
    if acc.used == 0 && !use_all {
        // В файле нет точек земли: берём все точки (как бейзлайн).
        let mut none: Option<[u64; 256]> = None;
        acc = accumulate(path, spec, ground_class, true, &mut none, batch)?.0;
        used_all = true;
    }
    let classes = hist
        .unwrap_or([0u64; 256])
        .iter()
        .enumerate()
        .filter(|(_, &n)| n > 0)
        .map(|(c, &n)| (c as u8, n))
        .collect();
    Ok(GridResult { dem: acc.mean_dem(), spec, points_used: acc.used, points_total: total, classes, used_all })
}
