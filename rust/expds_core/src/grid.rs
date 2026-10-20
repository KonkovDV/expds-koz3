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
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

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

/// Поля заголовка несжатого LAS (1.0–1.4), нужные для чтения записей точек напрямую.
struct RawHeader {
    point_offset: u64,
    format: u8,
    record_len: usize,
    count: u64,
    scale: [f64; 3],
    offset: [f64; 3],
    bounds: [f64; 4], // min_x, min_y, max_x, max_y
}

fn raw_header(path: &str) -> std::io::Result<Option<RawHeader>> {
    let mut head = Vec::with_capacity(375);
    File::open(path)?.take(375).read_to_end(&mut head)?;
    if head.len() < 227 || &head[0..4] != b"LASF" {
        return Ok(None);
    }
    let u16_at = |o: usize| u16::from_le_bytes([head[o], head[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes(head[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(head[o..o + 8].try_into().unwrap());
    let f64_at = |o: usize| f64::from_le_bytes(head[o..o + 8].try_into().unwrap());
    let fmt = head[104];
    if fmt & 0xC0 != 0 || fmt & 0x3F > 10 {
        return Ok(None); // LAZ (сжатие — в старших битах формата) или неизвестный формат
    }
    let legacy = u32_at(107) as u64;
    let count = if (head[24], head[25]) >= (1, 4) && head.len() >= 255 && u64_at(247) > 0 { u64_at(247) } else { legacy };
    Ok(Some(RawHeader {
        point_offset: u32_at(96) as u64,
        format: fmt & 0x3F,
        record_len: u16_at(105) as usize,
        count,
        scale: [f64_at(131), f64_at(139), f64_at(147)],
        offset: [f64_at(155), f64_at(163), f64_at(171)],
        bounds: [f64_at(187), f64_at(203), f64_at(179), f64_at(195)],
    }))
}

/// Читает буфер целиком или до конца файла; возвращает число прочитанных байт.
fn read_full<R: Read>(r: &mut R, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..])? {
            0 => break,
            n => got += n,
        }
    }
    Ok(got)
}

/// Как `accumulate`, но по сырым записям: координаты и ячейки считаются параллельно,
/// суммы копятся в исходном порядке точек (тот же результат, что у последовательного кода).
fn accumulate_raw(path: &str, hdr: &RawHeader, spec: GridSpec, ground_class: u8, use_all: bool,
                  histogram: &mut Option<[u64; 256]>) -> std::io::Result<(Accumulator, u64)> {
    let rec = hdr.record_len;
    let (cls_off, cls_mask) = if hdr.format <= 5 { (15usize, 0x1Fu8) } else { (16usize, 0xFFu8) };
    if rec <= cls_off {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "длина записи точки меньше формата"));
    }
    let mut f = BufReader::with_capacity(8 << 20, File::open(path)?);
    f.seek(SeekFrom::Start(hdr.point_offset))?;
    let mut acc = Accumulator::new(spec);
    let chunk = 2_000_000usize;
    let mut buf = vec![0u8; chunk * rec];
    let (mut remaining, mut total) = (hdr.count, 0u64);
    let ([sx, sy, sz], [ox, oy, oz]) = (hdr.scale, hdr.offset);
    let coord = |r: &[u8], o: usize| i32::from_le_bytes(r[o..o + 4].try_into().unwrap()) as f64;
    while remaining > 0 {
        let want = remaining.min(chunk as u64) as usize * rec;
        let got = read_full(&mut f, &mut buf[..want])?;
        let n = got / rec;
        if n == 0 {
            break;
        }
        let bytes = &buf[..n * rec];
        if let Some(h) = histogram.as_mut() {
            let part = bytes
                .par_chunks_exact(rec)
                .fold(|| [0u64; 256], |mut a, r| {
                    a[(r[cls_off] & cls_mask) as usize] += 1;
                    a
                })
                .reduce(|| [0u64; 256], |mut a, b| {
                    a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x += y);
                    a
                });
            h.iter_mut().zip(part.iter()).for_each(|(x, y)| *x += y);
        }
        let s = spec;
        let mut cells: Vec<(usize, f64)> = bytes
            .par_chunks_exact(rec)
            .filter_map(|r| {
                if !use_all && r[cls_off] & cls_mask != ground_class {
                    return None;
                }
                let x = coord(r, 0) * sx + ox;
                let y = coord(r, 4) * sy + oy;
                let col = ((x - s.left) / s.res).floor();
                let row = ((s.top - y) / s.res).floor();
                if col < 0.0 || row < 0.0 || col as usize >= s.width || row as usize >= s.height {
                    return None;
                }
                Some((row as usize * s.width + col as usize, coord(r, 8) * sz + oz))
            })
            .collect();
        // Стабильная сортировка по номеру ячейки: внутри ячейки точки остаются
        // в порядке файла, сумма z не меняется. Дальше запись идёт подряд по сетке,
        // а не случайными прыжками по всей DEM (так же устроен bincount).
        cells.sort_by_key(|c| c.0);
        for (i, z) in cells {
            acc.sums[i] += z;
            acc.counts[i] += 1;
            acc.used += 1;
        }
        total += n as u64;
        remaining -= n as u64;
        if got < want {
            break; // файл короче, чем обещает заголовок
        }
    }
    Ok((acc, total))
}

/// Читает LAS/LAZ потоково и строит DEM (средняя высота точек земли).
///
/// Несжатый LAS читается по сырым записям параллельно; LAZ — через крейт `las`.
pub fn las_grid(path: &str, res: f64, ground_class: u8, max_pixels: u64, use_all: bool) -> Result<GridResult, String> {
    let raw = raw_header(path).map_err(|e| e.to_string())?;
    let bounds = match &raw {
        Some(h) => h.bounds,
        None => {
            let reader = Reader::from_path(path).map_err(|e| e.to_string())?;
            let b = reader.header().bounds();
            [b.min.x, b.min.y, b.max.x, b.max.y]
        }
    };
    let spec = grid_spec(bounds[0], bounds[1], bounds[2], bounds[3], res, max_pixels);
    let batch = 2_000_000u64;
    let run = |all: bool, hist: &mut Option<[u64; 256]>| -> Result<(Accumulator, u64), String> {
        match &raw {
            Some(h) => accumulate_raw(path, h, spec, ground_class, all, hist).map_err(|e| e.to_string()),
            None => accumulate(path, spec, ground_class, all, hist, batch).map_err(|e| e.to_string()),
        }
    };
    let mut hist: Option<[u64; 256]> = Some([0u64; 256]);
    let (mut acc, total) = run(use_all, &mut hist)?;
    let mut used_all = use_all;
    if acc.used == 0 && !use_all {
        // В файле нет точек земли: берём все точки (как бейзлайн).
        let mut none: Option<[u64; 256]> = None;
        acc = run(true, &mut none)?.0;
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
