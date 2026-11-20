//! Потоковый фильтр больших GeoJSON по уверенности (замена `cpu/filter_oof.py` на ijson).
//!
//! Файл читается кусками (буфер 8 МБ), сканер отслеживает строки/экранирование и
//! глубину скобок, вырезает каждый элемент массива `features` как сырые байты.
//! Уверенность берётся как в Python: properties.confidence, иначе properties.score,
//! иначе 1.0. Разбор properties — serde_json (геометрия пропускается без построения),
//! пачками параллельно через rayon. Прошедшие объекты пишутся исходным текстом
//! (без переформатирования чисел), поэтому результат семантически равен выводу
//! Python-фильтра (json.load(a) == json.load(b)), но не побайтно.
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufWriter, Read, Write};

#[derive(Deserialize)]
struct Props {
    confidence: Option<serde_json::Value>,
    score: Option<serde_json::Value>,
}
#[derive(Deserialize)]
struct Feat {
    properties: Option<Props>,
}

fn conf_of(raw: &[u8]) -> Result<f64, String> {
    let f: Feat = serde_json::from_slice(raw).map_err(|e| format!("feature parse: {e}"))?;
    let p = match f.properties { Some(p) => p, None => return Ok(1.0) };
    // dict.get('confidence', default): ключ есть (даже null) — берётся он
    let v = match (p.confidence, p.score) {
        (Some(c), _) => c,
        (None, Some(s)) => s,
        (None, None) => return Ok(1.0),
    };
    v.as_f64().ok_or_else(|| format!("non-numeric confidence: {v}"))
}

pub struct Stats { pub total: u64, pub kept: u64 }

const HEADER: &[u8] = br#"{"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}}, "features": ["#;

pub fn filter_file(src: &str, dst: &str, thr: f64, batch: usize) -> Result<Stats, String> {
    let mut inp = File::open(src).map_err(|e| format!("{src}: {e}"))?;
    let mut out = BufWriter::with_capacity(8 << 20, File::create(dst).map_err(|e| format!("{dst}: {e}"))?);
    out.write_all(HEADER).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 8 << 20];
    // состояние сканера
    let mut depth: i32 = 0;
    let mut in_str = false;
    let mut esc = false;
    let mut feat_depth: i32 = -1; // глубина массива features (после '[')
    let mut last_key: Vec<u8> = Vec::new();
    let mut key_buf: Vec<u8> = Vec::new();
    let mut collecting_key = false;
    let mut cur: Vec<u8> = Vec::new();
    let mut in_feat = false;
    let mut pending: Vec<Vec<u8>> = Vec::with_capacity(batch);
    let mut st = Stats { total: 0, kept: 0 };
    let mut first_out = true;

    let mut flush = |pending: &mut Vec<Vec<u8>>, out: &mut BufWriter<File>, st: &mut Stats, first_out: &mut bool| -> Result<(), String> {
        let confs: Vec<Result<f64, String>> = pending.par_iter().map(|r| conf_of(r)).collect();
        for (raw, c) in pending.iter().zip(confs) {
            st.total += 1;
            if c? >= thr {
                if !*first_out { out.write_all(b", ").map_err(|e| e.to_string())?; }
                out.write_all(raw).map_err(|e| e.to_string())?;
                *first_out = false;
                st.kept += 1;
            }
        }
        pending.clear();
        Ok(())
    };

    loop {
        let n = inp.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        let chunk = &buf[..n];
        let mut seg_start = 0usize; // начало текущего куска feature в chunk
        let mut i = 0usize;
        while i < n {
            let b = chunk[i];
            if in_str {
                if collecting_key && !(b == b'"' && !esc) { key_buf.push(b); }
                if esc { esc = false; }
                else if b == b'\\' { esc = true; }
                else if b == b'"' {
                    in_str = false;
                    if collecting_key { collecting_key = false; last_key = std::mem::take(&mut key_buf); }
                }
                i += 1; continue;
            }
            match b {
                b'"' => {
                    in_str = true;
                    if !in_feat && depth == 1 { collecting_key = true; key_buf.clear(); }
                }
                b'{' | b'[' => {
                    if feat_depth >= 0 && depth == feat_depth && b == b'{' && !in_feat {
                        in_feat = true; seg_start = i;
                    }
                    depth += 1;
                    if b == b'[' && feat_depth < 0 && depth == 2 && last_key == b"features" {
                        feat_depth = depth;
                    }
                }
                b'}' | b']' => {
                    depth -= 1;
                    if in_feat && depth == feat_depth {
                        cur.extend_from_slice(&chunk[seg_start..=i]);
                        pending.push(std::mem::take(&mut cur));
                        in_feat = false;
                        if pending.len() >= batch { flush(&mut pending, &mut out, &mut st, &mut first_out)?; }
                    } else if feat_depth >= 0 && depth == feat_depth - 1 && b == b']' {
                        feat_depth = -2; // массив закончился
                    }
                }
                _ => {}
            }
            i += 1;
        }
        if in_feat { cur.extend_from_slice(&chunk[seg_start..n]); }
    }
    if depth != 0 || in_feat { return Err("truncated JSON".into()); }
    if feat_depth == -1 { return Err("no top-level 'features' array".into()); }
    flush(&mut pending, &mut out, &mut st, &mut first_out)?;
    out.write_all(b"]}").map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())?;
    Ok(st)
}
