//! Пул потоков под i5-13600KF (Raptor Lake): 6 P-ядер × 2 потока, без E-ядер.
//!
//! У E-ядер Gracemont команда AVX2 режется на две по 128 бит, и если отдать им
//! работу, планировщик уводит потоки с P-ядер. Двенадцати потоков P-ядер хватает,
//! чтобы забить два канала памяти. Число потоков переопределяется `EXPDS_THREADS`.
//! На Windows у рабочих потоков выключен EcoQoS, чтобы Thread Director держал их
//! на быстрых ядрах.

use std::sync::Once;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        let n = thread_count();
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("expds-{i}"))
            .spawn_handler(|thread| {
                std::thread::Builder::new().spawn(move || {
                    prefer_performance();
                    thread.run();
                })?;
                Ok(())
            })
            .build_global();
        prefer_performance();
    });
}

pub fn describe() -> String {
    let compiled = cfg!(target_feature = "avx2");
    let runtime = std::is_x86_feature_detected!("avx2");
    format!("avx2={compiled}/{runtime} threads={}", rayon::current_num_threads())
}

fn thread_count() -> usize {
    if let Ok(s) = std::env::var("EXPDS_THREADS") {
        if let Ok(n) = s.trim().parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    performance_threads().unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
}

#[cfg(windows)]
fn performance_threads() -> Option<usize> {
    // GetLogicalProcessorInformationEx(RelationProcessorCore): EfficiencyClass
    // больше у быстрых ядер. Считаем логические процессоры только этого класса.
    #[link(name = "kernel32")]
    extern "system" {
        fn GetLogicalProcessorInformationEx(relationship: u32, buffer: *mut u8, returned: *mut u32) -> i32;
    }
    const RELATION_PROCESSOR_CORE: u32 = 0;
    let mut len = 0u32;
    unsafe {
        GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len);
    }
    if len == 0 {
        return None;
    }
    let mut buf = vec![0u8; len as usize];
    let ok = unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf.as_mut_ptr(), &mut len) };
    if ok == 0 {
        return None;
    }
    let mut best = 0u8;
    let mut counts: Vec<(u8, usize)> = Vec::new();
    let mut off = 0usize;
    while off + 8 <= buf.len() {
        let rel = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        let size = u32::from_le_bytes(buf[off + 4..off + 8].try_into().unwrap()) as usize;
        if size < 8 || off + size > buf.len() {
            break;
        }
        // PROCESSOR_RELATIONSHIP: Flags, EfficiencyClass, Reserved[20], GroupCount, GroupMask[]
        if rel == RELATION_PROCESSOR_CORE && size >= 8 + 24 {
            let base = off + 8;
            let eff = buf[base + 1];
            let groups = u16::from_le_bytes(buf[base + 22..base + 24].try_into().unwrap()) as usize;
            let mut logical = 0usize;
            for g in 0..groups {
                let mask_at = base + 24 + g * 16;
                if mask_at + 8 > off + size {
                    break;
                }
                let mask = u64::from_le_bytes(buf[mask_at..mask_at + 8].try_into().unwrap());
                logical += mask.count_ones() as usize;
            }
            best = best.max(eff);
            if logical > 0 {
                counts.push((eff, logical));
            }
        }
        off += size;
    }
    let n: usize = counts.iter().filter(|(e, _)| *e == best).map(|(_, n)| *n).sum();
    if n > 0 { Some(n) } else { None }
}

#[cfg(not(windows))]
fn performance_threads() -> Option<usize> {
    None
}

#[cfg(windows)]
fn prefer_performance() {
    // StateMask = 0 при ControlMask = EXECUTION_SPEED снимает EcoQoS.
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadInformation(thread: isize, class: u32, info: *const u8, len: u32) -> i32;
    }
    const THREAD_POWER_THROTTLING: u32 = 3;
    const EXECUTION_SPEED: u32 = 1;
    #[repr(C)]
    struct State {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }
    let state = State { version: 1, control_mask: EXECUTION_SPEED, state_mask: 0 };
    unsafe {
        SetThreadInformation(
            GetCurrentThread(),
            THREAD_POWER_THROTTLING,
            &state as *const State as *const u8,
            std::mem::size_of::<State>() as u32,
        );
    }
}

#[cfg(not(windows))]
fn prefer_performance() {}
