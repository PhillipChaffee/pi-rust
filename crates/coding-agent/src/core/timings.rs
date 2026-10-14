//! Central timing instrumentation for startup profiling, upstream's
//! `src/core/timings.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Enabled with `PI_TIMING=1`. The namespace map is process-global, the way
//! upstream's module-scope `Map` is; every accessor takes the `PI_TIMING`
//! flag explicitly so tests can drive the machinery without environment
//! mutation (`_with` forms mirror the crate's other env seams).

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// The process-global namespace map, upstream's `timingNamespaces`.
static TIMING_NAMESPACES: LazyLock<Mutex<HashMap<String, TimingNamespace>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Whether `PI_TIMING=1` is set, upstream's module-load `ENABLED` read.
#[must_use]
pub fn is_enabled() -> bool {
    std::env::var("PI_TIMING").is_ok_and(|value| value == "1")
}

/// Whether `PI_TIMING=1` is set in the supplied lookup.
#[must_use]
pub fn is_enabled_with(value: Option<&str>) -> bool {
    value == Some("1")
}

/// One instrumented namespace, upstream's `TimingNamespace`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TimingNamespace {
    timings: Vec<(String, u64)>,
    last_time: u128,
}

/// Reset a namespace's timings, upstream's `resetTimings`. No-op while
/// disabled.
pub fn reset_timings(namespace: &str) {
    if !is_enabled() {
        return;
    }
    reset_timings_with(namespace, is_enabled(), now_ms());
}

/// The injectable form of [`reset_timings`]; `now_ms` pins the wall clock.
pub fn reset_timings_with(namespace: &str, enabled: bool, now_ms: u128) {
    if !enabled {
        return;
    }
    let mut namespaces = TIMING_NAMESPACES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    namespaces.insert(
        namespace.to_string(),
        TimingNamespace {
            timings: Vec::new(),
            last_time: now_ms,
        },
    );
}

/// Record one timing delta since the namespace's last mark, upstream's
/// `time`. Lazily resets the namespace; no-op while disabled.
pub fn time(label: &str, namespace: &str) {
    if !is_enabled() {
        return;
    }
    time_with(label, namespace, is_enabled(), now_ms());
}

/// The injectable form of [`time`].
pub fn time_with(label: &str, namespace: &str, enabled: bool, now_ms: u128) {
    if !enabled {
        return;
    }
    TIMING_NAMESPACES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .entry(namespace.to_string())
        .and_modify(|namespace| {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "millisecond deltas fit u64 for startup profiling"
            )]
            namespace.timings.push((
                label.to_string(),
                (now_ms.saturating_sub(namespace.last_time)) as u64,
            ));
            namespace.last_time = now_ms;
        })
        .or_insert_with(|| TimingNamespace {
            timings: Vec::new(),
            last_time: now_ms,
        });
}

/// Print every namespace's timings to stderr, upstream's `printTimings`.
/// No-op while disabled.
pub fn print_timings() {
    if !is_enabled() {
        return;
    }
    print_timings_with(is_enabled());
}

/// The injectable form of [`print_timings`]; `now_ms` pins the wall clock.
pub fn print_timings_with(enabled: bool) {
    if !enabled {
        return;
    }
    let namespaces = TIMING_NAMESPACES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let groups: Vec<(String, Vec<(String, u64)>)> = namespaces
        .iter()
        .map(|(name, namespace)| (name.clone(), namespace.timings.clone()))
        .collect();
    drop(namespaces);
    for (name, timings) in &groups {
        print_timing_group(&format!("Startup Timings: {name}"), timings);
    }
}

/// One namespace's timing block, upstream's `printTimingGroup`.
#[allow(
    clippy::print_stderr,
    reason = "timings report to stderr, upstream's console.error"
)]
fn print_timing_group(title: &str, timings: &[(String, u64)]) {
    // Upstream filters `ms >= 0`; the Rust deltas saturate, so the filter
    // is vacuous and drops.
    let printable: Vec<&(String, u64)> = timings.iter().collect();
    if printable.is_empty() {
        return;
    }
    eprintln!("\n--- {title} ---");
    for (label, ms) in &printable {
        eprintln!("  {label}: {ms}ms");
    }
    let total: u64 = printable.iter().map(|(_, ms)| *ms).sum();
    eprintln!("  TOTAL: {total}ms");
    eprintln!("{}\n", "-".repeat(title.len() + 8));
}

/// The wall-clock millisecond mark, upstream's `Date.now()`.
fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests;
