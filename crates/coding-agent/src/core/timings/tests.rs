#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use super::*;

#[test]
fn disabled_flag_leaves_namespaces_untouched() {
    let mut namespaces = TIMING_NAMESPACES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    namespaces.clear();
    drop(namespaces);
    time_with("load", "main", false, 10);
    assert!(
        TIMING_NAMESPACES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    );
}

#[test]
fn enabled_flag_records_deltas_and_resets() {
    let mut namespaces = TIMING_NAMESPACES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    namespaces.clear();
    drop(namespaces);
    reset_timings_with("main", true, 0);
    time_with("resourceLoader.reload", "main", true, 100);
    time_with("session", "main", true, 150);
    time_with("other", "main", true, 175);
    let mut namespaces = TIMING_NAMESPACES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let namespace = &namespaces["main"];
    assert_eq!(
        namespace.timings,
        vec![
            ("resourceLoader.reload".to_string(), 100),
            ("session".to_string(), 50),
            ("other".to_string(), 25),
        ]
    );
    namespaces.clear();
}

#[test]
fn print_timing_group_renders_totals_and_skips_empty_groups() {
    let timings = vec![("load".to_string(), 10), ("session".to_string(), 25)];
    print_timing_group("Startup Timings: main", &timings);
    // The empty group returns before printing anything.
    print_timing_group("Startup Timings: empty", &[]);
}

#[test]
fn the_env_gate_drives_the_flag_free_wrappers_in_a_dedicated_process() {
    // Edition 2024 makes `std::env::set_var` unsafe and this workspace
    // forbids `unsafe`, so the env-gated wrappers run in a child test
    // binary with PI_TIMING=1 instead of mutating this process.
    let exe = std::env::current_exe().expect("the test binary resolves");
    let mut handle = std::process::Command::new(exe)
        .arg("core::timings::tests::env_gated_wrappers_record_and_report")
        .arg("--exact")
        .arg("--nocapture")
        .env("PI_TIMING", "1")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("the timings probe spawns");
    let status = handle.wait().expect("the timings probe exits");
    assert!(status.success(), "the env-gated probe failed: {status:?}");
    reset_timings("probe-after"); // PI_TIMING unset here: no-op early return
    assert!(
        !TIMING_NAMESPACES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key("probe-after")
    );
}

#[test]
fn env_gated_wrappers_record_and_report() {
    reset_timings("probe");
    time("resourceLoader.reload", "probe");
    time("session", "probe");
    print_timings();
    {
        let namespaces = TIMING_NAMESPACES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if is_enabled() {
            assert_eq!(
                namespaces["probe"].timings.len(),
                2,
                "the env gate recorded both marks"
            );
        } else {
            // The direct (unflagged) run pins the disabled no-op.
            assert!(!namespaces.contains_key("probe"));
        }
        drop(namespaces);
    }
}
