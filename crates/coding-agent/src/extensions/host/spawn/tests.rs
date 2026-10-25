//! The spawn gate and loader cache: a fixture child over real pipes.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tempfile::TempDir;

use super::*;
use crate::extensions::host::{ExtensionIdentity, Frame, encode_frame};
use crate::extensions::runtime::ExtensionError;
use crate::package_manager::INSTALL_RECEIPT_FILE;

fn hello(protocol_version: u64) -> Vec<u8> {
    encode_frame(&Frame::Hello {
        protocol_version,
        extension: ExtensionIdentity {
            name: "demo".to_string(),
            version: "0.1.0".to_string(),
        },
    })
    .expect("encode hello")
}

fn registration(channel: &str) -> Vec<u8> {
    encode_frame(&Frame::RegisterEvent {
        channel: channel.to_string(),
    })
    .expect("encode registration")
}

fn ready() -> Vec<u8> {
    encode_frame(&Frame::Ready).expect("encode ready")
}

fn write_executable(root: &Path, relative: &str, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = root.join(relative);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    std::fs::write(&path, body).expect("write");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    path
}

/// One fixture child: a shell script that emits the handshake frames, then
/// consumes stdin until the host closes it.
struct FixtureChild {
    path: PathBuf,
    marker: PathBuf,
    counter: PathBuf,
}

fn local_child(root: &Path, name: &str) -> FixtureChild {
    let hello_path = root.join(format!("{name}-hello.bin"));
    let registration_path = root.join(format!("{name}-register.bin"));
    let ready_path = root.join(format!("{name}-ready.bin"));
    std::fs::write(&hello_path, hello(EXTENSION_PROTOCOL_VERSION)).expect("hello bytes");
    std::fs::write(&registration_path, registration("agent_start")).expect("registration bytes");
    std::fs::write(&ready_path, ready()).expect("ready bytes");
    let marker = root.join(format!("{name}-exited"));
    let counter = root.join(format!("{name}-starts"));
    let body = format!(
        "#!/bin/sh\nprintf x >> '{counter}'\ncat '{hello}' '{register}' '{ready}'\ncat > /dev/null\nprintf down > '{marker}'\n",
        counter = counter.display(),
        hello = hello_path.display(),
        register = registration_path.display(),
        ready = ready_path.display(),
        marker = marker.display(),
    );
    FixtureChild {
        path: write_executable(root, &format!("bin/{name}"), &body),
        marker,
        counter,
    }
}

fn starts(counter: &Path) -> usize {
    std::fs::read_to_string(counter).unwrap_or_default().len()
}

#[tokio::test]
async fn a_local_child_handshakes_registers_and_reuses_one_process() {
    let dir = TempDir::new().expect("temp dir");
    let child = local_child(dir.path(), "demo");
    let cwd = dir.path().to_string_lossy().into_owned();
    let path = child.path.to_string_lossy().into_owned();
    let loader = ExtensionLoader::new();

    let first = loader.load(std::slice::from_ref(&path), &cwd, None).await;
    assert!(first.errors.is_empty(), "{:?}", first.errors);
    let record = first.extensions[0].record.clone().expect("record");
    assert_eq!(record.resolved_path, path);
    assert!(
        record
            .handlers
            .lock()
            .expect("handlers")
            .contains_key("agent_start")
    );
    assert_eq!(starts(&child.counter), 1);

    let second = loader.load(std::slice::from_ref(&path), &cwd, None).await;
    let reused = second.extensions[0].record.clone().expect("record");
    assert!(Arc::ptr_eq(&record, &reused), "the cache reuses the child");
    assert_eq!(starts(&child.counter), 1);

    loader.invalidate().await;
    assert!(child.marker.exists(), "the child exited after stdin closed");
}

#[tokio::test]
async fn retain_shuts_down_the_paths_that_left_the_set() {
    let dir = TempDir::new().expect("temp dir");
    let kept = local_child(dir.path(), "kept");
    let dropped = local_child(dir.path(), "dropped");
    let cwd = dir.path().to_string_lossy().into_owned();
    let kept_path = kept.path.to_string_lossy().into_owned();
    let dropped_path = dropped.path.to_string_lossy().into_owned();
    let loader = ExtensionLoader::new();

    let paths = vec![kept_path.clone(), dropped_path];
    let outcome = loader.load(&paths, &cwd, None).await;
    assert_eq!(outcome.extensions.len(), 2);

    loader.retain(&[kept_path], &cwd).await;
    assert!(dropped.marker.exists(), "the left path shuts down");
    assert!(!kept.marker.exists(), "the kept path stays live");

    loader.invalidate().await;
    assert!(kept.marker.exists());
}

#[tokio::test]
async fn a_cwd_change_respawns_every_child() {
    let dir = TempDir::new().expect("temp dir");
    let child = local_child(dir.path(), "demo");
    let first_cwd = dir.path().to_string_lossy().into_owned();
    let second_cwd = dir.path().join("other").to_string_lossy().into_owned();
    std::fs::create_dir_all(&second_cwd).expect("other cwd");
    let path = child.path.to_string_lossy().into_owned();
    let loader = ExtensionLoader::new();

    let _ = loader
        .load(std::slice::from_ref(&path), &first_cwd, None)
        .await;
    assert_eq!(starts(&child.counter), 1);
    let second = loader
        .load(std::slice::from_ref(&path), &second_cwd, None)
        .await;
    assert!(second.errors.is_empty(), "{:?}", second.errors);
    assert_eq!(starts(&child.counter), 2, "a cwd change retires the cache");
}

#[tokio::test]
async fn a_handshake_version_drift_fails_closed() {
    let dir = TempDir::new().expect("temp dir");
    let hello_path = dir.path().join("hello.bin");
    std::fs::write(&hello_path, hello(EXTENSION_PROTOCOL_VERSION + 1)).expect("hello bytes");
    let body = format!(
        "#!/bin/sh\ncat '{}'\ncat > /dev/null\n",
        hello_path.display()
    );
    let script = write_executable(dir.path(), "bin/drifted", &body);
    let cwd = dir.path().to_string_lossy().into_owned();
    let loader = ExtensionLoader::new();

    let outcome = loader
        .load(&[script.to_string_lossy().into_owned()], &cwd, None)
        .await;
    assert!(outcome.extensions.is_empty());
    assert_eq!(outcome.errors.len(), 1);
    assert!(
        outcome.errors[0].error.contains("speaks wire version 2"),
        "{}",
        outcome.errors[0].error
    );
}

#[tokio::test]
async fn a_silent_child_hits_the_handshake_deadline() {
    let dir = TempDir::new().expect("temp dir");
    let script = write_executable(dir.path(), "bin/silent", "#!/bin/sh\nsleep 30\n");
    let cwd = dir.path().to_string_lossy().into_owned();
    let loader = ExtensionLoader {
        handshake_timeout: Duration::from_millis(50),
        shutdown_grace: Duration::from_millis(50),
        ..ExtensionLoader::new()
    };

    let outcome = loader
        .load(&[script.to_string_lossy().into_owned()], &cwd, None)
        .await;
    assert!(outcome.extensions.is_empty());
    assert!(
        outcome.errors[0].error.contains("handshake within 50ms"),
        "{}",
        outcome.errors[0].error
    );
}

#[tokio::test]
async fn a_child_that_ignores_stdin_end_is_killed_after_the_grace() {
    let dir = TempDir::new().expect("temp dir");
    let hello_path = dir.path().join("hello.bin");
    let ready_path = dir.path().join("ready.bin");
    std::fs::write(&hello_path, hello(EXTENSION_PROTOCOL_VERSION)).expect("hello bytes");
    std::fs::write(&ready_path, ready()).expect("ready bytes");
    let body = format!(
        "#!/bin/sh\ncat '{}' '{}'\nexec sleep 30\n",
        hello_path.display(),
        ready_path.display()
    );
    let script = write_executable(dir.path(), "bin/stubborn", &body);
    let cwd = dir.path().to_string_lossy().into_owned();
    let loader = ExtensionLoader {
        shutdown_grace: Duration::from_millis(50),
        ..ExtensionLoader::new()
    };

    let outcome = loader
        .load(&[script.to_string_lossy().into_owned()], &cwd, None)
        .await;
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);

    let started = std::time::Instant::now();
    loader.invalidate().await;
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "the grace expires and the child is killed"
    );
}

#[tokio::test]
async fn a_missing_binary_reports_a_load_error() {
    let dir = TempDir::new().expect("temp dir");
    let missing = dir.path().join("bin/missing");
    let cwd = dir.path().to_string_lossy().into_owned();
    let loader = ExtensionLoader::new();

    let outcome = loader
        .load(&[missing.to_string_lossy().into_owned()], &cwd, None)
        .await;
    assert!(outcome.extensions.is_empty());
    assert!(
        outcome.errors[0]
            .error
            .starts_with("failed to spawn extension"),
        "{}",
        outcome.errors[0].error
    );
}

#[tokio::test]
async fn the_receipt_gate_fails_closed() {
    let dir = TempDir::new().expect("temp dir");
    let root = dir.path();
    let binary = write_executable(root, "bin/demo", "#!/bin/sh\nexit 0\n");

    // No receipt: an explicit local path loads in place.
    verify_install_receipt(&binary).expect("a local path bypasses the receipt");

    std::fs::write(root.join(INSTALL_RECEIPT_FILE), "{not json").expect("malformed receipt");
    assert!(
        verify_install_receipt(&binary)
            .expect_err("malformed JSON fails")
            .message()
            .contains("is not JSON")
    );
    std::fs::write(
        root.join(INSTALL_RECEIPT_FILE),
        r#"{"kind":"other","schemaVersion":1}"#,
    )
    .expect("foreign receipt");
    assert!(
        verify_install_receipt(&binary)
            .expect_err("a foreign receipt fails")
            .message()
            .contains("is invalid")
    );

    write_receipt(root, Some(EXTENSION_PROTOCOL_VERSION));
    verify_install_receipt(&binary).expect("a matching receipt passes");

    write_receipt(root, Some(EXTENSION_PROTOCOL_VERSION + 1));
    assert!(
        verify_install_receipt(&binary)
            .expect_err("version drift fails")
            .message()
            .contains("extension protocol version 2")
    );

    write_receipt(root, None);
    assert!(
        verify_install_receipt(&binary)
            .expect_err("a receipt without a version fails")
            .message()
            .contains("records no extension protocol version")
    );

    write_receipt(root, Some(EXTENSION_PROTOCOL_VERSION));
    std::fs::write(&binary, "#!/bin/sh\nprintf tampered\n").expect("tamper");
    assert!(
        verify_install_receipt(&binary)
            .expect_err("changed bytes fail")
            .message()
            .contains("changed since install: bin/demo")
    );

    write_receipt(root, Some(EXTENSION_PROTOCOL_VERSION));
    std::fs::write(root.join("extra"), "surprise").expect("extra file");
    assert!(
        verify_install_receipt(&binary)
            .expect_err("an unlisted file fails")
            .message()
            .contains("does not list: extra")
    );

    std::fs::remove_file(root.join("extra")).expect("remove extra");
    std::fs::remove_file(&binary).expect("remove binary");
    assert!(
        verify_install_receipt(&binary)
            .expect_err("a missing file fails")
            .message()
            .contains("missing: bin/demo")
    );
}

fn write_receipt(root: &Path, protocol_version: Option<u64>) {
    let receipt = InstallReceipt {
        channel: "crate".to_string(),
        source: "crate:demo".to_string(),
        resolved_version: None,
        resolved_ref: None,
        protocol_version,
        files: hash_package_files(root),
    };
    std::fs::write(
        root.join(INSTALL_RECEIPT_FILE),
        serde_json::to_string(&receipt.to_value()).expect("serialize receipt"),
    )
    .expect("write receipt");
}

#[test]
fn the_relay_reports_to_the_latest_bound_listener() {
    let loader = ExtensionLoader::new();
    let seen = Arc::new(Mutex::new(0));
    let counter = Arc::clone(&seen);
    *lock(&loader.listener) = Some(Arc::new(move |_error: &ExtensionError| {
        *counter.lock().expect("counter") += 1;
    }));
    let relay = loader.relay();
    let error = ExtensionError {
        extension_path: "/ext/demo".to_string(),
        event: "agent_start".to_string(),
        error: "boom".to_string(),
        stack: None,
    };
    relay(&error);
    assert_eq!(*seen.lock().expect("seen"), 1);

    *lock(&loader.listener) = None;
    relay(&error);
    assert_eq!(*seen.lock().expect("seen"), 1);
}
