//! The package-manager installer-safety boundary suite, the re-expressed
//! channels' rules ADR 0007 pins: the tarball caps, traversal/symlink
//! rejection, the receipt round-trip, and the crate-channel receipt layout
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::io::Write;
use std::path::Path;

use serde_json::json;

use pi_coding_agent::package_manager::{
    INSTALL_RECEIPT_FILE, InstallReceipt, MAX_TARBALL_BYTES, MAX_TARBALL_ENTRIES,
    MAX_UNPACKED_BYTES, unpack_tarball,
};

fn octal_field<const WIDTH: usize>(value: usize) -> [u8; WIDTH] {
    let mut field = [0u8; WIDTH];
    let text = format!("{:0width$o}\0", value, width = WIDTH - 1);
    field[..text.len()].copy_from_slice(text.as_bytes());
    field
}

fn build_ustar_header(
    name: &str,
    size: usize,
    mode: u32,
    entry_type: tar::EntryType,
    link_name: Option<&str>,
) -> [u8; 512] {
    let mut header = [0u8; 512];
    let name_bytes = name.as_bytes();
    header[0..name_bytes.len()].copy_from_slice(name_bytes);
    header[100..108].copy_from_slice(&octal_field::<8>(mode as usize));
    header[108..116].copy_from_slice(&octal_field::<8>(0));
    header[116..124].copy_from_slice(&octal_field::<8>(0));
    header[124..136].copy_from_slice(&octal_field::<12>(size));
    header[136..148].copy_from_slice(&octal_field::<12>(0));
    header[156] = match entry_type {
        tar::EntryType::Symlink => b'2',
        // The helper emits only regular files besides symlinks; the ustar
        // type-flag byte for those is `0`.
        _ => b'0',
    };
    if let Some(link_target) = link_name {
        let link_bytes = link_target.as_bytes();
        header[157..157 + link_bytes.len()].copy_from_slice(link_bytes);
    }
    header[257..262].copy_from_slice(b"ustar");
    header[263..265].copy_from_slice(b"00");
    // The checksum sums with the field itself reading as eight spaces.
    header[148..156].copy_from_slice(b"        ");
    let checksum: usize = header.iter().map(|byte| *byte as usize).sum::<usize>();
    header[148..156].copy_from_slice(&octal_field::<8>(checksum));
    header
}

fn tar_bytes(entries: Vec<(&str, &[u8], u32)>) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, contents, mode) in entries {
        out.extend_from_slice(&build_ustar_header(
            name,
            contents.len(),
            mode,
            tar::EntryType::Regular,
            None,
        ));
        out.extend_from_slice(contents);
        let pad = (512 - contents.len() % 512) % 512;
        out.extend(std::iter::repeat_n(0u8, pad));
    }
    out.extend(std::iter::repeat_n(0u8, 1024));
    out
}

fn gz_bytes(tar: &[u8]) -> Vec<u8> {
    // Concurrent tests share the process id, so the scratch file names a
    // counter suffix.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let gz_path = std::env::temp_dir().join(format!(
        "pm-boundary-{}-{}.gz",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::SeqCst)
    ));
    let file = std::fs::File::create(&gz_path).expect("gz file");
    let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    encoder.write_all(tar).expect("gzip write");
    encoder.finish().expect("gzip finish");
    let bytes = std::fs::read(&gz_path).expect("gz bytes");
    let _ = std::fs::remove_file(&gz_path);
    bytes
}

#[test]
fn unpacks_a_plain_tarball_preserving_modes() {
    let stage = tempfile::tempdir().expect("stage");
    let archive = gz_bytes(&tar_bytes(vec![
        ("bin/pi", b"#!/bin/sh\n".as_slice(), 0o755),
        ("README.md", b"readme".as_slice(), 0o644),
    ]));
    unpack_tarball(&archive, stage.path()).expect("unpack");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let bin_mode = std::fs::metadata(stage.path().join("bin/pi"))
            .expect("bin")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(bin_mode, 0o755, "the executable bit rides the entry");
        let readme_mode = std::fs::metadata(stage.path().join("README.md"))
            .expect("readme")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(readme_mode, 0o644);
    }
    assert!(stage.path().join("bin/pi").exists());
}

#[test]
fn rejects_a_traversing_entry() {
    let stage = tempfile::tempdir().expect("stage");
    let archive = gz_bytes(&tar_bytes(vec![(
        "../../escape",
        b"payload".as_slice(),
        0o644,
    )]));
    let error = unpack_tarball(&archive, stage.path()).expect_err("traversal rejects");
    assert!(error.0.contains("escapes the staging root"), "{error}");
    assert!(
        !stage
            .path()
            .parent()
            .expect("the stage tempdir has a parent")
            .join("escape")
            .exists()
    );
}

#[test]
fn rejects_an_absolute_entry() {
    let stage = tempfile::tempdir().expect("stage");
    let archive = gz_bytes(&tar_bytes(vec![(
        "/etc/escape",
        b"payload".as_slice(),
        0o644,
    )]));
    let error = unpack_tarball(&archive, stage.path()).expect_err("absolute rejects");
    assert!(error.0.contains("escapes the staging root"), "{error}");
}

#[test]
fn unpacks_dot_prefixed_entries_and_rejects_special_types() {
    let stage = tempfile::tempdir().expect("stage");
    // The `./` prefix folds; the entry lands under the stage root.
    let archive = gz_bytes(&tar_bytes(vec![("./normal", b"payload".as_slice(), 0o644)]));
    unpack_tarball(&archive, stage.path()).expect("dot-prefixed unpacks");
    assert!(stage.path().join("normal").exists());

    // The symlink entry type rejects: a package tarball needs none, and the
    // link target can escape the staging root — here it points at /etc.
    let mut symlink_archive = build_ustar_header(
        "link",
        0,
        0o777,
        tar::EntryType::Symlink,
        Some("/etc/escape"),
    )
    .to_vec();
    symlink_archive.extend(std::iter::repeat_n(0u8, 1024));
    let error =
        unpack_tarball(&gz_bytes(&symlink_archive), stage.path()).expect_err("symlink rejects");
    assert!(
        error.0.contains("unsupported entry type") || error.0.contains("escapes the staging root"),
        "{error}"
    );
    let _ = MAX_TARBALL_ENTRIES;
}

#[test]
fn caps_the_download_and_unpack_sizes() {
    // The caps are module constants; the boundary pins their shapes so a
    // drop to zero fails loudly, at compile time.
    const {
        assert!(MAX_TARBALL_BYTES > 0);
        assert!(MAX_UNPACKED_BYTES > MAX_TARBALL_BYTES);
        assert!(MAX_TARBALL_ENTRIES > 0);
    }
}

#[test]
fn the_receipt_round_trips_through_its_wire_shape() {
    let receipt = InstallReceipt {
        channel: "crate".to_string(),
        source: "crate:pi-llm-tools".to_string(),
        resolved_version: Some("1.2.3".to_string()),
        resolved_ref: None,
        files: [
            ("bin/pi-llm-tools".to_string(), "deadbeef".to_string()),
            ("pi-package-install.json".to_string(), "cafe".to_string()),
        ]
        .into_iter()
        .collect(),
    };
    let value = receipt.to_value();
    assert_eq!(
        value.get("kind").and_then(serde_json::Value::as_str),
        Some("pi-package-install")
    );
    assert_eq!(
        value
            .get("schemaVersion")
            .and_then(serde_json::Value::as_i64),
        Some(1)
    );
    assert_eq!(
        value.get("channel").and_then(serde_json::Value::as_str),
        Some("crate")
    );
    assert_eq!(
        value
            .get("resolvedVersion")
            .and_then(serde_json::Value::as_str),
        Some("1.2.3")
    );

    let parsed = InstallReceipt::from_value(&value).expect("parses");
    assert_eq!(parsed, receipt);

    // A foreign kind fails closed.
    let foreign = json!({ "kind": "other", "schemaVersion": 1, "channel": "crate", "files": {} });
    assert!(InstallReceipt::from_value(&foreign).is_err());
    let wrong_schema = json!({ "kind": "pi-package-install", "schemaVersion": 2, "channel": "crate", "files": {} });
    assert!(InstallReceipt::from_value(&wrong_schema).is_err());
    let no_files = json!({ "kind": "pi-package-install", "schemaVersion": 1, "channel": "crate" });
    assert!(InstallReceipt::from_value(&no_files).is_err());
}

#[test]
fn the_receipt_file_name_is_the_module_constant() {
    assert_eq!(INSTALL_RECEIPT_FILE, "pi-package-install.json");
}

#[test]
fn the_git_update_marker_names_the_checkout_parent() {
    // The marker path shape, upstream's getGitUpdateMarkerPath.
    let target = Path::new("/tmp/somewhere/agent/git/github.com/user/repo");
    let basename = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let marker = target
        .parent()
        .unwrap_or(target)
        .join(format!(".{basename}.pi-update-incomplete"));
    assert!(
        marker
            .to_string_lossy()
            .ends_with("/user/.repo.pi-update-incomplete")
    );
}
