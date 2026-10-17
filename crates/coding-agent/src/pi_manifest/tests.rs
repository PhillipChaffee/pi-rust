#![expect(
    clippy::expect_used,
    reason = "the unit tests pin manifest reads; an unexpected result panics the test by design"
)]

use super::*;

#[test]
fn reads_manifest_fields() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("package.json");
    std::fs::write(
        &path,
        r#"{"name":"pkg","pi":{"extensions":["./src/index.ts"],"skills":["./skills"],"prompts":[1,2]}}"#,
    )
    .expect("write");
    let manifest = read_pi_manifest(&path.to_string_lossy()).expect("manifest");
    assert_eq!(
        manifest.extensions,
        Some(vec!["./src/index.ts".to_string()])
    );
    assert_eq!(manifest.skills, Some(vec!["./skills".to_string()]));
    // Non-string entries invalidate the whole field.
    assert_eq!(manifest.prompts, None);
    assert_eq!(manifest.themes, None);
}

#[test]
fn returns_none_without_a_pi_object() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("package.json");
    std::fs::write(&path, r#"{"name":"pkg"}"#).expect("write");
    assert_eq!(read_pi_manifest(&path.to_string_lossy()), None);
}

#[test]
fn returns_none_on_invalid_json_or_missing_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("package.json");
    std::fs::write(&path, "{invalid").expect("write");
    assert_eq!(read_pi_manifest(&path.to_string_lossy()), None);
    assert_eq!(
        read_pi_manifest(&dir.path().join("absent.json").to_string_lossy()),
        None
    );
}

#[test]
fn reads_a_parsed_value_and_keeps_string_arrays() {
    let parsed: serde_json::Value =
        serde_json::from_str(r#"{"pi":{"themes":["a.json"],"skills":"not-an-array"}}"#)
            .expect("parses");
    let manifest = read_pi_manifest_value(&parsed).expect("manifest");
    assert_eq!(manifest.themes, Some(vec!["a.json".to_string()]));
    assert_eq!(manifest.skills, None);
    // A non-object `pi` and a non-object root both answer None.
    let flat: serde_json::Value = serde_json::from_str(r#"{"pi":[1]}"#).expect("parses");
    assert_eq!(read_pi_manifest_value(&flat), None);
    let scalar: serde_json::Value = serde_json::from_str("3").expect("parses");
    assert_eq!(read_pi_manifest_value(&scalar), None);
}
