//! The generated model-data validation suite, ported from
//! `packages/ai/test/model-data-validation.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The fixture writes the same directory shapes upstream builds: provider
//! shards, an aggregator, and a manifest. The Rust port additionally gates
//! the shards actually committed to the crate.
//!
//! Porting restatement: upstream's `readModelDataStructure` scrapes
//! TypeScript import lines from `models.generated.ts`; the Rust aggregator is
//! the embedded registry, so the fixture's "missing provider shard" case
//! asserts on the directory/structure mismatch path instead.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use pi_ai::model_data::{
    MODEL_DATA_MANIFEST_FILE, MODEL_DATA_SCHEMA_VERSION, assert_exact_model_ids,
    create_model_data_manifest, embedded_data_dir, read_model_data_structure,
    validate_model_data_directory,
};

const GENERATED_AT: &str = "2026-07-23T10:00:00.000Z";

struct Fixture {
    data_dir: std::path::PathBuf,
    package_root: std::path::PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.package_root);
    }
}

/// Build the same fixture tree upstream's `createFixture` writes: one
/// provider shard with a single model under `openai-completions`, plus the
/// manifest.
fn create_fixture() -> Fixture {
    // The atomic suffix keeps parallel fixture builds from colliding on one
    // temp root, which would make one test's Drop delete another's tree.
    static FIXTURE_SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = FIXTURE_SEQ.fetch_add(1, Ordering::SeqCst);
    let package_root =
        std::env::temp_dir().join(format!("pi-model-data-{}-{seq}", std::process::id()));
    let providers_dir = package_root.join("src").join("providers");
    let data_dir = providers_dir.join("data");
    std::fs::create_dir_all(&data_dir).expect("mkdir");

    let structure: pi_ai::model_data::ModelDataStructure = BTreeMap::from([(
        "test-provider".to_owned(),
        BTreeMap::from([("model-a".to_owned(), "openai-completions".to_owned())]),
    )]);
    let values = serde_json::json!({
        "model-a": {
            "id": "model-a",
            "name": "Model A",
            "api": "openai-completions",
            "provider": "test-provider",
            "baseUrl": "https://example.test/v1",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 1000,
            "maxTokens": 100,
        },
    });
    write_fixture_data(
        &data_dir,
        &structure,
        &values,
        MODEL_DATA_SCHEMA_VERSION,
        "openai-completions",
    );
    Fixture {
        data_dir,
        package_root,
    }
}

/// Write the shard and its manifest, upstream's `writeFixtureData`.
fn write_fixture_data(
    data_dir: &Path,
    structure: &pi_ai::model_data::ModelDataStructure,
    values: &serde_json::Value,
    manifest_schema_version: u64,
    api_group: &str,
) {
    let filename = "test-provider.json";
    let content = format!("{}\n", serde_json::json!({ api_group: values }));
    std::fs::write(data_dir.join(filename), &content).expect("write shard");
    let mut manifest = create_model_data_manifest(
        structure,
        &BTreeMap::from([(filename.to_owned(), content)]),
        GENERATED_AT,
    );
    manifest.schema_version = manifest_schema_version;
    std::fs::write(
        data_dir.join(MODEL_DATA_MANIFEST_FILE),
        format!(
            "{}\n",
            serde_json::to_string(&manifest).expect("manifest json")
        ),
    )
    .expect("write manifest");
}

#[test]
fn rejects_a_missing_upstream_model_from_an_exact_generated_allowlist() {
    let error = assert_exact_model_ids(
        "qwen-token-plan-individual",
        ["model-a".to_owned(), "model-b".to_owned()],
        ["model-a".to_owned()],
    )
    .expect_err("allowlist mismatch fails");
    assert_eq!(
        error.to_string(),
        "qwen-token-plan-individual model IDs do not match (missing: model-b)"
    );
}

#[test]
fn rejects_an_unexpected_model_from_an_exact_generated_allowlist() {
    let error = assert_exact_model_ids(
        "test-provider",
        ["model-a".to_owned()],
        ["model-a".to_owned(), "model-b".to_owned()],
    )
    .expect_err("allowlist mismatch fails");
    assert_eq!(
        error.to_string(),
        "test-provider model IDs do not match (extra: model-b)"
    );
}

#[test]
fn reads_and_validates_api_grouped_model_data() {
    let fixture = create_fixture();
    let structure = read_model_data_structure(&fixture.package_root).expect("structure reads");
    assert_eq!(
        structure,
        BTreeMap::from([(
            "test-provider".to_owned(),
            BTreeMap::from([("model-a".to_owned(), "openai-completions".to_owned())])
        )])
    );
    validate_model_data_directory(&structure, &fixture.data_dir).expect("valid fixture passes");
}

#[test]
fn rejects_a_missing_model_data_directory() {
    let fixture = create_fixture();
    let structure = read_model_data_structure(&fixture.package_root).expect("structure reads");
    std::fs::remove_dir_all(&fixture.data_dir).expect("rm");
    let error = validate_model_data_directory(&structure, &fixture.data_dir)
        .expect_err("missing directory fails");
    assert!(error.to_string().contains("does not exist"), "got: {error}");
}

#[test]
fn rejects_a_wrong_model_id() {
    rejects_wrong_field("id", "wrong-id", "has id");
}

#[test]
fn rejects_a_wrong_model_provider() {
    rejects_wrong_field("provider", "wrong-provider", "has provider");
}

#[test]
fn rejects_a_wrong_model_api() {
    rejects_wrong_field("api", "anthropic-messages", "has api");
}

fn rejects_wrong_field(field: &str, value: &str, expected_message: &str) {
    let fixture = create_fixture();
    let mut model = read_fixture_values(&fixture);
    model[field] = serde_json::json!(value);
    let structure = read_model_data_structure(&fixture.package_root).expect("structure");
    write_fixture_data(
        &fixture.data_dir,
        &structure,
        &serde_json::json!({ "model-a": model }),
        MODEL_DATA_SCHEMA_VERSION,
        "openai-completions",
    );
    let structure = read_model_data_structure(&fixture.package_root).expect("structure");
    let error = validate_model_data_directory(&structure, &fixture.data_dir)
        .expect_err("wrong field fails");
    assert!(
        error.to_string().contains(expected_message),
        "expected {expected_message:?} in: {error}"
    );
}

fn read_fixture_values(fixture: &Fixture) -> serde_json::Value {
    let shard: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture.data_dir.join("test-provider.json")).expect("read"),
    )
    .expect("parse");
    // Upstream's `fixture.values["model-a"]`: the model entry inside the API
    // group.
    shard["openai-completions"]["model-a"].clone()
}

#[test]
fn rejects_a_model_in_the_wrong_api_group() {
    let fixture = create_fixture();
    let structure = read_model_data_structure(&fixture.package_root).expect("structure");
    let model = read_fixture_values(&fixture);
    write_fixture_data(
        &fixture.data_dir,
        &structure,
        &serde_json::json!({ "model-a": model }),
        MODEL_DATA_SCHEMA_VERSION,
        "anthropic-messages",
    );
    let error = validate_model_data_directory(&structure, &fixture.data_dir)
        .expect_err("wrong group fails");
    assert!(
        error.to_string().contains("grouped under API"),
        "got: {error}"
    );
}

#[test]
fn rejects_duplicate_model_ids_across_api_groups() {
    let fixture = create_fixture();
    let structure = read_model_data_structure(&fixture.package_root).expect("structure");
    let values = read_fixture_values(&fixture);
    let content = format!(
        "{}\n",
        serde_json::json!({
            "openai-completions": { "model-a": values },
            "anthropic-messages": { "model-a": values },
        })
    );
    std::fs::write(fixture.data_dir.join("test-provider.json"), &content).expect("write shard");
    let manifest = create_model_data_manifest(
        &structure,
        &BTreeMap::from([("test-provider.json".to_owned(), content)]),
        GENERATED_AT,
    );
    std::fs::write(
        fixture.data_dir.join(MODEL_DATA_MANIFEST_FILE),
        format!(
            "{}\n",
            serde_json::to_string(&manifest).expect("manifest json")
        ),
    )
    .expect("write manifest");
    let error =
        validate_model_data_directory(&structure, &fixture.data_dir).expect_err("duplicate fails");
    assert!(
        error.to_string().contains("more than one API group"),
        "got: {error}"
    );
}

#[test]
fn rejects_missing_model_ids_and_stale_file_hashes() {
    let fixture = create_fixture();
    // The structure snapshot precedes the overwrite, upstream's
    // `fixture.structure`.
    let structure = read_model_data_structure(&fixture.package_root).expect("structure");
    std::fs::write(fixture.data_dir.join("test-provider.json"), "{}\n").expect("write shard");
    let error = validate_model_data_directory(&structure, &fixture.data_dir)
        .expect_err("empty shard fails");
    let message = error.to_string();
    assert!(
        message.contains("manifest hash") || message.contains("model IDs"),
        "got: {message}"
    );
}

#[test]
fn rejects_incompatible_schema_and_generation_stamps() {
    let fixture = create_fixture();
    let structure = read_model_data_structure(&fixture.package_root).expect("structure");
    let values = read_fixture_values(&fixture);
    write_fixture_data(
        &fixture.data_dir,
        &structure,
        &values,
        MODEL_DATA_SCHEMA_VERSION + 1,
        "openai-completions",
    );
    let error = validate_model_data_directory(&structure, &fixture.data_dir)
        .expect_err("wrong schema fails");
    assert!(
        error.to_string().contains("model data schema"),
        "got: {error}"
    );

    let manifest_path = fixture.data_dir.join(MODEL_DATA_MANIFEST_FILE);
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("read manifest"))
            .expect("manifest");
    manifest["structureHash"] = serde_json::json!("stale");
    std::fs::write(
        &manifest_path,
        format!(
            "{}\n",
            serde_json::to_string(&manifest).expect("manifest json")
        ),
    )
    .expect("write manifest");
    let error = validate_model_data_directory(&structure, &fixture.data_dir)
        .expect_err("stale stamp fails");
    assert!(
        error.to_string().contains("generation stamp"),
        "got: {error}"
    );
}

#[test]
fn rejects_an_invalid_generation_timestamp() {
    let fixture = create_fixture();
    let manifest_path = fixture.data_dir.join(MODEL_DATA_MANIFEST_FILE);
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("read manifest"))
            .expect("manifest");
    manifest["generatedAt"] = serde_json::json!("invalid");
    std::fs::write(
        &manifest_path,
        serde_json::to_string(&manifest).expect("json"),
    )
    .expect("write manifest");
    let structure = read_model_data_structure(&fixture.package_root).expect("structure");
    let error = validate_model_data_directory(&structure, &fixture.data_dir)
        .expect_err("invalid timestamp fails");
    assert!(
        error.to_string().contains("generation timestamp"),
        "got: {error}"
    );
}

#[test]
fn gates_the_committed_shards_against_their_manifest() {
    // The committed data dir is the crate's own; the aggregator the upstream
    // fixture mimics is the embedded registry.
    let data_dir = embedded_data_dir();
    let structure =
        read_model_data_structure(Path::new(env!("CARGO_MANIFEST_DIR"))).expect("structure");
    validate_model_data_directory(&structure, &data_dir).expect("committed shards validate");
}

#[test]
fn rejects_a_missing_provider_shard_against_the_aggregator() {
    // The upstream fixture removes a shard and expects the aggregator
    // cross-check to fail; the Rust aggregator is the shard directory, so a
    // removed shard changes the structure itself and the manifest stamp no
    // longer matches.
    let fixture = create_fixture();
    let structure = read_model_data_structure(&fixture.package_root).expect("structure");
    std::fs::remove_file(fixture.data_dir.join("test-provider.json")).expect("remove shard");
    let error = validate_model_data_directory(&structure, &fixture.data_dir)
        .expect_err("missing shard fails");
    assert!(
        error
            .to_string()
            .contains("provider data files do not match"),
        "got: {error}"
    );
}
