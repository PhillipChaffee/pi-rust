//! Upstream `test/resolve-config-value.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated.
//!
//! Upstream's `process.env` writes ride the crate's env seam: the values
//! upstream plants in `process.env` ride the injected process lookup of the
//! `_with` variants (the `env` argument stays `None` — upstream passes no
//! credential map in these cases), and the credential-scoped case passes its
//! map through the `env` parameter. The plain functions run the real
//! environment, so command cases call them directly: the command branch never
//! reads the seam.
//!
//! Cache hygiene diverges deliberately: upstream clears the process-wide
//! command cache in `beforeEach`/`afterEach`, but these tests run on parallel
//! threads sharing that cache, and a concurrent clear would race the cache
//! test's execution-count assertions. Command keys embed per-test temp paths,
//! so cross-test collisions cannot happen, and only the cache test clears.
//!
//! Dropped: `uses stdin when the configured Windows shell requires it` —
//! win32-gated upstream, and the win32 configured-shell branch rides the
//! map's Windows exclusion (see the module doc of `resolve_config_value`).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use pi_coding_agent::resolve_config_value::{
    EnvLookup, clear_config_value_cache, resolve_config_value, resolve_config_value_uncached,
    resolve_config_value_with,
};

/// A scratch directory with upstream's `pi-config-value-` prefix, unique and
/// removed at scope end.
fn temp_dir(prefix: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(prefix)
        .tempdir()
        .expect("temp dir")
}

/// Upstream's counter-command escaping: backslashes become slashes and
/// embedded quotes are escaped, keeping the command text shell-safe.
fn escaped(temp_path: &std::path::Path) -> String {
    temp_path
        .to_string_lossy()
        .replace('\\', "/")
        .replace('"', "\\\"")
}

#[test]
fn resolves_literals_environment_templates_and_escapes() {
    // upstream sets process.env.TEST_CONFIG_LEFT/RIGHT and resolves without
    // a credential map, so the values ride the injected process seam
    let process_env: EnvLookup = Box::new(|name| match name {
        "TEST_CONFIG_LEFT" => Some("left".to_string()),
        "TEST_CONFIG_RIGHT" => Some("right".to_string()),
        _ => None,
    });

    assert_eq!(
        resolve_config_value_with("literal-key", None, &process_env),
        Some("literal-key".to_string())
    );
    assert_eq!(
        resolve_config_value_with("$TEST_CONFIG_LEFT", None, &process_env),
        Some("left".to_string())
    );
    assert_eq!(
        resolve_config_value_with("${TEST_CONFIG_LEFT}_$TEST_CONFIG_RIGHT", None, &process_env),
        Some("left_right".to_string())
    );
    assert_eq!(
        resolve_config_value_with("$$TEST_CONFIG_LEFT", None, &process_env),
        Some("$TEST_CONFIG_LEFT".to_string())
    );
    assert_eq!(
        resolve_config_value_with("$!literal-$TEST_CONFIG_RIGHT", None, &process_env),
        Some("!literal-right".to_string())
    );
}

#[test]
fn uses_credential_scoped_environment_before_process_env() {
    // upstream sets process.env.TEST_CONFIG_SCOPED to "process" and passes
    // the credential map carrying "credential": the map wins
    let mut credential = BTreeMap::new();
    credential.insert("TEST_CONFIG_SCOPED".to_string(), "credential".to_string());
    let process_env: EnvLookup = Box::new(|name| match name {
        "TEST_CONFIG_SCOPED" => Some("process".to_string()),
        _ => None,
    });

    assert_eq!(
        resolve_config_value_with("$TEST_CONFIG_SCOPED", Some(&credential), &process_env),
        Some("credential".to_string())
    );
}

#[test]
fn executes_shell_commands_and_trims_their_output() {
    assert_eq!(
        resolve_config_value("!echo '  spaced-key  '", None),
        Some("spaced-key".to_string())
    );
    assert_eq!(
        resolve_config_value("!printf 'line1\\nline2'", None),
        Some("line1\nline2".to_string())
    );
    assert_eq!(
        resolve_config_value("!echo 'hello world' | tr ' ' '-'", None),
        Some("hello-world".to_string())
    );
}

#[test]
fn returns_undefined_when_command_resolution_fails() {
    // upstream's test.each rows, each resolving to nothing
    for command in ["!exit 1", "!nonexistent-command-12345", "!printf ''"] {
        assert_eq!(resolve_config_value(command, None), None, "row: {command}");
    }
}

#[test]
fn caches_successful_and_failed_commands_until_explicitly_cleared() {
    clear_config_value_cache();
    let temp = temp_dir("pi-config-value-cache-");
    let counter_path = temp.path().join("counter");
    std::fs::write(&counter_path, "0").expect("counter seed");
    let path = escaped(&counter_path);
    let success =
        format!("!sh -c 'count=$(cat \"{path}\"); echo $((count + 1)) > \"{path}\"; echo value'");

    assert_eq!(
        resolve_config_value(&success, None),
        Some("value".to_string())
    );
    assert_eq!(
        resolve_config_value(&success, None),
        Some("value".to_string())
    );
    assert_eq!(
        std::fs::read_to_string(&counter_path)
            .expect("counter read")
            .trim(),
        "1"
    );

    clear_config_value_cache();
    assert_eq!(
        resolve_config_value(&success, None),
        Some("value".to_string())
    );
    assert_eq!(
        std::fs::read_to_string(&counter_path)
            .expect("counter read")
            .trim(),
        "2"
    );

    let failure =
        format!("!sh -c 'count=$(cat \"{path}\"); echo $((count + 1)) > \"{path}\"; exit 1'");
    assert_eq!(resolve_config_value(&failure, None), None);
    assert_eq!(resolve_config_value(&failure, None), None);
    assert_eq!(
        std::fs::read_to_string(&counter_path)
            .expect("counter read")
            .trim(),
        "3"
    );

    // upstream's afterEach hygiene, last so no other test's state is raced
    clear_config_value_cache();
}

#[test]
fn does_not_cache_environment_values() {
    // the seam is a shared mutable map: the second resolve must observe the
    // update, proving env templates never enter the command cache
    let shared: Arc<Mutex<BTreeMap<String, String>>> = Arc::new(Mutex::new(BTreeMap::new()));
    shared
        .lock()
        .expect("env map lock")
        .insert("TEST_CONFIG_DYNAMIC".to_string(), "first".to_string());
    let process_env: EnvLookup = {
        let shared = Arc::clone(&shared);
        Box::new(move |name| shared.lock().expect("env map lock").get(name).cloned())
    };

    assert_eq!(
        resolve_config_value_with("$TEST_CONFIG_DYNAMIC", None, &process_env),
        Some("first".to_string())
    );
    shared
        .lock()
        .expect("env map lock")
        .insert("TEST_CONFIG_DYNAMIC".to_string(), "second".to_string());
    assert_eq!(
        resolve_config_value_with("$TEST_CONFIG_DYNAMIC", None, &process_env),
        Some("second".to_string())
    );
}

#[test]
fn uncached_resolution_executes_a_command_on_every_call() {
    let temp = temp_dir("pi-config-value-uncached-");
    let counter_path = temp.path().join("uncached-counter");
    std::fs::write(&counter_path, "0").expect("counter seed");
    let path = escaped(&counter_path);
    let command =
        format!("!sh -c 'count=$(cat \"{path}\"); echo $((count + 1)) > \"{path}\"; echo value'");

    assert_eq!(
        resolve_config_value_uncached(&command, None),
        Some("value".to_string())
    );
    assert_eq!(
        resolve_config_value_uncached(&command, None),
        Some("value".to_string())
    );
    assert_eq!(
        std::fs::read_to_string(&counter_path)
            .expect("counter read")
            .trim(),
        "2"
    );
}
