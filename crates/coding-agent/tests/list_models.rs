//! The `--list-models` table, upstream's `src/cli/list-models.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream ships no test file for `list-models.ts`, so the suite pins the
//! output contract the port keeps: the token-count scales (`200000` →
//! `"200K"`, `1500000` → `"1.5M"`), the `yes`/`no` flag cells, the
//! provider-then-id sort, the six-column space-padded grid, and the writer
//! split (table to stdout, load warning to stderr). The `localeCompare`
//! sort restates to byte order, which the ASCII provider and model ids the
//! fixtures use agree with.
//!
//! Porting restatements this suite records:
//!
//! - The `console.log`/`console.error` split restates to the [`std::io::Write`]
//!   writers `list_models_into` takes; upstream's stdout contract is
//!   asserted on captured buffers.
//! - The ambient environment can configure built-in providers (an
//!   `OPENAI_API_KEY` present in the environment configures `openai`), so
//!   the runtime-driven cases assert on the registered fixtures' rows, and
//!   the exact-grid contract lives on [`render_models_table`], which no
//!   environment can perturb.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

#[expect(
    dead_code,
    reason = "the fixture module compiles whole into every test binary; this suite drives only its model helpers"
)]
mod common;

use std::sync::Arc;

use common::model_layer::{empty_auth_storage, model, zero_cost};
use pi_ai::types::{Api, Modality};
use pi_coding_agent::cli::list_models::{
    format_token_count, list_models, list_models_into, render_models_table,
};
use pi_coding_agent::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use pi_coding_agent::models_store::InMemoryCodingAgentModelsStore;
use pi_coding_agent::provider_composer::{ProviderConfigInput, ProviderModelInput};

/// One table-provider model fixture: the shape variety upstream's table
/// renders — the million and thousand scales, the sub-thousand whole
/// count, both flag cells.
fn table_model(
    id: &str,
    context: u64,
    max: u64,
    reasoning: bool,
    image: bool,
) -> ProviderModelInput {
    ProviderModelInput {
        id: id.to_owned(),
        name: id.to_owned(),
        api: None,
        base_url: None,
        reasoning,
        thinking_level_map: None,
        input: vec![Modality::Text]
            .into_iter()
            .chain(image.then_some(Modality::Image))
            .collect(),
        cost: zero_cost(),
        context_window: context,
        max_tokens: max,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The runtime with the two table providers registered and their static
/// keys configured, upstream's `ModelRuntime.create` over an empty store.
async fn table_runtime() -> ModelRuntime {
    let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
        credentials: Some(empty_auth_storage()),
        models_path: Some(None),
        models_store: Some(Arc::new(InMemoryCodingAgentModelsStore::default())),
        allow_model_network: false,
        refresh_on_create: Some(false),
        ..CreateModelRuntimeOptions::default()
    })
    .await
    .expect("runtime creates");
    for (provider, key, models) in [
        (
            "alpha",
            "sk-alpha",
            vec![
                table_model("zeta", 1_500_000, 64_000, true, true),
                table_model("mike", 200_000, 8_000, false, false),
            ],
        ),
        (
            "beta",
            "sk-beta",
            vec![table_model("apricot", 999, 1_000, true, true)],
        ),
    ] {
        runtime
            .register_provider(
                provider,
                ProviderConfigInput {
                    name: Some(provider.to_owned()),
                    base_url: Some("https://example.test/v1".to_owned()),
                    api: Some(Api::from("openai-completions")),
                    api_key: Some(key.to_owned()),
                    models: Some(models),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the registration validates");
    }
    runtime
}

/// The captured writer as lines, upstream's `console.log` calls.
fn lines(buffer: &[u8]) -> Vec<String> {
    String::from_utf8(buffer.to_vec())
        .expect("the writers stay utf-8")
        .lines()
        .map(str::to_owned)
        .collect()
}

#[test]
fn formats_token_counts_across_the_scales() {
    assert_eq!(format_token_count(0), "0");
    assert_eq!(format_token_count(999), "999");
    assert_eq!(format_token_count(1_000), "1K");
    assert_eq!(format_token_count(1_500), "1.5K");
    assert_eq!(format_token_count(200_000), "200K");
    // The rounding restates upstream's `toFixed(1)`: 999.999K prints as
    // "1000.0K", the boundary the f64 quotient produces.
    assert_eq!(format_token_count(999_999), "1000.0K");
    assert_eq!(format_token_count(1_000_000), "1M");
    assert_eq!(format_token_count(1_500_000), "1.5M");
}

#[test]
fn renders_the_sorted_padded_table() {
    // The shapes the runtime fixture registers, built directly so no
    // environment can perturb the exact-grid assertion.
    let mut zeta = model("alpha", "zeta");
    zeta.context_window = 1_500_000;
    zeta.max_tokens = 64_000;
    zeta.reasoning = true;
    zeta.input = vec![Modality::Text, Modality::Image];
    let mut mike = model("alpha", "mike");
    mike.context_window = 200_000;
    mike.max_tokens = 8_000;
    let mut apricot = model("beta", "apricot");
    apricot.context_window = 999;
    apricot.max_tokens = 1_000;
    apricot.reasoning = true;
    apricot.input = vec![Modality::Text, Modality::Image];

    let table = render_models_table(&[zeta, mike, apricot], None).expect("table");
    // Widths: provider 8, model 7, context 7, max-out 7, thinking 8,
    // images 6. Every cell pads to its column, the last included, and the
    // join is two spaces.
    assert_eq!(
        table,
        vec![
            "provider  model    context  max-out  thinking  images",
            "alpha     mike     200K     8K       no        no    ",
            "alpha     zeta     1.5M     64K      yes       yes   ",
            "beta      apricot  999      1K       yes       yes   ",
        ]
    );
}

#[test]
fn renders_the_no_models_guidance_for_an_empty_catalog() {
    let table = render_models_table(&[], None).expect("table");
    assert_eq!(
        table,
        vec![pi_coding_agent::auth_guidance::format_no_models_available_message()]
    );
}

#[test]
fn filters_the_table_with_the_search_pattern() {
    let models = [
        model("alpha", "zeta"),
        model("alpha", "mike"),
        model("beta", "apricot"),
    ];

    // The empty pattern restates upstream's falsy check: no filter.
    let unfiltered = render_models_table(&models, Some("")).expect("table");
    assert_eq!(unfiltered.len(), 4);

    let beta_only = render_models_table(&models, Some("beta")).expect("table");
    assert_eq!(
        beta_only,
        vec![
            "provider  model    context  max-out  thinking  images",
            "beta      apricot  1K       100      no        no    ",
        ]
    );

    let zeta_only = render_models_table(&models, Some("zeta")).expect("table");
    assert_eq!(zeta_only.len(), 2);
    assert!(zeta_only[1].contains("zeta"), "{:?}", zeta_only[1]);

    let no_match = render_models_table(&models, Some("qqqq")).expect("table");
    assert_eq!(no_match, vec!["No models matching \"qqqq\""]);
}

#[tokio::test]
async fn lists_models_into_the_writers() {
    let runtime = table_runtime().await;
    let mut out: Vec<u8> = Vec::new();
    let mut err: Vec<u8> = Vec::new();
    list_models_into(&mut out, &mut err, &runtime, Some("zeta"), None)
        .await
        .expect("lists");
    // The filter drops every other provider's rows, ambient environment
    // included; the header rides along.
    assert_eq!(
        lines(&out).len(),
        2,
        "the filter keeps one row: {:?}",
        lines(&out)
    );
    assert!(lines(&err).is_empty(), "no warning: {:?}", lines(&err));

    let mut out: Vec<u8> = Vec::new();
    list_models_into(&mut out, &mut err, &runtime, None, None)
        .await
        .expect("lists");
    let table = lines(&out);
    assert!(
        table.iter().any(|line| line.starts_with("provider")),
        "the header prints: {table:?}"
    );
    assert!(
        table.iter().any(|line| line.starts_with("alpha")),
        "the alpha rows print: {table:?}"
    );
    assert!(
        table.iter().any(|line| line.starts_with("beta")),
        "the beta rows print: {table:?}"
    );

    // The no-match message, upstream's `No models matching "..."`.
    let mut out: Vec<u8> = Vec::new();
    list_models_into(&mut out, &mut err, &runtime, Some("qqqq"), None)
        .await
        .expect("lists");
    assert_eq!(lines(&out), vec!["No models matching \"qqqq\""]);
}

#[tokio::test]
async fn prints_the_load_warning_to_the_error_writer() {
    // The unparseable models.json restates upstream's
    // `modelRuntime.getError()` warning: the load failure prints to stderr
    // ahead of the catalog output.
    let temp = tempfile::tempdir().expect("tempdir");
    let models_path = temp.path().join("models.json");
    std::fs::write(&models_path, "{invalid-json").expect("models file");
    let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
        credentials: Some(empty_auth_storage()),
        models_path: Some(Some(models_path.display().to_string())),
        models_store: Some(Arc::new(InMemoryCodingAgentModelsStore::default())),
        allow_model_network: false,
        refresh_on_create: Some(false),
        ..CreateModelRuntimeOptions::default()
    })
    .await
    .expect("runtime creates");
    assert!(runtime.get_error().is_some(), "the load failure reports");

    let mut out: Vec<u8> = Vec::new();
    let mut err: Vec<u8> = Vec::new();
    list_models_into(&mut out, &mut err, &runtime, None, None)
        .await
        .expect("lists");
    let warning = lines(&err);
    assert!(
        warning
            .first()
            .is_some_and(|line| line == "Warning: errors loading models.json:"),
        "the warning leads stderr: {warning:?}"
    );
    assert!(!lines(&out).is_empty(), "the catalog output still prints");
}

#[tokio::test]
async fn lists_models_over_stdout() {
    // Upstream's `listModels` prints through `console`; the port's
    // stdout/stderr variant runs the same walk on the real writers.
    let runtime = table_runtime().await;
    list_models(&runtime, Some("zeta"), None)
        .await
        .expect("lists");
}
