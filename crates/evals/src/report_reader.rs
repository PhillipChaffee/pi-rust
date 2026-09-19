//! The inlined minimal report-reader slice of `@vitest-evals/core` (0.15.0,
//! getsentry/vitest-evals) that `report::read_task_observation` pins.
//!
//! Slice boundary, per ADR 0005: only what the 13 ported report tests
//! exercise — reading one Vitest JSON report file, validating its shape the
//! way `parseVitestJsonReport` does, extracting the eval/harness metadata
//! the way `readEvalTaskMeta` and `collectReportWorkspace` do (strict
//! objects: an unknown key drops the meta), and persisting the session
//! artifact under its sha256 identity directory. Everything else upstream
//! carries is either validated and dropped (run totals, file names,
//! assertion durations, transcript events) or validated leniently because
//! no ported test pins it (`eval.toolCalls`, `run.traces`); the rig tickets
//! revisit that surface when they land.
//!
//! Upstream reads the report file twice (once per reader entry point) and
//! runs both concurrently; this slice parses once and derives both the raw
//! assertions and the meta-bearing cases from the same validated value.

use std::io::Write as _;
use std::path::Path;

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::plan::EvalTask;
use crate::report::PI_SESSION_SNAPSHOT_ARTIFACT;

/// A Vitest assertion status as it appears in JSON reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReportStatus {
    /// The assertion passed.
    Passed,
    /// The assertion failed.
    Failed,
    /// The assertion was skipped.
    Skipped,
    /// The assertion was pending.
    Pending,
    /// The assertion was a todo stub.
    Todo,
    /// The assertion was disabled.
    Disabled,
}

impl ReportStatus {
    fn parse(status: &str) -> Option<Self> {
        match status {
            "passed" => Some(Self::Passed),
            "failed" => Some(Self::Failed),
            "skipped" => Some(Self::Skipped),
            "pending" => Some(Self::Pending),
            "todo" => Some(Self::Todo),
            "disabled" => Some(Self::Disabled),
            _ => None,
        }
    }
}

/// A validated Vitest JSON report, carrying the assertions the observation
/// reader matches against. Top-level totals are validated at parse time and
/// dropped — no ported test reads them.
#[derive(Debug, PartialEq)]
pub struct VitestJsonReport {
    /// Every test file's assertions, in report order.
    pub test_results: Vec<VitestTestFile>,
}

/// One test file's validated assertion list. File-level fields (message,
/// name, status, times) are validated at parse time and dropped.
#[derive(Debug, PartialEq)]
pub struct VitestTestFile {
    /// The file's assertions in reporter order.
    pub assertions: Vec<VitestJsonAssertion>,
}

/// One validated Vitest assertion. Only the fields the observation reader
/// reads are carried; the rest are validated at parse time and dropped.
#[derive(Debug, PartialEq)]
#[expect(
    clippy::derive_partial_eq_without_eq,
    reason = "the meta field is serde_json::Value, which implements PartialEq \
              but not Eq; that equality is exactly what the reader pins"
)]
pub struct VitestJsonAssertion {
    /// The full Vitest test name.
    pub full_name: String,
    /// The assertion's status.
    pub status: ReportStatus,
    /// The `eval`/`harness` metadata blob, when the assertion carries one.
    pub meta: Option<Value>,
}

/// One meta-bearing eval case collected from a report, mirroring
/// `collectReportWorkspace`'s case selection: assertions whose meta carries
/// an `eval` or `harness` object.
#[derive(Clone, Debug, PartialEq)]
pub struct ReportCase {
    /// The assertion's full name.
    pub full_name: String,
    /// The assertion's status.
    pub status: ReportStatus,
    /// The `eval` metadata, or the harness-backed fallback when only
    /// `harness` is present.
    pub eval: Option<EvalMeta>,
    /// The `harness` metadata.
    pub harness: Option<HarnessMeta>,
}

/// The `eval` metadata slice the observation reader reads: the average
/// score. Scores, threshold flag, output, and tool calls are validated at
/// parse time and dropped.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalMeta {
    /// `None` when the report carries no average score (absent or null);
    /// `Some` otherwise. Range is checked by `report::validate_score`.
    pub avg_score: Option<f64>,
}

/// The `harness` metadata slice the observation reader reads: the run.
/// The harness name is validated at parse time and dropped.
#[derive(Clone, Debug, PartialEq)]
pub struct HarnessMeta {
    /// The normalized run, when the harness recorded one.
    pub run: Option<HarnessRun>,
}

/// The harness-run slice the observation reader reads: usage, timings,
/// artifacts, and errors. Output, session, and traces are validated at
/// parse time and dropped.
#[derive(Clone, Debug, PartialEq)]
pub struct HarnessRun {
    /// Provider usage the run consumed.
    pub usage: UsageSummary,
    /// Wall-clock timing for the run, when reported.
    pub timings: Option<TimingSummary>,
    /// JSON artifacts the harness captured, keyed by artifact name.
    pub artifacts: Option<Map<String, Value>>,
    /// Normalized error objects the run captured; empty means none.
    pub errors: Vec<Map<String, Value>>,
}

/// The usage slice the observation reader reads. Retry and reasoning-token
/// counts are validated at parse time and dropped.
#[derive(Clone, Debug, PartialEq)]
pub struct UsageSummary {
    /// Provider that served the run.
    pub provider: Option<String>,
    /// Model that served the run.
    pub model: Option<String>,
    /// Input tokens consumed.
    pub input_tokens: Option<f64>,
    /// Output tokens produced.
    pub output_tokens: Option<f64>,
    /// Total tokens reported.
    pub total_tokens: Option<f64>,
    /// Tool calls observed.
    pub tool_calls: Option<f64>,
    /// Provider-specific usage details; cost estimates and cache counters
    /// live here.
    pub metadata: Option<Map<String, Value>>,
}

/// The timing slice the observation reader reads: the total duration.
#[derive(Clone, Debug, PartialEq)]
pub struct TimingSummary {
    /// End-to-end run duration in milliseconds.
    pub total_ms: Option<f64>,
}

/// The eval and harness metadata carried by one assertion, mirroring
/// `readEvalTaskMeta`'s parsed result.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalTaskMeta {
    /// The `eval` metadata, when present and valid.
    pub eval: Option<EvalMeta>,
    /// The `harness` metadata, when present and valid.
    pub harness: Option<HarnessMeta>,
}

/// Reads and validates one Vitest JSON report file.
///
/// Mirrors `@vitest-evals/core/node`'s `readVitestJsonReportFile`: UTF-8
/// file read, JSON parse, and full top-level shape validation (required
/// counters, `success` flag, test files and their assertions). Extra keys
/// are allowed, mirroring the upstream schemas' passthrough.
///
/// # Errors
///
/// Upstream wraps every failure as
/// `Failed to read eval result file <path>: <reason>`; the port keeps that
/// message shape for unreadable files, unparsable JSON, and shape
/// violations.
pub fn read_vitest_json_report_file(path: &Path) -> Result<VitestJsonReport, crate::EvalError> {
    let wrap = |reason: String| {
        crate::EvalError::new(format!(
            "Failed to read eval result file {}: {reason}",
            path.display()
        ))
    };
    let text = std::fs::read_to_string(path).map_err(|error| wrap(error.to_string()))?;
    let value: Value = serde_json::from_str(&text).map_err(|error| wrap(error.to_string()))?;
    validate_vitest_json_report(&value).map_err(|error| wrap(error.to_string()))
}

fn invalid() -> crate::EvalError {
    crate::EvalError::new("Invalid Vitest JSON report.")
}

fn finite(value: &Value) -> Option<f64> {
    value.as_f64().filter(|number| number.is_finite())
}

/// `OptionalFiniteNumberSchema`: null and non-finite numbers degrade to
/// absent; anything else (notably a string) fails.
fn optional_finite(value: &Value) -> bool {
    value.is_null() || finite(value).is_some() || matches!(value, Value::Number(_))
}

/// `NullableFiniteNumberSchema`'s acceptance rule: null and non-finite
/// numbers mean absent; anything else fails.
const fn nullable_finite_ok(value: &Value) -> bool {
    matches!(value, Value::Null | Value::Number(_))
}

fn strict_keys(object: &Map<String, Value>, allowed: &[&str]) -> bool {
    object.keys().all(|key| allowed.contains(&key.as_str()))
}

fn string_of<'a>(object: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    object.get(key).and_then(Value::as_str)
}

fn finite_of(object: &Map<String, Value>, key: &str) -> Option<f64> {
    object.get(key).and_then(finite)
}

fn validate_vitest_json_report(value: &Value) -> Result<VitestJsonReport, crate::EvalError> {
    let object = value.as_object().ok_or_else(invalid)?;
    for key in [
        "numFailedTests",
        "numPassedTests",
        "numPendingTests",
        "numTodoTests",
        "numTotalTests",
        "startTime",
    ] {
        finite_of(object, key).ok_or_else(invalid)?;
    }
    object
        .get("success")
        .and_then(Value::as_bool)
        .ok_or_else(invalid)?;
    let test_results = match object.get("testResults") {
        None => Vec::new(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|file| validate_test_file(file).ok_or_else(invalid))
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => return Err(invalid()),
    };
    Ok(VitestJsonReport { test_results })
}

fn validate_test_file(value: &Value) -> Option<VitestTestFile> {
    let object = value.as_object()?;
    string_of(object, "message")?;
    string_of(object, "name")?;
    matches!(string_of(object, "status"), Some("failed" | "passed")).then_some(())?;
    for key in ["startTime", "endTime"] {
        match object.get(key) {
            None => {}
            Some(present) if optional_finite(present) => {}
            Some(_) => return None,
        }
    }
    let assertion_items = match object.get("assertionResults") {
        None => Vec::new(),
        Some(Value::Array(items)) => items.clone(),
        Some(_) => return None,
    };
    let assertions = assertion_items
        .iter()
        .map(validate_assertion)
        .collect::<Option<Vec<_>>>()?;
    Some(VitestTestFile { assertions })
}

fn validate_assertion(value: &Value) -> Option<VitestJsonAssertion> {
    let object = value.as_object()?;
    match object.get("ancestorTitles") {
        None => {}
        Some(Value::Array(items)) => {
            items.iter().all(Value::is_string).then_some(())?;
        }
        Some(_) => return None,
    }
    let full_name = string_of(object, "fullName")?.to_string();
    let status = ReportStatus::parse(string_of(object, "status")?)?;
    string_of(object, "title")?;
    let meta = object.get("meta").cloned();
    match object.get("duration") {
        None | Some(Value::Null) => {}
        Some(duration) if finite(duration).is_some() => {}
        Some(_) => return None,
    }
    match object.get("failureMessages") {
        None | Some(Value::Null) => {}
        Some(Value::Array(items)) => {
            items.iter().all(Value::is_string).then_some(())?;
        }
        Some(_) => return None,
    }
    match object.get("location") {
        None | Some(Value::Null) => {}
        Some(location) => {
            let location = location.as_object()?;
            finite_of(location, "line")?;
            finite_of(location, "column")?;
        }
    }
    match object.get("tags") {
        None => {}
        Some(Value::Array(items)) => {
            items.iter().all(Value::is_string).then_some(())?;
        }
        Some(_) => return None,
    }
    Some(VitestJsonAssertion {
        full_name,
        status,
        meta,
    })
}

/// Reads the `eval`/`harness` metadata blob off one assertion, mirroring
/// upstream `readEvalTaskMeta`: absent from objects without either key,
/// `None` from objects whose keys fail strict validation.
///
/// Upstream validates with a Zod `.strict()` schema, so an unknown key
/// anywhere inside `eval`/`harness` drops the whole meta; the port keeps
/// that rejection behavior, which the report tests observe as an errored
/// outcome (`collectReportWorkspace` yields no case).
#[must_use]
pub fn read_eval_task_meta(meta: Option<&Value>) -> Option<EvalTaskMeta> {
    let object = meta?.as_object()?;
    let has_eval = object.contains_key("eval");
    let has_harness = object.contains_key("harness");
    if !has_eval && !has_harness {
        return None;
    }
    let eval = if has_eval {
        Some(parse_eval_meta(&object["eval"])?)
    } else {
        None
    };
    let harness = if has_harness {
        Some(parse_harness_meta(&object["harness"])?)
    } else {
        None
    };
    Some(EvalTaskMeta { eval, harness })
}

fn parse_eval_meta(value: &Value) -> Option<EvalMeta> {
    let object = value.as_object()?;
    if !strict_keys(
        object,
        &[
            "scores",
            "avgScore",
            "output",
            "thresholdFailed",
            "toolCalls",
        ],
    ) {
        return None;
    }
    let avg_score = match object.get("avgScore") {
        None => None,
        Some(score) if nullable_finite_ok(score) => finite(score),
        Some(_) => return None,
    };
    match object.get("scores") {
        None => {}
        Some(Value::Array(items)) => {
            for score in items {
                parse_eval_score(score)?;
            }
        }
        Some(_) => return None,
    }
    match object.get("thresholdFailed") {
        None => {}
        Some(flag) if flag.is_boolean() => {}
        Some(_) => return None,
    }
    // `output` is JSON-generic upstream, so any parsed value validates.
    // `toolCalls` has a structured schema no ported test pins; its elements
    // are validated leniently (objects) rather than fully, and dropped.
    match object.get("toolCalls") {
        None => {}
        Some(Value::Array(items)) => {
            items.iter().all(Value::is_object).then_some(())?;
        }
        Some(_) => return None,
    }
    Some(EvalMeta { avg_score })
}

fn parse_eval_score(value: &Value) -> Option<()> {
    let object = value.as_object()?;
    if !strict_keys(object, &["name", "score", "metadata"]) {
        return None;
    }
    match object.get("name") {
        None => {}
        Some(name) if name.is_string() => {}
        Some(_) => return None,
    }
    match object.get("score") {
        Some(score) if nullable_finite_ok(score) => {}
        _ => return None,
    }
    match object.get("metadata") {
        None => {}
        Some(metadata) if metadata.is_object() => {}
        Some(_) => return None,
    }
    Some(())
}

fn parse_harness_meta(value: &Value) -> Option<HarnessMeta> {
    let object = value.as_object()?;
    if !strict_keys(object, &["name", "run"]) {
        return None;
    }
    match object.get("name") {
        None => {}
        Some(name) if name.is_string() => {}
        Some(_) => return None,
    }
    let run = match object.get("run") {
        None => None,
        Some(run) => Some(parse_harness_run(run)?),
    };
    Some(HarnessMeta { run })
}

fn parse_harness_run(value: &Value) -> Option<HarnessRun> {
    let object = value.as_object()?;
    if !strict_keys(
        object,
        &[
            "output",
            "session",
            "usage",
            "timings",
            "artifacts",
            "traces",
            "errors",
        ],
    ) {
        return None;
    }
    let session = object.get("session")?;
    validate_session(session)?;
    let usage = parse_usage(object.get("usage")?)?;
    let timings = match object.get("timings") {
        None => None,
        Some(timings) => Some(parse_timings(timings)?),
    };
    let artifacts = match object.get("artifacts") {
        None => None,
        Some(artifacts) => Some(artifacts.as_object()?.clone()),
    };
    // `traces` carries a structured schema no ported test pins; validated
    // leniently as an array of objects.
    match object.get("traces") {
        None => {}
        Some(Value::Array(items)) => {
            items.iter().all(Value::is_object).then_some(())?;
        }
        Some(_) => return None,
    }
    let errors = object
        .get("errors")?
        .as_array()?
        .iter()
        .map(|error| error.as_object().cloned())
        .collect::<Option<Vec<_>>>()?;
    Some(HarnessRun {
        usage,
        timings,
        artifacts,
        errors,
    })
}

fn parse_usage(value: &Value) -> Option<UsageSummary> {
    let object = value.as_object()?;
    if !strict_keys(
        object,
        &[
            "provider",
            "model",
            "inputTokens",
            "outputTokens",
            "reasoningTokens",
            "totalTokens",
            "toolCalls",
            "retries",
            "metadata",
        ],
    ) {
        return None;
    }
    let provider = match object.get("provider") {
        None => None,
        Some(provider) => Some(provider.as_str()?.to_string()),
    };
    let model = match object.get("model") {
        None => None,
        Some(model) => Some(model.as_str()?.to_string()),
    };
    for key in [
        "inputTokens",
        "outputTokens",
        "reasoningTokens",
        "totalTokens",
        "toolCalls",
        "retries",
    ] {
        match object.get(key) {
            None => {}
            Some(value) if finite(value).is_some() => {}
            Some(_) => return None,
        }
    }
    let metadata = match object.get("metadata") {
        None => None,
        Some(metadata) => Some(metadata.as_object()?.clone()),
    };
    Some(UsageSummary {
        provider,
        model,
        input_tokens: finite_of(object, "inputTokens"),
        output_tokens: finite_of(object, "outputTokens"),
        total_tokens: finite_of(object, "totalTokens"),
        tool_calls: finite_of(object, "toolCalls"),
        metadata,
    })
}

fn parse_timings(value: &Value) -> Option<TimingSummary> {
    let object = value.as_object()?;
    if !strict_keys(object, &["totalMs", "metadata"]) {
        return None;
    }
    match object.get("totalMs") {
        None => {}
        Some(total_ms) if finite(total_ms).is_some() => {}
        Some(_) => return None,
    }
    match object.get("metadata") {
        None => {}
        Some(metadata) if metadata.is_object() => {}
        Some(_) => return None,
    }
    Some(TimingSummary {
        total_ms: finite_of(object, "totalMs"),
    })
}

fn validate_session(value: &Value) -> Option<()> {
    let object = value.as_object()?;
    if !strict_keys(object, &["events", "provider", "model", "metadata"]) {
        return None;
    }
    let events = object.get("events")?.as_array()?;
    for event in events {
        validate_transcript_event(event)?;
    }
    for key in ["provider", "model"] {
        match object.get(key) {
            None => {}
            Some(value) if value.is_string() => {}
            Some(_) => return None,
        }
    }
    match object.get("metadata") {
        None => {}
        Some(metadata) if metadata.is_object() => {}
        Some(_) => return None,
    }
    Some(())
}

fn validate_transcript_event(value: &Value) -> Option<()> {
    let object = value.as_object()?;
    match string_of(object, "type")? {
        "message" => {
            if !strict_keys(object, &["type", "role", "content", "metadata"]) {
                return None;
            }
            match string_of(object, "role")? {
                "system" | "user" | "assistant" => {}
                _ => return None,
            }
        }
        "tool_call" => {
            if !strict_keys(
                object,
                &[
                    "type",
                    "id",
                    "name",
                    "arguments",
                    "startedAt",
                    "finishedAt",
                    "durationMs",
                    "metadata",
                ],
            ) {
                return None;
            }
            string_of(object, "id")?;
            string_of(object, "name")?;
            for key in ["arguments", "metadata"] {
                match object.get(key) {
                    None => {}
                    Some(value) if value.is_object() => {}
                    Some(_) => return None,
                }
            }
            for key in ["startedAt", "finishedAt"] {
                match object.get(key) {
                    None => {}
                    Some(value) if value.is_string() => {}
                    Some(_) => return None,
                }
            }
            match object.get("durationMs") {
                None => {}
                Some(duration) if finite(duration).is_some() => {}
                Some(_) => return None,
            }
        }
        "tool_result" => {
            if !strict_keys(
                object,
                &[
                    "type",
                    "toolCallId",
                    "name",
                    "content",
                    "error",
                    "startedAt",
                    "finishedAt",
                    "durationMs",
                    "metadata",
                ],
            ) {
                return None;
            }
            string_of(object, "toolCallId")?;
            if let Some(name) = object.get("name") {
                name.as_str()?;
            }
            if let Some(error) = object.get("error") {
                validate_error(error)?;
            }
            for key in ["startedAt", "finishedAt"] {
                match object.get(key) {
                    None => {}
                    Some(value) if value.is_string() => {}
                    Some(_) => return None,
                }
            }
            match object.get("durationMs") {
                None => {}
                Some(duration) if finite(duration).is_some() => {}
                Some(_) => return None,
            }
            for key in ["metadata"] {
                match object.get(key) {
                    None => {}
                    Some(value) if value.is_object() => {}
                    Some(_) => return None,
                }
            }
        }
        _ => return None,
    }
    Some(())
}

fn validate_error(value: &Value) -> Option<()> {
    let object = value.as_object()?;
    string_of(object, "message")?;
    match object.get("type") {
        None => {}
        Some(type_name) if type_name.is_string() => {}
        Some(_) => return None,
    }
    Some(())
}

/// Collects the meta-bearing eval cases from one validated report, mirroring
/// `collectReportWorkspace`'s exercised behavior.
///
/// Every assertion whose meta carries a valid `eval` or `harness` object
/// becomes a case. When only `harness` is present, the case's eval metadata
/// falls back to the assertion status (passed scores 1, failed scores 0,
/// otherwise unscored).
#[must_use]
pub fn collect_report_workspace(report: &VitestJsonReport) -> Vec<ReportCase> {
    let mut cases = Vec::new();
    for file in &report.test_results {
        for assertion in &file.assertions {
            let Some(meta) = read_eval_task_meta(assertion.meta.as_ref()) else {
                continue;
            };
            let eval = meta.eval.or_else(|| {
                meta.harness.as_ref().map(|_| EvalMeta {
                    avg_score: match assertion.status {
                        ReportStatus::Passed => Some(1.0),
                        ReportStatus::Failed => Some(0.0),
                        _ => None,
                    },
                })
            });
            cases.push(ReportCase {
                full_name: assertion.full_name.clone(),
                status: assertion.status,
                eval,
                harness: meta.harness,
            });
        }
    }
    cases
}

/// Persists the case's session snapshot artifact, if the run captured one.
///
/// Mirrors upstream `persistSession`: the artifact named
/// [`PI_SESSION_SNAPSHOT_ARTIFACT`] is written to
/// `<artifact directory>/<variant>/sessions/<sha256 of the JSON-encoded
/// `[eval set, case, variant, model, run number]` identity>/session.jsonl`,
/// directory mode 0o700, file mode 0o600. No artifact string, no write.
///
/// # Errors
///
/// Propagates filesystem failures; upstream `readTaskObservation` rejects
/// its promise the same way rather than mapping them to an errored outcome.
pub fn persist_session(
    case_result: &ReportCase,
    task: &EvalTask,
    artifact_directory: &Path,
) -> std::io::Result<()> {
    let Some(session) = case_result
        .harness
        .as_ref()
        .and_then(|harness| harness.run.as_ref())
        .and_then(|run| run.artifacts.as_ref())
        .and_then(|artifacts| artifacts.get(PI_SESSION_SNAPSHOT_ARTIFACT))
        .and_then(Value::as_str)
    else {
        return Ok(());
    };
    let identity = serde_json::to_string(&Value::Array(vec![
        Value::String(task.eval_set.clone()),
        Value::String(task.case_id.clone()),
        Value::String(task.variant.as_str().to_string()),
        Value::String(task.model.clone()),
        Value::from(task.run_number),
    ]))
    .map_err(std::io::Error::other)?;
    let digest = identity_digest(&identity);
    let directory = artifact_directory
        .join(task.variant.as_str())
        .join("sessions")
        .join(digest);
    #[cfg(unix)]
    {
        use std::fs::DirBuilder;
        use std::os::unix::fs::DirBuilderExt as _;
        DirBuilder::new()
            .mode(0o700)
            .recursive(true)
            .create(&directory)?;
        let file_path = directory.join("session.jsonl");
        let mut file = {
            use std::os::unix::fs::OpenOptionsExt as _;
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(file_path)?
        };
        file.write_all(session.as_bytes())?;
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(&directory)?;
        let file_path = directory.join("session.jsonl");
        std::fs::write(file_path, session)?;
    }
    Ok(())
}

/// Hex-digests the identity JSON, mirroring upstream's
/// `createHash("sha256").update(identity).digest("hex")`.
fn identity_digest(identity: &str) -> String {
    use std::fmt::Write as _;
    let digest = Sha256::digest(identity.as_bytes());
    // `write!` into a `String` is infallible; the discard keeps the loop
    // honest without an unwrap.
    digest
        .iter()
        .fold(String::with_capacity(digest.len() * 2), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

#[cfg(test)]
mod tests {
    use super::identity_digest;

    // FIPS 180-4 vectors: the digest contract the artifact directory names
    // ride on, pinned beyond what the ported report tests observe.
    #[test]
    fn digests_like_node_crypto_sha256() {
        assert_eq!(
            identity_digest(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            identity_digest("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
