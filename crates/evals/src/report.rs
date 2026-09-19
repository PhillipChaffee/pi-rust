//! Observation reading, paired comparison, and report formatting, ported
//! from `packages/evals/src/report.ts`.

use std::path::Path;

use serde_json::Value;

use crate::EvalError;
use crate::plan::{DocumentationVariant, EvalTask};
use crate::report_reader::{self, ReportStatus};

/// The harness artifact name carrying a run's serialized session snapshot.
pub const PI_SESSION_SNAPSHOT_ARTIFACT: &str = "piSessionJsonl";

/// The control arm of the A/B comparison.
const CONTROL: DocumentationVariant = DocumentationVariant::WithoutDocs;
/// The treatment arm of the A/B comparison.
const TREATMENT: DocumentationVariant = DocumentationVariant::WithDocs;
/// Variant iteration order for blocked-pair reasons and totals, upstream's
/// `[CONTROL, TREATMENT]`.
const VARIANTS: [DocumentationVariant; 2] = [CONTROL, TREATMENT];

/// The identity of one planned or observed run, without its outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalRunIdentity {
    /// The eval set segment of the full name.
    pub eval_set: String,
    /// The case segment of the full name.
    pub case_id: String,
    /// Which documentation variant the run executes.
    pub variant: DocumentationVariant,
    /// Model identity, `"<provider>/<model>"`.
    pub model: String,
    /// One-based repetition number.
    pub run_number: u32,
}

/// The runs a completed experiment was planned to produce; the summary
/// blocks a pair unless every planned run was observed and scored.
pub type ExpectedEvalRun = EvalRunIdentity;

/// Wire-reported operational metrics of one run. Every count and cost comes
/// from the session backend's usage block; nothing is derived.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EvalMetrics {
    /// Input tokens consumed by the run, when the backend reported them.
    pub input_tokens: Option<f64>,
    /// Output tokens produced by the run, when the backend reported them.
    pub output_tokens: Option<f64>,
    /// Cache-read tokens, when the backend reported them.
    pub cache_read_tokens: Option<f64>,
    /// Cache-write tokens, when the backend reported them.
    pub cache_write_tokens: Option<f64>,
    /// Total tokens, when the backend reported them.
    pub total_tokens: Option<f64>,
    /// Tool calls the run made, when the backend reported them.
    pub tool_calls: Option<f64>,
    /// End-to-end duration in milliseconds, when the run reported it.
    pub total_ms: Option<f64>,
    /// Estimated cost in USD, when the backend reported it.
    pub estimated_cost_usd: Option<f64>,
}

/// What became of one planned run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EvalOutcome {
    /// The case completed and was scored at `score` in `[0, 1]`.
    Scored {
        /// The average judge score.
        score: f64,
    },
    /// The case completed but carried no score.
    Unscored,
    /// The case was skipped, todo, or disabled.
    Skipped,
    /// The case has not run yet.
    Pending,
    /// The case errored: infrastructure failure, run errors, or a failed
    /// identity/usage check.
    Errored,
}

impl EvalOutcome {
    /// The outcome spelling used in blocked-pair reasons.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Scored { .. } => "scored",
            Self::Unscored => "unscored",
            Self::Skipped => "skipped",
            Self::Pending => "pending",
            Self::Errored => "errored",
        }
    }
}

/// One planned run's outcome with whatever wire-reported metrics it carried.
///
/// Errored runs may still carry metrics — operational totals count them —
/// mirroring upstream's `{ ...identity, ...metrics, outcome: "errored" }`.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalObservation {
    /// Eval set segment of the case identity.
    pub eval_set: String,
    /// Case segment of the case identity.
    pub case_id: String,
    /// Which documentation variant ran.
    pub variant: DocumentationVariant,
    /// Model identity the task planned.
    pub model: String,
    /// One-based repetition number.
    pub run_number: u32,
    /// What became of the run.
    pub outcome: EvalOutcome,
    /// Wire-reported operational metrics, absent where unreported.
    pub metrics: EvalMetrics,
}

impl EvalObservation {
    /// The judge score, when the outcome is scored.
    #[must_use]
    pub const fn score(&self) -> Option<f64> {
        match self.outcome {
            EvalOutcome::Scored { score } => Some(score),
            _ => None,
        }
    }
}

/// Control/treatment means over the pairs where both sides reported a
/// metric; `mean_delta` is treatment minus control.
#[derive(Clone, Debug, PartialEq)]
pub struct PairedMetricSummary {
    /// Pairs eligible for this metric (both sides reported a value).
    pub eligible_pairs: usize,
    /// Mean over control observations, or none when no pair qualified.
    pub control_mean: Option<f64>,
    /// Mean over treatment observations, or none.
    pub treatment_mean: Option<f64>,
    /// Treatment mean minus control mean, precision-guarded.
    pub mean_delta: Option<f64>,
}

/// A comparison flag the summary raises about an eval set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparisonFlag {
    /// Both pass rates are equal.
    NoLift,
    /// Treatment pass rate is below control.
    NegativeDelta,
    /// Control pass rate is saturated at 1.
    ControlSaturated,
    /// Treatment pass rate is saturated at 1.
    TreatmentSaturated,
    /// A case passed in one pair and failed in another.
    Flaky,
}

impl ComparisonFlag {
    /// The wire spelling used in formatted reports.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::NoLift => "no-lift",
            Self::NegativeDelta => "negative-delta",
            Self::ControlSaturated => "control-saturated",
            Self::TreatmentSaturated => "treatment-saturated",
            Self::Flaky => "flaky",
        }
    }
}

/// The paired comparison of one eval set.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalSetComparison {
    /// The eval set's name.
    pub eval_set: String,
    /// Planned pairs for the eval set.
    pub total_pairs: usize,
    /// Pairs eligible for headline rates (both sides scored).
    pub eligible_pairs: usize,
    /// Planned pairs that failed their blocked checks.
    pub blocked_pairs: usize,
    /// Control pass rate, published only when no pair is blocked.
    pub control_pass_rate: Option<f64>,
    /// Treatment pass rate, published only when no pair is blocked.
    pub treatment_pass_rate: Option<f64>,
    /// Treatment pass rate minus control, precision-guarded.
    pub lift: Option<f64>,
    /// Flags raised for the eval set.
    pub flags: Vec<ComparisonFlag>,
    /// Paired totals for total token usage.
    pub total_tokens: PairedMetricSummary,
    /// Paired tool-call counts.
    pub tool_calls: PairedMetricSummary,
    /// Paired total durations, in milliseconds.
    pub total_ms: PairedMetricSummary,
    /// Paired estimated costs, in USD.
    pub estimated_cost_usd: PairedMetricSummary,
}

/// A planned pair that failed its blocked checks, with the reasons why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockedPair {
    /// Eval set segment of the case identity.
    pub eval_set: String,
    /// Case segment of the case identity.
    pub case_id: String,
    /// Model identity.
    pub model: String,
    /// Repetition number.
    pub run_number: u32,
    /// Every blocking reason, in upstream's per-variant check order.
    pub reasons: Vec<String>,
}

/// One operational metric summed over a variant's runs: how many runs
/// reported the metric, and their total.
///
/// `None` total with reported runs means no run measured the metric;
/// upstream keeps that distinct from a measured zero.
#[derive(Clone, Debug, PartialEq)]
pub struct OperationalMetricTotal {
    /// Runs that reported the metric.
    pub available_runs: usize,
    /// Sum over reporting runs, or none when no run reported it.
    pub total: Option<f64>,
}

/// Operational totals for one variant across all observed runs.
#[derive(Clone, Debug, PartialEq)]
pub struct VariantTotals {
    /// Which variant the totals cover.
    pub variant: DocumentationVariant,
    /// Observed runs of the variant.
    pub runs: usize,
    /// Input-token totals.
    pub input_tokens: OperationalMetricTotal,
    /// Output-token totals.
    pub output_tokens: OperationalMetricTotal,
    /// Cache-read-token totals.
    pub cache_read_tokens: OperationalMetricTotal,
    /// Cache-write-token totals.
    pub cache_write_tokens: OperationalMetricTotal,
    /// Total-token totals.
    pub total_tokens: OperationalMetricTotal,
    /// Tool-call totals.
    pub tool_calls: OperationalMetricTotal,
    /// Total-duration totals, in milliseconds.
    pub total_ms: OperationalMetricTotal,
    /// Estimated-cost totals, in USD.
    pub estimated_cost_usd: OperationalMetricTotal,
}

/// The full comparison report over one experiment's observations.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalComparisonReport {
    /// Report schema version, upstream's `schemaVersion: 3`.
    pub schema_version: u8,
    /// Digest of the experiment protocol the runs executed.
    pub protocol_digest: String,
    /// The control arm.
    pub control: DocumentationVariant,
    /// The treatment arm.
    pub treatment: DocumentationVariant,
    /// Per-eval-set comparisons, ordered by eval set.
    pub comparisons: Vec<EvalSetComparison>,
    /// Blocked pairs, ordered like the sorted groups.
    pub blocked_pairs: Vec<BlockedPair>,
    /// Operational totals per variant, control first then treatment.
    pub operational_totals: Vec<VariantTotals>,
}

/// Maps a Vitest assertion status to an observation outcome.
///
/// Mirrors upstream `classifyCaseStatus`: infrastructure failure errors,
/// skipped / todo / disabled skip, pending stays pending, and a passed
/// assertion carries no outcome of its own.
#[must_use]
pub const fn classify_case_status(status: ReportStatus) -> Option<EvalOutcome> {
    match status {
        ReportStatus::Failed => Some(EvalOutcome::Errored),
        ReportStatus::Skipped | ReportStatus::Todo | ReportStatus::Disabled => {
            Some(EvalOutcome::Skipped)
        }
        ReportStatus::Pending => Some(EvalOutcome::Pending),
        ReportStatus::Passed => None,
    }
}

/// The identity of one task, mirroring upstream `taskIdentity`.
fn task_identity(task: &EvalTask) -> EvalRunIdentity {
    EvalRunIdentity {
        eval_set: task.eval_set.clone(),
        case_id: task.case_id.clone(),
        variant: task.variant,
        model: task.model.clone(),
        run_number: task.run_number,
    }
}

fn observation(
    identity: EvalRunIdentity,
    outcome: EvalOutcome,
    metrics: EvalMetrics,
) -> EvalObservation {
    EvalObservation {
        eval_set: identity.eval_set,
        case_id: identity.case_id,
        variant: identity.variant,
        model: identity.model,
        run_number: identity.run_number,
        outcome,
        metrics,
    }
}

/// Builds the observation for a task whose run errored before any metric
/// could be read.
#[must_use]
pub fn errored_observation(task: &EvalTask) -> EvalObservation {
    observation(
        task_identity(task),
        EvalOutcome::Errored,
        EvalMetrics::default(),
    )
}

fn optional_metric(value: Option<f64>, name: &str) -> Result<Option<f64>, EvalError> {
    match value {
        None => Ok(None),
        Some(value) if value.is_finite() && value >= 0.0 => Ok(Some(value)),
        Some(_) => Err(EvalError::new(format!(
            "{name} must be a finite non-negative number."
        ))),
    }
}

fn validate_score(value: Option<f64>) -> Result<Option<f64>, EvalError> {
    match value {
        None => Ok(None),
        Some(score) if score.is_finite() && (0.0..=1.0).contains(&score) => Ok(Some(score)),
        Some(_) => Err(EvalError::new("Eval score must be between 0 and 1.")),
    }
}

fn build_metrics(
    usage: &report_reader::UsageSummary,
    timings: Option<&report_reader::TimingSummary>,
) -> Result<EvalMetrics, EvalError> {
    let metadata = usage.metadata.as_ref();
    Ok(EvalMetrics {
        input_tokens: optional_metric(usage.input_tokens, "inputTokens")?,
        output_tokens: optional_metric(usage.output_tokens, "outputTokens")?,
        cache_read_tokens: optional_metric(
            metadata
                .and_then(|map| map.get("cacheReadTokens"))
                .and_then(Value::as_f64),
            "cacheReadTokens",
        )?,
        cache_write_tokens: optional_metric(
            metadata
                .and_then(|map| map.get("cacheWriteTokens"))
                .and_then(Value::as_f64),
            "cacheWriteTokens",
        )?,
        total_tokens: optional_metric(usage.total_tokens, "totalTokens")?,
        tool_calls: optional_metric(usage.tool_calls, "toolCalls")?,
        total_ms: optional_metric(timings.and_then(|timings| timings.total_ms), "totalMs")?,
        estimated_cost_usd: optional_metric(
            metadata
                .and_then(|map| map.get("estimatedCostUsd"))
                .and_then(Value::as_f64),
            "estimatedCostUsd",
        )?,
    })
}

/// Reads one planned task's observation out of its Vitest JSON report.
///
/// Mirrors upstream `readTaskObservation`: the report must carry exactly one
/// assertion whose full name is `"<eval set> <case>"`, the collected case
/// must agree with it, and the harness run's reported model must match the
/// task. Status classification short-circuits skipped, pending, and failed
/// cases; otherwise usage metrics are validated, run errors and score-range
/// violations fail closed to an errored outcome, and a missing score yields
/// `unscored`.
///
/// # Errors
///
/// Filesystem failures while persisting the session artifact propagate as
/// `Err`, mirroring upstream's rejected promise; every report-level failure
/// becomes `Ok` with [`EvalOutcome::Errored`] instead.
pub fn read_task_observation(
    task: &EvalTask,
    report_path: &Path,
    artifact_directory: &Path,
) -> Result<EvalObservation, EvalError> {
    let Ok(parsed) = report_reader::read_vitest_json_report_file(report_path) else {
        return Ok(errored_observation(task));
    };
    let assertions = parsed
        .test_results
        .iter()
        .flat_map(|file| file.assertions.iter())
        .collect::<Vec<_>>();
    if assertions.len() != 1 {
        return Ok(errored_observation(task));
    }
    let assertion = assertions[0];
    let reported_full_name = format!("{} {}", task.eval_set, task.case_id);
    if assertion.full_name != reported_full_name {
        return Ok(errored_observation(task));
    }
    if let Some(outcome) = classify_case_status(assertion.status) {
        return Ok(observation(
            task_identity(task),
            outcome,
            EvalMetrics::default(),
        ));
    }
    let cases = report_reader::collect_report_workspace(&parsed);
    if cases.len() != 1 {
        return Ok(errored_observation(task));
    }
    let case_result = &cases[0];
    if case_result.full_name != reported_full_name {
        return Ok(errored_observation(task));
    }
    if case_result.status != assertion.status {
        return Ok(errored_observation(task));
    }
    let Some(run) = case_result
        .harness
        .as_ref()
        .and_then(|harness| harness.run.as_ref())
    else {
        return Ok(errored_observation(task));
    };
    report_reader::persist_session(case_result, task, artifact_directory)?;
    let actual_model = match (run.usage.provider.as_deref(), run.usage.model.as_deref()) {
        (Some(provider), Some(model)) => Some(format!("{provider}/{model}")),
        _ => None,
    };
    if actual_model.as_deref() != Some(task.model.as_str()) {
        return Ok(errored_observation(task));
    }
    let Ok(metrics) = build_metrics(&run.usage, run.timings.as_ref()) else {
        return Ok(errored_observation(task));
    };
    if !run.errors.is_empty() {
        return Ok(observation(
            task_identity(task),
            EvalOutcome::Errored,
            metrics,
        ));
    }
    let Ok(score) = validate_score(case_result.eval.as_ref().and_then(|eval| eval.avg_score))
    else {
        return Ok(observation(
            task_identity(task),
            EvalOutcome::Errored,
            metrics,
        ));
    };
    Ok(match score {
        None => observation(task_identity(task), EvalOutcome::Unscored, metrics),
        Some(score) => observation(task_identity(task), EvalOutcome::Scored { score }, metrics),
    })
}

/// A matched control/treatment pair of scored observations.
struct Pair {
    control: EvalObservation,
    treatment: EvalObservation,
}

/// One identity's planned and observed runs per variant.
struct PairGroup {
    eval_set: String,
    case_id: String,
    model: String,
    run_number: u32,
    expected: std::collections::HashMap<DocumentationVariant, usize>,
    observations: std::collections::HashMap<DocumentationVariant, Vec<EvalObservation>>,
}

/// Guards upstream's `Number((value).toPrecision(15))`: rounds to fifteen
/// significant decimal digits and parses back, so float noise from
/// subtraction does not reach the report.
fn difference(treatment: f64, control: f64) -> f64 {
    format!("{:.14e}", treatment - control)
        .parse::<f64>()
        .unwrap_or_default()
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / count_as_f64(values.len()))
    }
}

/// `usize` → `f64` for count arithmetic in means, rates, and totals.
#[expect(
    clippy::cast_precision_loss,
    reason = "the counts cast are pair and run counts, bounded far below f64's \
              exact-integer range; the upstream values are JS numbers"
)]
const fn count_as_f64(count: usize) -> f64 {
    count as f64
}

fn group_pairs(
    expected_runs: &[ExpectedEvalRun],
    observations: &[EvalObservation],
) -> Vec<PairGroup> {
    let mut groups: std::collections::HashMap<(String, String, String, u32), PairGroup> =
        std::collections::HashMap::new();
    for expected in expected_runs {
        let group = get_group(&mut groups, expected);
        *group.expected.entry(expected.variant).or_insert(0) += 1;
    }
    for observation in observations {
        let identity = EvalRunIdentity {
            eval_set: observation.eval_set.clone(),
            case_id: observation.case_id.clone(),
            variant: observation.variant,
            model: observation.model.clone(),
            run_number: observation.run_number,
        };
        let group = get_group(&mut groups, &identity);
        group
            .observations
            .entry(observation.variant)
            .or_default()
            .push(observation.clone());
    }
    let mut list: Vec<PairGroup> = groups.into_values().collect();
    list.sort_by(|left, right| {
        left.eval_set
            .cmp(&right.eval_set)
            .then(left.case_id.cmp(&right.case_id))
            .then(left.model.cmp(&right.model))
            .then(left.run_number.cmp(&right.run_number))
    });
    list
}

fn get_group<'a>(
    groups: &'a mut std::collections::HashMap<(String, String, String, u32), PairGroup>,
    identity: &EvalRunIdentity,
) -> &'a mut PairGroup {
    groups
        .entry((
            identity.eval_set.clone(),
            identity.case_id.clone(),
            identity.model.clone(),
            identity.run_number,
        ))
        .or_insert_with(|| PairGroup {
            eval_set: identity.eval_set.clone(),
            case_id: identity.case_id.clone(),
            model: identity.model.clone(),
            run_number: identity.run_number,
            expected: std::collections::HashMap::new(),
            observations: std::collections::HashMap::new(),
        })
}

fn resolve_pair(group: &PairGroup) -> Result<Pair, BlockedPair> {
    let identity = EvalRunIdentity {
        eval_set: group.eval_set.clone(),
        case_id: group.case_id.clone(),
        variant: CONTROL,
        model: group.model.clone(),
        run_number: group.run_number,
    };
    let mut reasons = Vec::new();
    for variant in VARIANTS {
        let expected = group.expected.get(&variant).copied().unwrap_or(0);
        let observed = group
            .observations
            .get(&variant)
            .map_or::<&[EvalObservation], _>(&[], Vec::as_slice);
        if expected != 1 {
            reasons.push(format!(
                "{variant}: design expected 1 run, found {expected}"
            ));
        }
        if observed.len() != expected {
            let plural = if expected == 1 { "" } else { "s" };
            reasons.push(format!(
                "{}: expected {expected} observation{plural}, found {}",
                variant.as_str(),
                observed.len()
            ));
        }
        if expected == 1
            && observed.len() == 1
            && !matches!(observed[0].outcome, EvalOutcome::Scored { .. })
        {
            reasons.push(format!(
                "{}: {}",
                variant.as_str(),
                observed[0].outcome.as_str()
            ));
        }
    }
    if !reasons.is_empty() {
        return Err(BlockedPair {
            eval_set: identity.eval_set,
            case_id: identity.case_id,
            model: identity.model,
            run_number: identity.run_number,
            reasons,
        });
    }
    Ok(Pair {
        control: scored_run(group, CONTROL),
        treatment: scored_run(group, TREATMENT),
    })
}

/// Extracts the single scored run a blocked-check-clean group holds for one
/// variant.
#[expect(
    clippy::expect_used,
    reason = "resolve_pair's reasons loop guarantees exactly one scored run per \
              variant; a miss is a logic bug worth panicking on rather than \
              publishing a silently mispaired report"
)]
fn scored_run(group: &PairGroup, variant: DocumentationVariant) -> EvalObservation {
    group
        .observations
        .get(&variant)
        .map_or::<&[EvalObservation], _>(&[], Vec::as_slice)
        .first()
        .expect("blocked checks guarantee one scored run per variant")
        .clone()
}

fn summarize_metric(
    pairs: &[Pair],
    select: impl Fn(&EvalMetrics) -> Option<f64>,
) -> PairedMetricSummary {
    let mut control = Vec::new();
    let mut treatment = Vec::new();
    for pair in pairs {
        let (Some(control_value), Some(treatment_value)) = (
            select(&pair.control.metrics),
            select(&pair.treatment.metrics),
        ) else {
            continue;
        };
        control.push(control_value);
        treatment.push(treatment_value);
    }
    let control_mean = mean(&control);
    let treatment_mean = mean(&treatment);
    PairedMetricSummary {
        eligible_pairs: control.len(),
        control_mean,
        treatment_mean,
        mean_delta: match (control_mean, treatment_mean) {
            (Some(control_mean), Some(treatment_mean)) => {
                Some(difference(treatment_mean, control_mean))
            }
            _ => None,
        },
    }
}

fn operational_total(
    runs: &[EvalObservation],
    select: impl Fn(&EvalMetrics) -> Option<f64>,
) -> OperationalMetricTotal {
    let values = runs
        .iter()
        .filter_map(|run| select(&run.metrics))
        .collect::<Vec<_>>();
    OperationalMetricTotal {
        available_runs: values.len(),
        total: if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>())
        },
    }
}

fn variant_totals(
    observations: &[EvalObservation],
    variant: DocumentationVariant,
) -> VariantTotals {
    let runs = observations
        .iter()
        .filter(|observation| observation.variant == variant)
        .cloned()
        .collect::<Vec<_>>();
    VariantTotals {
        variant,
        runs: runs.len(),
        input_tokens: totals(&runs, |metrics| metrics.input_tokens),
        output_tokens: totals(&runs, |metrics| metrics.output_tokens),
        cache_read_tokens: totals(&runs, |metrics| metrics.cache_read_tokens),
        cache_write_tokens: totals(&runs, |metrics| metrics.cache_write_tokens),
        total_tokens: totals(&runs, |metrics| metrics.total_tokens),
        tool_calls: totals(&runs, |metrics| metrics.tool_calls),
        total_ms: totals(&runs, |metrics| metrics.total_ms),
        estimated_cost_usd: totals(&runs, |metrics| metrics.estimated_cost_usd),
    }
}

fn totals(
    runs: &[EvalObservation],
    select: impl Fn(&EvalMetrics) -> Option<f64>,
) -> OperationalMetricTotal {
    operational_total(runs, select)
}

fn comparison_flags(
    pairs: &[Pair],
    control_pass_rate: Option<f64>,
    treatment_pass_rate: Option<f64>,
) -> Vec<ComparisonFlag> {
    #[expect(
        clippy::float_cmp,
        reason = "upstream compares pass rates with ===; the rates are quotients \
                  of identical small integer counts, so exact equality is the \
                  pinned behavior, not float slop"
    )]
    fn equal(left: f64, right: f64) -> bool {
        left == right
    }
    let mut flags = Vec::new();
    if let (Some(control_pass_rate), Some(treatment_pass_rate)) =
        (control_pass_rate, treatment_pass_rate)
    {
        if equal(control_pass_rate, treatment_pass_rate) {
            flags.push(ComparisonFlag::NoLift);
        }
        if treatment_pass_rate < control_pass_rate {
            flags.push(ComparisonFlag::NegativeDelta);
        }
        if equal(control_pass_rate, 1.0) {
            flags.push(ComparisonFlag::ControlSaturated);
        }
        if equal(treatment_pass_rate, 1.0) {
            flags.push(ComparisonFlag::TreatmentSaturated);
        }
    }
    let mut outcomes: std::collections::HashMap<(String, DocumentationVariant), Vec<bool>> =
        std::collections::HashMap::new();
    for pair in pairs {
        outcomes
            .entry((pair.control.case_id.clone(), pair.control.variant))
            .or_default()
            .push(pair.control.score().unwrap_or_default() >= 1.0);
        outcomes
            .entry((pair.treatment.case_id.clone(), pair.treatment.variant))
            .or_default()
            .push(pair.treatment.score().unwrap_or_default() >= 1.0);
    }
    if outcomes
        .values()
        .any(|values| values.contains(&true) && values.contains(&false))
    {
        flags.push(ComparisonFlag::Flaky);
    }
    flags
}

/// Summarizes one experiment's observations into a paired comparison report.
///
/// Mirrors upstream `summarizeEvalObservations`: planned runs and observed
/// runs are grouped by identity, each group is resolved into a scored pair
/// or a blocked pair with reasons, headline pass rates and lift publish only
/// when no pair of the eval set is blocked, and per-variant operational
/// totals sum whatever the wire reported.
#[must_use]
pub fn summarize_eval_observations(
    protocol_digest: &str,
    expected_runs: &[ExpectedEvalRun],
    observations: &[EvalObservation],
) -> EvalComparisonReport {
    let groups = group_pairs(expected_runs, observations);
    let mut blocked_pairs = Vec::new();
    let mut pairs_by_eval_set: std::collections::HashMap<String, Vec<Pair>> =
        std::collections::HashMap::new();
    // BTreeMap sorts by eval set, upstream's localeCompare of ASCII names.
    let mut totals_by_eval_set: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for group in &groups {
        *totals_by_eval_set
            .entry(group.eval_set.clone())
            .or_insert(0) += 1;
        match resolve_pair(group) {
            Err(blocked) => blocked_pairs.push(blocked),
            Ok(pair) => pairs_by_eval_set
                .entry(group.eval_set.clone())
                .or_default()
                .push(pair),
        }
    }
    let comparisons = totals_by_eval_set
        .iter()
        .map(|(eval_set, total_pairs)| {
            let pairs = pairs_by_eval_set
                .get(eval_set)
                .map_or::<&[Pair], _>(&[], Vec::as_slice);
            let blocked_pair_count = total_pairs - pairs.len();
            let publish_headline = blocked_pair_count == 0 && !pairs.is_empty();
            let pass_rate = |side: fn(&Pair) -> &EvalObservation| -> Option<f64> {
                if !publish_headline {
                    return None;
                }
                let passing = pairs
                    .iter()
                    .filter(|pair| side(pair).score().unwrap_or_default() >= 1.0)
                    .count();
                Some(count_as_f64(passing) / count_as_f64(pairs.len()))
            };
            let control_pass_rate = pass_rate(|pair| &pair.control);
            let treatment_pass_rate = pass_rate(|pair| &pair.treatment);
            let lift = match (control_pass_rate, treatment_pass_rate) {
                (Some(control), Some(treatment)) => Some(difference(treatment, control)),
                _ => None,
            };
            EvalSetComparison {
                eval_set: eval_set.clone(),
                total_pairs: *total_pairs,
                eligible_pairs: pairs.len(),
                blocked_pairs: blocked_pair_count,
                control_pass_rate,
                treatment_pass_rate,
                lift,
                flags: comparison_flags(pairs, control_pass_rate, treatment_pass_rate),
                total_tokens: summarize_metric(pairs, |metrics| metrics.total_tokens),
                tool_calls: summarize_metric(pairs, |metrics| metrics.tool_calls),
                total_ms: summarize_metric(pairs, |metrics| metrics.total_ms),
                estimated_cost_usd: summarize_metric(pairs, |metrics| metrics.estimated_cost_usd),
            }
        })
        .collect::<Vec<_>>();
    EvalComparisonReport {
        schema_version: 3,
        protocol_digest: protocol_digest.to_string(),
        control: CONTROL,
        treatment: TREATMENT,
        comparisons,
        blocked_pairs,
        operational_totals: vec![
            variant_totals(observations, CONTROL),
            variant_totals(observations, TREATMENT),
        ],
    }
}

fn percentage(value: Option<f64>) -> String {
    value.map_or_else(
        || String::from("unavailable"),
        |value| format!("{:.1}%", value * 100.0),
    )
}

fn signed(value: f64, digits: usize) -> String {
    let sign = if value >= 0.0 { "+" } else { "" };
    format!("{sign}{value:.digits$}")
}

/// Renders the bold-face header the way `node:util`'s `styleText("bold", …)`
/// does; the comparison report tests strip these sequences before matching.
fn style_text_bold(text: &str) -> String {
    format!("\x1b[1m{text}\x1b[22m")
}

fn paired_metric(label: &str, metric: &PairedMetricSummary, unit: &str) -> String {
    match (
        metric.mean_delta,
        metric.control_mean,
        metric.treatment_mean,
    ) {
        (Some(delta), Some(control_mean), Some(treatment_mean)) => format!(
            "    {:>10}  {}{} (with {:.1}, without {:.1}, {} pairs)",
            label,
            signed(delta, 1),
            unit,
            treatment_mean,
            control_mean,
            metric.eligible_pairs
        ),
        _ => format!("    {label:>10}  unavailable"),
    }
}

fn operational_metric(
    metric: &OperationalMetricTotal,
    runs: usize,
    format_total: impl Fn(f64) -> String,
) -> String {
    let Some(total) = metric.total else {
        return format!("unavailable (0/{runs} measured)");
    };
    let coverage = if metric.available_runs == runs {
        String::new()
    } else {
        format!(" ({}/{} measured)", metric.available_runs, runs)
    };
    format!("{}{}", format_total(total), coverage)
}

/// Formats the comparison report for terminal output, mirroring upstream
/// `formatEvalComparisonReport` line for line, including ANSI bold on the
/// header.
pub fn format_eval_comparison_report(report: &EvalComparisonReport) -> String {
    if report.comparisons.is_empty() {
        return String::new();
    }
    let mut lines = vec![style_text_bold("Documentation Eval Comparisons")];
    for comparison in &report.comparisons {
        lines.push(format!("  {}", comparison.eval_set));
        lines.push(format!(
            "         Pairs  {}/{} eligible",
            comparison.eligible_pairs, comparison.total_pairs
        ));
        match comparison.lift {
            None => lines.push(
                if comparison.blocked_pairs > 0 {
                    "     Pass rate  withheld because pairs are blocked"
                } else {
                    "     Pass rate  unavailable"
                }
                .to_string(),
            ),
            Some(lift) => lines.push(format!(
                "     Pass rate  {} pp (with {}, without {})",
                signed(lift * 100.0, 1),
                percentage(comparison.treatment_pass_rate),
                percentage(comparison.control_pass_rate)
            )),
        }
        if !comparison.flags.is_empty() {
            lines.push(format!(
                "         Flags  {}",
                comparison
                    .flags
                    .iter()
                    .map(ComparisonFlag::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        lines.push(paired_metric("Tokens", &comparison.total_tokens, ""));
        lines.push(paired_metric("Tools", &comparison.tool_calls, ""));
        lines.push(paired_metric("Latency", &comparison.total_ms, "ms"));
        let cost = &comparison.estimated_cost_usd;
        match (cost.mean_delta, cost.control_mean, cost.treatment_mean) {
            (Some(delta), Some(control_mean), Some(treatment_mean)) => {
                let sign = if delta >= 0.0 { "+" } else { "-" };
                lines.push(format!(
                    "     Est. cost  {sign}${:.4} (with ${:.4}, without ${:.4}, {} pairs)",
                    delta.abs(),
                    treatment_mean,
                    control_mean,
                    cost.eligible_pairs
                ));
            }
            _ => lines.push("     Est. cost  unavailable".to_string()),
        }
    }
    lines.push(String::from("  Operational totals"));
    for totals in &report.operational_totals {
        let tokens = operational_metric(&totals.total_tokens, totals.runs, |total| {
            format!("{total} tokens")
        });
        let tools = operational_metric(&totals.tool_calls, totals.runs, |total| {
            format!("{total} tools")
        });
        let latency = operational_metric(&totals.total_ms, totals.runs, |total| {
            format!("{:.2}s", total / 1000.0)
        });
        let cost = operational_metric(&totals.estimated_cost_usd, totals.runs, |total| {
            format!("${total:.4} cost")
        });
        lines.push(format!(
            "    {}: {} runs, {tokens}, {tools}, {latency}, {cost}",
            totals.variant, totals.runs
        ));
    }
    if !report.blocked_pairs.is_empty() {
        lines.push(String::from("  Blocked pairs"));
        for blocked in &report.blocked_pairs {
            lines.push(format!(
                "    {}/{}/{}/run-{}: {}",
                blocked.eval_set,
                blocked.case_id,
                blocked.model,
                blocked.run_number,
                blocked.reasons.join("; ")
            ));
        }
    }
    lines.join("\n")
}
