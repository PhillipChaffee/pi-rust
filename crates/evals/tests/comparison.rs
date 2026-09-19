//! Port of `packages/evals/test/comparison.test.ts` — the 4 portable
//! comparison cases, 1:1 against upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use pi_evals::plan::DocumentationVariant::{self, WithDocs, WithoutDocs};
use pi_evals::report::{
    EvalMetrics, EvalObservation, EvalOutcome, EvalRunIdentity, ExpectedEvalRun,
    PairedMetricSummary, format_eval_comparison_report, summarize_eval_observations,
};

/// A scored observation with the same defaults upstream's `scored` helper
/// carries; fields default to 100 tokens / 2 tools / 1000 ms / $0.01 and
/// each can be overridden or dropped (`None`).
#[derive(Clone, Copy)]
struct Metrics {
    total_tokens: Option<f64>,
    tool_calls: Option<f64>,
    total_ms: Option<f64>,
    estimated_cost_usd: Option<f64>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            total_tokens: Some(100.0),
            tool_calls: Some(2.0),
            total_ms: Some(1000.0),
            estimated_cost_usd: Some(0.01),
        }
    }
}

fn scored(
    variant: DocumentationVariant,
    run_number: u32,
    score: f64,
    metrics: Metrics,
) -> EvalObservation {
    EvalObservation {
        eval_set: String::from("tool access"),
        case_id: String::from("create"),
        variant,
        model: String::from("fixture/model"),
        run_number,
        outcome: EvalOutcome::Scored { score },
        metrics: EvalMetrics {
            input_tokens: None,
            output_tokens: None,
            cache_read_tokens: None,
            cache_write_tokens: None,
            total_tokens: metrics.total_tokens,
            tool_calls: metrics.tool_calls,
            total_ms: metrics.total_ms,
            estimated_cost_usd: metrics.estimated_cost_usd,
        },
    }
}

fn errored(
    variant: DocumentationVariant,
    run_number: u32,
    total_tokens: Option<f64>,
) -> EvalObservation {
    EvalObservation {
        eval_set: String::from("tool access"),
        case_id: String::from("create"),
        variant,
        model: String::from("fixture/model"),
        run_number,
        outcome: EvalOutcome::Errored,
        metrics: EvalMetrics {
            total_tokens,
            ..EvalMetrics::default()
        },
    }
}

fn expected_for(run_numbers: &[u32]) -> Vec<ExpectedEvalRun> {
    run_numbers
        .iter()
        .flat_map(|run_number| {
            [WithoutDocs, WithDocs]
                .into_iter()
                .map(move |variant| EvalRunIdentity {
                    eval_set: String::from("tool access"),
                    case_id: String::from("create"),
                    variant,
                    model: String::from("fixture/model"),
                    run_number: *run_number,
                })
        })
        .collect()
}

const fn summary(
    eligible_pairs: usize,
    control_mean: Option<f64>,
    treatment_mean: Option<f64>,
    mean_delta: Option<f64>,
) -> PairedMetricSummary {
    PairedMetricSummary {
        eligible_pairs,
        control_mean,
        treatment_mean,
        mean_delta,
    }
}

/// `node:util`'s `stripVTControlCharacters` over the sequences this
/// formatter emits: CSI sequences (`ESC [ params final-byte`) and
/// two-character escapes.
fn strip_vt(text: &str) -> String {
    let mut stripped = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(char) = chars.next() {
        if char != '\x1b' {
            stripped.push(char);
            continue;
        }
        if chars.peek() == Some(&'[') {
            chars.next();
            for next in chars.by_ref() {
                if ('\x40'..='\x7e').contains(&next) {
                    break;
                }
            }
        } else {
            chars.next();
        }
    }
    stripped
}

#[test]
fn computes_paired_lift_and_efficiency_deltas() {
    let observations = vec![
        scored(
            WithoutDocs,
            1,
            0.0,
            Metrics {
                tool_calls: Some(3.0),
                ..Metrics::default()
            },
        ),
        scored(
            WithDocs,
            1,
            1.0,
            Metrics {
                total_tokens: Some(120.0),
                tool_calls: Some(2.0),
                total_ms: Some(800.0),
                ..Metrics::default()
            },
        ),
        scored(
            WithoutDocs,
            2,
            1.0,
            Metrics {
                total_tokens: Some(200.0),
                ..Metrics::default()
            },
        ),
        scored(
            WithDocs,
            2,
            1.0,
            Metrics {
                total_tokens: Some(180.0),
                ..Metrics::default()
            },
        ),
    ];
    let report = summarize_eval_observations("digest", &expected_for(&[1, 2]), &observations);
    assert_eq!(report.comparisons.len(), 1);
    let comparison = &report.comparisons[0];
    assert_eq!(comparison.eval_set, "tool access");
    assert_eq!(comparison.total_pairs, 2);
    assert_eq!(comparison.eligible_pairs, 2);
    assert_eq!(comparison.blocked_pairs, 0);
    assert_eq!(comparison.control_pass_rate, Some(0.5));
    assert_eq!(comparison.treatment_pass_rate, Some(1.0));
    assert_eq!(comparison.lift, Some(0.5));
    assert_eq!(
        comparison.total_tokens,
        summary(2, Some(150.0), Some(150.0), Some(0.0))
    );
    assert_eq!(
        comparison.tool_calls,
        summary(2, Some(2.5), Some(2.0), Some(-0.5))
    );
}

#[test]
fn fails_closed_for_incomplete_and_errored_pairs_while_retaining_totals() {
    let observations = vec![
        scored(WithoutDocs, 1, 0.0, Metrics::default()),
        errored(WithDocs, 1, Some(120.0)),
        scored(
            WithoutDocs,
            2,
            1.0,
            Metrics {
                total_tokens: Some(200.0),
                ..Metrics::default()
            },
        ),
    ];
    let report = summarize_eval_observations("digest", &expected_for(&[1, 2]), &observations);
    assert_eq!(report.comparisons.len(), 1);
    let comparison = &report.comparisons[0];
    assert_eq!(comparison.total_pairs, 2);
    assert_eq!(comparison.eligible_pairs, 0);
    assert_eq!(comparison.blocked_pairs, 2);
    assert_eq!(comparison.lift, None);
    assert_eq!(report.blocked_pairs.len(), 2);
    assert_eq!(report.blocked_pairs[0].run_number, 1);
    assert_eq!(report.blocked_pairs[0].reasons, vec!["with_docs: errored"]);
    assert_eq!(report.blocked_pairs[1].run_number, 2);
    assert_eq!(
        report.blocked_pairs[1].reasons,
        vec!["with_docs: expected 1 observation, found 0"]
    );
    assert_eq!(report.operational_totals[0].total_tokens.available_runs, 2);
    assert_eq!(report.operational_totals[0].total_tokens.total, Some(300.0));
}

#[test]
fn blocks_duplicate_observations_and_keeps_missing_metrics_distinct_from_zero() {
    let without_tokens = scored(
        WithoutDocs,
        1,
        1.0,
        Metrics {
            total_tokens: None,
            ..Metrics::default()
        },
    );
    let observations = vec![
        without_tokens.clone(),
        without_tokens,
        scored(
            WithDocs,
            1,
            1.0,
            Metrics {
                total_tokens: Some(0.0),
                ..Metrics::default()
            },
        ),
    ];
    let report = summarize_eval_observations("digest", &expected_for(&[1]), &observations);
    assert_eq!(
        report.blocked_pairs[0].reasons,
        vec!["without_docs: expected 1 observation, found 2"]
    );
    assert_eq!(report.operational_totals[0].total_tokens.available_runs, 0);
    assert_eq!(report.operational_totals[0].total_tokens.total, None);
    assert_eq!(report.operational_totals[1].total_tokens.available_runs, 1);
    assert_eq!(report.operational_totals[1].total_tokens.total, Some(0.0));
}

#[test]
fn formats_blocked_comparisons_and_operational_totals() {
    let report = summarize_eval_observations(
        "digest",
        &expected_for(&[1, 2]),
        &[
            scored(WithoutDocs, 1, 1.0, Metrics::default()),
            scored(WithDocs, 1, 1.0, Metrics::default()),
        ],
    );
    let formatted = strip_vt(&format_eval_comparison_report(&report));
    assert!(formatted.contains("Documentation Eval Comparisons"));
    assert!(formatted.contains("Pass rate  withheld because pairs are blocked"));
    assert!(formatted.contains("without_docs: 1 runs"));
}

#[test]
fn formats_published_pass_rates_flags_and_costs() {
    let report = summarize_eval_observations(
        "digest",
        &expected_for(&[1, 2]),
        &[
            scored(WithoutDocs, 1, 0.0, Metrics::default()),
            scored(WithDocs, 1, 1.0, Metrics::default()),
            scored(WithoutDocs, 2, 0.0, Metrics::default()),
            scored(WithDocs, 2, 1.0, Metrics::default()),
        ],
    );
    let formatted = strip_vt(&format_eval_comparison_report(&report));
    assert!(formatted.contains("Pairs  2/2 eligible"));
    assert!(formatted.contains("Pass rate  +100.0 pp (with 100.0%, without 0.0%)"));
    assert!(formatted.contains("Flags  treatment-saturated"));
    assert!(formatted.contains("Est. cost  +$0.0000 (with $0.0100, without $0.0100, 2 pairs)"));
}

#[test]
fn formats_the_remaining_flags_and_negative_deltas() {
    // Two runs where treatment wins pair 1 but loses pair 2: treatment
    // saturates at 1 and the shared case outcome set turns flaky.
    let report = summarize_eval_observations(
        "digest",
        &expected_for(&[1, 2]),
        &[
            scored(
                WithoutDocs,
                1,
                1.0,
                Metrics {
                    tool_calls: Some(3.0),
                    ..Metrics::default()
                },
            ),
            scored(WithDocs, 1, 1.0, Metrics::default()),
            scored(WithoutDocs, 2, 0.0, Metrics::default()),
            scored(WithDocs, 2, 1.0, Metrics::default()),
        ],
    );
    let formatted = strip_vt(&format_eval_comparison_report(&report));
    assert!(formatted.contains("Pass rate  +50.0 pp (with 100.0%, without 50.0%)"));
    assert!(formatted.contains("Flags  treatment-saturated, flaky"));
    // Tool calls: control mean 2.5 against treatment mean 2.0.
    assert!(formatted.contains("Tools  -0.5 (with 2.0, without 2.5, 2 pairs)"));
}

#[test]
fn formats_unavailable_metrics_and_partial_operational_coverage() {
    let bare = Metrics {
        total_tokens: None,
        tool_calls: None,
        total_ms: None,
        estimated_cost_usd: None,
    };
    let observations = vec![
        scored(WithoutDocs, 1, 1.0, bare),
        scored(WithDocs, 1, 1.0, bare),
        // One of the two control runs reports tokens, so the total is partial.
        scored(
            WithoutDocs,
            2,
            1.0,
            Metrics {
                tool_calls: None,
                total_ms: None,
                estimated_cost_usd: None,
                ..Metrics::default()
            },
        ),
        scored(WithDocs, 2, 1.0, bare),
    ];
    let report = summarize_eval_observations("digest", &expected_for(&[1, 2]), &observations);
    let formatted = strip_vt(&format_eval_comparison_report(&report));
    assert!(formatted.contains("Tokens  unavailable"));
    assert!(formatted.contains("Est. cost  unavailable"));
    assert!(formatted.contains("without_docs: 2 runs, 100 tokens (1/2 measured), unavailable (0/2 measured), unavailable (0/2 measured), unavailable (0/2 measured)"));
}

#[test]
fn formats_the_blocked_reason_variants() {
    // Two planned runs per variant on run 1: the design-expected check
    // fails, and the observed-count reason pluralizes. Runs 2 and 3 each
    // fail on one unscored outcome spelling.
    let duplicated_expected = [expected_for(&[1]), expected_for(&[1])].concat();
    let mut expected = duplicated_expected;
    expected.extend(expected_for(&[2, 3, 4]));
    let observations = vec![
        pending(WithoutDocs, 1),
        skipped(WithDocs, 2),
        pending(WithoutDocs, 3),
        unscored(WithoutDocs, 4),
    ];
    let report = summarize_eval_observations("digest", &expected, &observations);
    let formatted = strip_vt(&format_eval_comparison_report(&report));
    assert!(formatted.contains("without_docs: design expected 1 run, found 2"));
    assert!(formatted.contains("without_docs: expected 2 observations, found 1"));
    assert!(formatted.contains("with_docs: expected 2 observations, found 0"));
    assert!(formatted.contains("without_docs: unscored"));
    assert!(formatted.contains("with_docs: skipped"));
    assert!(formatted.contains("without_docs: pending"));
    assert!(formatted.contains("with_docs: expected 1 observation, found 0"));
    assert!(formatted.contains("tool access/create/fixture/model/run-1: "));
}

/// An observation that completed without a score.
fn unscored(variant: DocumentationVariant, run_number: u32) -> EvalObservation {
    EvalObservation {
        eval_set: String::from("tool access"),
        case_id: String::from("create"),
        variant,
        model: String::from("fixture/model"),
        run_number,
        outcome: EvalOutcome::Unscored,
        metrics: EvalMetrics::default(),
    }
}

/// An observation the framework skipped before running.
fn skipped(variant: DocumentationVariant, run_number: u32) -> EvalObservation {
    EvalObservation {
        eval_set: String::from("tool access"),
        case_id: String::from("create"),
        variant,
        model: String::from("fixture/model"),
        run_number,
        outcome: EvalOutcome::Skipped,
        metrics: EvalMetrics::default(),
    }
}

/// An observation that has not run yet.
fn pending(variant: DocumentationVariant, run_number: u32) -> EvalObservation {
    EvalObservation {
        eval_set: String::from("tool access"),
        case_id: String::from("create"),
        variant,
        model: String::from("fixture/model"),
        run_number,
        outcome: EvalOutcome::Pending,
        metrics: EvalMetrics::default(),
    }
}

#[test]
fn formats_an_empty_comparison_set_as_an_empty_report() {
    let report = summarize_eval_observations("digest", &expected_for(&[]), &[]);
    assert_eq!(format_eval_comparison_report(&report), "");
}
