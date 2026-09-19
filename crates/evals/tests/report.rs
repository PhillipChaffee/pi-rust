//! Port of `packages/evals/test/report.test.ts` — the 13 portable report
//! cases, 1:1 against upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//! The Vitest JSON fixture, the `scoredMeta` blob, and every assertion carry
//! upstream's shapes unchanged.
#![expect(
    clippy::unwrap_used,
    reason = "fixture IO and observation reads failing is the test environment \
              failing; unwrapping keeps the ported assertions readable, and a \
              failure still fails the test with the underlying message"
)]

use std::path::PathBuf;

use pi_evals::plan::{DocumentationVariant, EvalTask};
use pi_evals::report::{
    EvalMetrics, EvalObservation, EvalOutcome, classify_case_status, read_task_observation,
};
use pi_evals::report_reader::ReportStatus;
use serde_json::{Value, json};

const SESSION: &str = "{\"type\":\"session\"}\n";

fn task() -> EvalTask {
    EvalTask {
        file: String::from("evals/example.docs.eval.ts"),
        full_name: String::from("Example workflow > handles the case"),
        eval_set: String::from("Example workflow"),
        case_id: String::from("handles the case"),
        variant: DocumentationVariant::WithoutDocs,
        model: String::from("fixture/model"),
        run_number: 1,
    }
}

/// One Vitest assertion's fixture overrides; `None` status means "passed"
/// and `None` meta means an empty object, upstream's default arguments.
#[derive(Default)]
struct ReportAssertion {
    status: Option<&'static str>,
    meta: Option<Value>,
}

/// Writes the single-assertion Vitest JSON report upstream's
/// `writeTaskReport` fixture builds, into a fresh temp directory.
fn write_task_report(assertion: &ReportAssertion) -> (tempfile::TempDir, PathBuf) {
    let directory = tempfile::tempdir().unwrap();
    let report_path = directory.path().join("vitest.json");
    let status = assertion.status.unwrap_or("passed");
    let num_passed = u8::from(status == "passed");
    let num_pending = u8::from(status == "pending" || status == "skipped");
    let report = json!({
        "numFailedTests": 0,
        "numPassedTests": num_passed,
        "numPendingTests": num_pending,
        "numTodoTests": 0,
        "numTotalTests": 1,
        "startTime": 0,
        "success": true,
        "testResults": [
            {
                "message": "",
                "name": "/repo/packages/evals/evals/example.docs.eval.ts",
                "status": "passed",
                "assertionResults": [
                    {
                        "ancestorTitles": [task().eval_set],
                        "fullName": format!("{} {}", task().eval_set, task().case_id),
                        "status": status,
                        "title": task().case_id,
                        "failureMessages": [],
                        "meta": assertion.meta.clone().unwrap_or_else(|| json!({})),
                    }
                ],
            }
        ],
    });
    std::fs::write(&report_path, serde_json::to_string(&report).unwrap()).unwrap();
    (directory, report_path)
}

/// The scored eval/harness metadata blob upstream's `scoredMeta` carries,
/// with the same override points: average score, reported model, run errors,
/// and captured artifacts.
fn scored_meta(overrides: &ScoredOverrides) -> Value {
    json!({
        "eval": {
            "avgScore": overrides.avg_score.clone().unwrap_or_else(|| json!(0.5)),
            "scores": [{ "name": "StructuredOutputJudge", "score": 0.5 }],
            "thresholdFailed": false,
        },
        "harness": {
            "name": "without_docs",
            "run": {
                "output": { "ok": true },
                "session": { "events": [{ "type": "message", "role": "user", "content": "prompt" }] },
                "usage": {
                    "provider": "fixture",
                    "model": overrides.model.unwrap_or("model"),
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "totalTokens": 15,
                    "toolCalls": 1,
                    "metadata": { "cacheReadTokens": 2, "cacheWriteTokens": 3, "estimatedCostUsd": 0.01 },
                },
                "timings": { "totalMs": 1234 },
                "artifacts": overrides
                    .artifacts
                    .clone()
                    .unwrap_or_else(|| json!({ "runId": "run-1", "piSessionJsonl": SESSION })),
                "errors": overrides.errors.clone().unwrap_or_else(|| json!([])),
            }
        }
    })
}

#[derive(Default)]
struct ScoredOverrides {
    avg_score: Option<Value>,
    model: Option<&'static str>,
    errors: Option<Value>,
    artifacts: Option<Value>,
}

fn read_observation(assertion: &ReportAssertion) -> (tempfile::TempDir, EvalObservation) {
    let (directory, report_path) = write_task_report(assertion);
    let observation = read_task_observation(&task(), &report_path, directory.path()).unwrap();
    (directory, observation)
}

#[test]
fn maps_skipped_to_skipped() {
    assert_eq!(
        classify_case_status(ReportStatus::Skipped),
        Some(EvalOutcome::Skipped)
    );
}

#[test]
fn maps_todo_to_skipped() {
    assert_eq!(
        classify_case_status(ReportStatus::Todo),
        Some(EvalOutcome::Skipped)
    );
}

#[test]
fn maps_disabled_to_skipped() {
    assert_eq!(
        classify_case_status(ReportStatus::Disabled),
        Some(EvalOutcome::Skipped)
    );
}

#[test]
fn maps_failed_infrastructure_to_errored() {
    assert_eq!(
        classify_case_status(ReportStatus::Failed),
        Some(EvalOutcome::Errored)
    );
}

#[test]
fn preserves_a_skipped_outcome_when_no_harness_run_exists() {
    let (_, observation) = read_observation(&ReportAssertion {
        status: Some("skipped"),
        ..Default::default()
    });
    assert_eq!(observation.outcome, EvalOutcome::Skipped);
}

#[test]
fn preserves_a_pending_outcome_when_no_harness_run_exists() {
    let (_, observation) = read_observation(&ReportAssertion {
        status: Some("pending"),
        ..Default::default()
    });
    assert_eq!(observation.outcome, EvalOutcome::Pending);
}

#[test]
fn records_an_errored_outcome_when_a_passed_eval_has_no_harness_run() {
    let (_, observation) = read_observation(&ReportAssertion::default());
    assert_eq!(observation.outcome, EvalOutcome::Errored);
}

#[test]
fn reads_a_scored_harness_run_and_persists_the_session_artifact() {
    let (directory, observation) = read_observation(&ReportAssertion {
        meta: Some(scored_meta(&ScoredOverrides::default())),
        ..Default::default()
    });
    let planned = task();
    assert_eq!(
        observation,
        EvalObservation {
            eval_set: planned.eval_set.clone(),
            case_id: planned.case_id.clone(),
            variant: planned.variant,
            model: planned.model.clone(),
            run_number: planned.run_number,
            outcome: EvalOutcome::Scored { score: 0.5 },
            metrics: EvalMetrics {
                input_tokens: Some(10.0),
                output_tokens: Some(5.0),
                cache_read_tokens: Some(2.0),
                cache_write_tokens: Some(3.0),
                total_tokens: Some(15.0),
                tool_calls: Some(1.0),
                total_ms: Some(1234.0),
                estimated_cost_usd: Some(0.01),
            },
        }
    );
    let sessions = directory
        .path()
        .join(planned.variant.as_str())
        .join("sessions");
    let entries: Vec<PathBuf> = std::fs::read_dir(sessions)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(entries.len(), 1);
    let persisted = std::fs::read_to_string(entries[0].join("session.jsonl")).unwrap();
    assert_eq!(persisted, SESSION);
}

#[test]
fn treats_a_zero_score_as_scored_data() {
    let (_, observation) = read_observation(&ReportAssertion {
        meta: Some(scored_meta(&ScoredOverrides {
            avg_score: Some(json!(0)),
            ..Default::default()
        })),
        ..Default::default()
    });
    assert_eq!(observation.outcome, EvalOutcome::Scored { score: 0.0 });
}

#[test]
fn records_an_unscored_outcome_when_a_completed_eval_has_no_score() {
    let (_, observation) = read_observation(&ReportAssertion {
        meta: Some(scored_meta(&ScoredOverrides {
            avg_score: Some(json!(null)),
            ..Default::default()
        })),
        ..Default::default()
    });
    assert_eq!(observation.outcome, EvalOutcome::Unscored);
}

#[test]
fn records_an_errored_outcome_when_the_reported_model_does_not_match_the_task() {
    let (_, observation) = read_observation(&ReportAssertion {
        meta: Some(scored_meta(&ScoredOverrides {
            model: Some("other"),
            ..Default::default()
        })),
        ..Default::default()
    });
    assert_eq!(observation.outcome, EvalOutcome::Errored);
}

#[test]
fn records_an_errored_outcome_when_a_completed_harness_run_contains_errors() {
    let (_, observation) = read_observation(&ReportAssertion {
        meta: Some(scored_meta(&ScoredOverrides {
            errors: Some(json!([{ "message": "boom" }])),
            ..Default::default()
        })),
        ..Default::default()
    });
    assert_eq!(observation.outcome, EvalOutcome::Errored);
}

#[test]
fn still_scores_a_completed_eval_when_the_session_artifact_is_missing() {
    let (_, observation) = read_observation(&ReportAssertion {
        meta: Some(scored_meta(&ScoredOverrides {
            artifacts: Some(json!({ "runId": "run-1" })),
            ..Default::default()
        })),
        ..Default::default()
    });
    assert_eq!(observation.outcome, EvalOutcome::Scored { score: 0.5 });
}

#[test]
fn records_an_errored_outcome_when_the_reported_score_is_out_of_range() {
    let (_, observation) = read_observation(&ReportAssertion {
        meta: Some(scored_meta(&ScoredOverrides {
            avg_score: Some(json!(1.5)),
            ..Default::default()
        })),
        ..Default::default()
    });
    assert_eq!(observation.outcome, EvalOutcome::Errored);
}

#[test]
fn records_an_errored_outcome_without_metrics_when_a_usage_metric_is_negative() {
    // Negative usage cannot reach the report: the reader's optional-metric
    // validation fails closed before any metric is carried.
    let (_, observation) = read_observation(&ReportAssertion {
        meta: Some(scored_meta_with(|meta| {
            meta["harness"]["run"]["usage"]["inputTokens"] = json!(-1);
        })),
        ..Default::default()
    });
    assert_eq!(observation.outcome, EvalOutcome::Errored);
    assert_eq!(observation.metrics, EvalMetrics::default());
}

/// Builds the scored blob, mutates it, and reads the observation.
fn scored_meta_with(mutate: impl FnOnce(&mut Value)) -> Value {
    let mut meta = scored_meta(&ScoredOverrides::default());
    mutate(&mut meta);
    meta
}

#[test]
fn carries_todo_and_disabled_and_failed_statuses_through_the_reader() {
    for (status, expected) in [
        ("todo", EvalOutcome::Skipped),
        ("disabled", EvalOutcome::Skipped),
        ("failed", EvalOutcome::Errored),
    ] {
        let (_, observation) = read_observation(&ReportAssertion {
            status: Some(status),
            ..Default::default()
        });
        assert_eq!(observation.outcome, expected, "status {status}");
    }
}
