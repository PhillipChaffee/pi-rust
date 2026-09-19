//! Additional tests on top of the 21 ported cases: the inlined reader
//! slice's validation boundaries and error paths that the ported report
//! tests do not pin but the crate's behavior guarantees. These keep the
//! strict-meta acceptance rules and the persist-failure contract from
//! silently drifting from the upstream slice.
#![expect(
    clippy::unwrap_used,
    reason = "fixture IO failing is the test environment failing; unwrapping \
              keeps these boundary assertions readable"
)]

use std::path::PathBuf;

use pi_evals::plan::{DocumentationVariant, EvalTask};
use pi_evals::report::{EvalOutcome, read_task_observation};
use pi_evals::report_reader::{
    collect_report_workspace, read_eval_task_meta, read_vitest_json_report_file,
};
use serde_json::{Value, json};

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

/// A minimal valid report with one passed assertion carrying the scored
/// meta blob, mutated by each boundary test before it is written.
fn scored_report(mutate: impl FnOnce(&mut Value)) -> (tempfile::TempDir, PathBuf) {
    let directory = tempfile::tempdir().unwrap();
    let report_path = directory.path().join("vitest.json");
    let mut report = json!({
        "numFailedTests": 0,
        "numPassedTests": 1,
        "numPendingTests": 0,
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
                        "ancestorTitles": ["Example workflow"],
                        "fullName": "Example workflow handles the case",
                        "status": "passed",
                        "title": "handles the case",
                        "failureMessages": [],
                        "meta": {
                            "eval": { "avgScore": 0.5, "scores": [{ "name": "StructuredOutputJudge", "score": 0.5 }], "thresholdFailed": false },
                            "harness": {
                                "name": "without_docs",
                                "run": {
                                    "session": { "events": [{ "type": "message", "role": "user", "content": "prompt" }] },
                                    "usage": { "provider": "fixture", "model": "model", "inputTokens": 10, "outputTokens": 5, "totalTokens": 15, "toolCalls": 1, "metadata": {} },
                                    "artifacts": { "runId": "run-1" },
                                    "errors": []
                                }
                            }
                        }
                    }
                ],
            }
        ],
    });
    mutate(&mut report);
    std::fs::write(&report_path, serde_json::to_string(&report).unwrap()).unwrap();
    (directory, report_path)
}

fn outcome_of(report_path: &std::path::Path) -> EvalOutcome {
    let artifact_directory = report_path.parent().unwrap().to_path_buf();
    read_task_observation(&task(), report_path, &artifact_directory)
        .unwrap()
        .outcome
}

/// Pins the shared fixture itself to the scored path: without this, a
/// broken fixture would make every boundary test below pass vacuously.
#[test]
fn unmutated_fixture_reads_as_scored() {
    let (directory, path) = scored_report(|_report| {});
    assert_eq!(outcome_of(&path), EvalOutcome::Scored { score: 0.5 });
    drop(directory);
}

#[test]
fn errored_when_the_report_file_is_missing() {
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("missing.json");
    assert_eq!(outcome_of(&missing), EvalOutcome::Errored);
}

#[test]
fn errored_when_the_report_is_not_json() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vitest.json");
    std::fs::write(&path, "not json").unwrap();
    assert_eq!(outcome_of(&path), EvalOutcome::Errored);
    drop(directory);
}

#[test]
fn errored_when_the_report_shape_fails_validation() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vitest.json");
    std::fs::write(&path, "{\"numTotalTests\": 0}").unwrap();
    assert_eq!(outcome_of(&path), EvalOutcome::Errored);
    drop(directory);
}

#[test]
fn errored_when_assertions_are_not_exactly_one() {
    let (directory, path) = scored_report(|report| {
        let second = report["testResults"][0]["assertionResults"][0].clone();
        report["testResults"][0]["assertionResults"]
            .as_array_mut()
            .unwrap()
            .push(second);
    });
    assert_eq!(outcome_of(&path), EvalOutcome::Errored);
    drop(directory);
}

#[test]
fn errored_when_the_full_name_does_not_match_the_task() {
    let (directory, path) = scored_report(|report| {
        report["testResults"][0]["assertionResults"][0]["fullName"] =
            json!("Different set > different case");
    });
    assert_eq!(outcome_of(&path), EvalOutcome::Errored);
    drop(directory);
}

#[test]
fn errored_when_meta_carries_an_unknown_eval_key() {
    let (directory, path) = scored_report(|report| {
        report["testResults"][0]["assertionResults"][0]["meta"]["eval"]["bogus"] = json!(1);
    });
    assert_eq!(outcome_of(&path), EvalOutcome::Errored);
    drop(directory);
}

#[test]
fn errored_when_meta_carries_an_unknown_harness_key() {
    let (directory, path) = scored_report(|report| {
        report["testResults"][0]["assertionResults"][0]["meta"]["harness"]["bogus"] = json!(1);
    });
    assert_eq!(outcome_of(&path), EvalOutcome::Errored);
    drop(directory);
}

#[test]
fn errored_when_a_transcript_event_fails_validation() {
    let (directory, path) = scored_report(|report| {
        report["testResults"][0]["assertionResults"][0]["meta"]["harness"]["run"]["session"]["events"] =
            json!([{ "type": "nonsense" }]);
    });
    assert_eq!(outcome_of(&path), EvalOutcome::Errored);
    drop(directory);
}

#[test]
fn errored_when_a_usage_token_count_is_not_a_number() {
    let (directory, path) = scored_report(|report| {
        report["testResults"][0]["assertionResults"][0]["meta"]["harness"]["run"]["usage"]["inputTokens"] =
            json!("10");
    });
    assert_eq!(outcome_of(&path), EvalOutcome::Errored);
    drop(directory);
}

#[test]
fn errored_when_traces_are_not_an_array() {
    let (directory, path) = scored_report(|report| {
        report["testResults"][0]["assertionResults"][0]["meta"]["harness"]["run"]["traces"] =
            json!("x");
    });
    assert_eq!(outcome_of(&path), EvalOutcome::Errored);
    drop(directory);
}

#[test]
fn persist_failure_propagates_as_err() {
    let directory = tempfile::tempdir().unwrap();
    let blocked = directory.path().join("blocked.json");
    std::fs::write(&blocked, "").unwrap();
    let (_report_directory, path) = scored_report(|report| {
        report["testResults"][0]["assertionResults"][0]["meta"]["harness"]["run"]["artifacts"]["piSessionJsonl"] =
            json!("{\"type\":\"session\"}\n");
    });
    let observation = read_task_observation(&task(), &path, &blocked);
    assert!(observation.is_err());
}

/// A meta blob shaped like the scored fixture, parameterized for the meta
/// validation matrix below.
fn meta_with(mutate: impl FnOnce(&mut Value)) -> Value {
    let mut meta = json!({
        "eval": { "avgScore": 0.5, "scores": [{ "name": "StructuredOutputJudge", "score": 0.5 }], "thresholdFailed": false },
        "harness": {
            "name": "without_docs",
            "run": {
                "session": { "events": [{ "type": "message", "role": "user", "content": "prompt" }] },
                "usage": { "provider": "fixture", "model": "model", "inputTokens": 10, "outputTokens": 5, "totalTokens": 15, "toolCalls": 1, "metadata": {} },
                "artifacts": { "runId": "run-1" },
                "errors": []
            }
        }
    });
    mutate(&mut meta);
    meta
}

/// Drives a Vitest JSON report (single file, single assertion, no meta) and
/// asserts whether the reader accepts the shape.
fn report_shape(mutate: impl FnOnce(&mut Value)) -> Result<usize, String> {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vitest.json");
    let mut report = json!({
        "numFailedTests": 0,
        "numPassedTests": 1,
        "numPendingTests": 0,
        "numTodoTests": 0,
        "numTotalTests": 1,
        "startTime": 0,
        "success": true,
        "testResults": [
            {
                "message": "msg",
                "name": "/repo/example.docs.eval.ts",
                "status": "passed",
                "assertionResults": [
                    {
                        "ancestorTitles": ["Example workflow"],
                        "fullName": "Example workflow handles the case",
                        "status": "passed",
                        "title": "handles the case",
                        "failureMessages": [],
                    }
                ],
            }
        ],
    });
    mutate(&mut report);
    std::fs::write(&path, serde_json::to_string(&report).unwrap()).unwrap();
    match read_vitest_json_report_file(&path) {
        Ok(parsed) => Ok(parsed
            .test_results
            .iter()
            .map(|file| file.assertions.len())
            .sum()),
        Err(error) => Err(error.message().to_string()),
    }
}

#[test]
fn accepts_every_optional_report_field() {
    let accepted = report_shape(|report| {
        report["testResults"][0]["startTime"] = json!(0);
        report["testResults"][0]["endTime"] = json!(100);
        report["testResults"][0]["assertionResults"][0]["duration"] = json!(12);
        report["testResults"][0]["assertionResults"][0]["failureMessages"] = json!(["boom"]);
        report["testResults"][0]["assertionResults"][0]["location"] =
            json!({ "line": 1, "column": 2 });
        report["testResults"][0]["assertionResults"][0]["tags"] = json!(["tag"]);
        report["testResults"][0]["assertionResults"][0]["meta"] = json!(null);
    });
    assert_eq!(accepted, Ok(1));
}

#[test]
fn accepts_null_optional_report_fields_and_extra_keys() {
    let accepted = report_shape(|report| {
        report["testResults"][0]["startTime"] = json!(null);
        report["testResults"][0]["endTime"] = json!(null);
        report["testResults"][0]["extra"] = json!("passthrough");
        report["testResults"][0]["assertionResults"][0]["duration"] = json!(null);
        report["testResults"][0]["assertionResults"][0]["failureMessages"] = json!(null);
        report["testResults"][0]["assertionResults"][0]["location"] = json!(null);
        report["testResults"][0]["assertionResults"][0]["extra"] = json!(1);
    });
    assert_eq!(accepted, Ok(1));
}

#[test]
fn rejects_malformed_report_shapes() {
    let missing_success = |report: &mut Value| report["success"] = json!(1.5);
    let bad_top_level = |report: &mut Value| report["numTotalTests"] = json!("three");
    let bad_test_results = |report: &mut Value| report["testResults"] = json!("x");
    let bad_file_status =
        |report: &mut Value| report["testResults"][0]["status"] = json!("nonsense");
    let bad_file_time = |report: &mut Value| report["testResults"][0]["startTime"] = json!("x");
    let bad_assertions =
        |report: &mut Value| report["testResults"][0]["assertionResults"] = json!("x");
    let bad_ancestors = |report: &mut Value| {
        report["testResults"][0]["assertionResults"][0]["ancestorTitles"] = json!([1]);
    };
    let bad_status = |report: &mut Value| {
        report["testResults"][0]["assertionResults"][0]["status"] = json!("nonsense");
    };
    let bad_duration = |report: &mut Value| {
        report["testResults"][0]["assertionResults"][0]["duration"] = json!("fast");
    };
    let bad_failures = |report: &mut Value| {
        report["testResults"][0]["assertionResults"][0]["failureMessages"] = json!(["x", 2]);
    };
    let bad_location = |report: &mut Value| {
        report["testResults"][0]["assertionResults"][0]["location"] = json!({ "line": "1" });
    };
    let bad_tags =
        |report: &mut Value| report["testResults"][0]["assertionResults"][0]["tags"] = json!("x");
    let mutations = [
        missing_success,
        bad_top_level,
        bad_test_results,
        bad_file_status,
        bad_file_time,
        bad_assertions,
        bad_ancestors,
        bad_status,
        bad_duration,
        bad_failures,
        bad_location,
        bad_tags,
    ];
    for mutate in mutations {
        let result = report_shape(mutate);
        assert!(result.is_err(), "expected rejection");
        let Err(message) = result else {
            unreachable_ish()
        };
        assert!(
            message.starts_with("Failed to read eval result file"),
            "unexpected message: {message}"
        );
    }
}

fn unreachable_ish() -> ! {
    unreachable!("the branch above guarantees Err")
}

#[test]
fn meta_without_both_keys_or_a_non_object_is_ignored() {
    assert_eq!(read_eval_task_meta(None), None);
    assert_eq!(read_eval_task_meta(Some(&json!("x"))), None);
    assert_eq!(read_eval_task_meta(Some(&json!({ "other": 1 }))), None);
    assert_eq!(read_eval_task_meta(Some(&json!({ "eval": null }))), None);
}

#[test]
fn accepts_every_optional_harness_surface() {
    let accepted = meta_with(|meta| {
        meta["eval"]["output"] = json!({ "ok": true });
        meta["eval"]["toolCalls"] = json!([
            { "status": "pending", "name": "edit" },
            { "status": "ok", "name": "edit", "result": 1 },
            { "status": "error", "name": "edit", "error": { "message": "boom", "extra": [1] } }
        ]);
        meta["harness"]["name"] = json!("with_docs");
        meta["harness"]["run"]["output"] = json!(null);
        meta["harness"]["run"]["timings"] = json!({ "totalMs": 5, "metadata": { "unit": "ms" } });
        meta["harness"]["run"]["traces"] = json!([{ "name": "run" }]);
        meta["harness"]["run"]["usage"]["reasoningTokens"] = json!(1);
        meta["harness"]["run"]["usage"]["retries"] = json!(2);
        meta["harness"]["run"]["session"]["provider"] = json!("fixture");
        meta["harness"]["run"]["session"]["model"] = json!("model");
        meta["harness"]["run"]["session"]["metadata"] = json!({});
        meta["harness"]["run"]["session"]["events"] = json!([
            { "type": "message", "role": "system", "content": null, "metadata": {} },
            { "type": "tool_call", "id": "t1", "name": "edit", "arguments": {}, "startedAt": "0", "finishedAt": "1", "durationMs": 1, "metadata": {} },
            { "type": "tool_result", "toolCallId": "t1", "name": "edit", "content": "ok", "error": { "message": "boom", "type": "Tool" }, "startedAt": "0", "finishedAt": "1", "durationMs": 1, "metadata": {} }
        ]);
        meta["harness"]["run"]["errors"] = json!([{ "message": "boom", "extra": { "x": 1 } }]);
    });
    let parsed = read_eval_task_meta(Some(&accepted)).unwrap();
    assert_eq!(
        parsed.eval.as_ref().and_then(|eval| eval.avg_score),
        Some(0.5)
    );
    assert!(
        parsed
            .harness
            .as_ref()
            .and_then(|harness| harness.run.as_ref())
            .is_some()
    );
}

#[test]
fn harness_meta_without_a_run_or_eval_only_meta_both_parse() {
    let harness_only = read_eval_task_meta(Some(&json!({
        "eval": { "avgScore": 0.5 },
        "harness": { "name": "without_docs" }
    })));
    let parsed = harness_only.unwrap();
    assert!(
        parsed
            .harness
            .as_ref()
            .and_then(|harness| harness.run.as_ref())
            .is_none()
    );

    let eval_only = read_eval_task_meta(Some(&json!({ "eval": { "avgScore": null } })));
    let parsed = eval_only.unwrap();
    assert_eq!(parsed.eval.as_ref().and_then(|eval| eval.avg_score), None);
    assert!(parsed.harness.is_none());
}

#[expect(
    clippy::too_many_lines,
    reason = "the matrix is one exhaustive table of meta shapes; splitting it               would scatter the strict-schema arms the port must mirror"
)]
#[test]
fn rejects_every_remaining_meta_arm() {
    let mutations: Vec<MetaMutation> = vec![
        (
            "avgScore non-number",
            Box::new(|meta: &mut Value| meta["eval"]["avgScore"] = json!("x")),
        ),
        (
            "scores non-array",
            Box::new(|meta: &mut Value| meta["eval"]["scores"] = json!("x")),
        ),
        (
            "thresholdFailed non-bool",
            Box::new(|meta: &mut Value| meta["eval"]["thresholdFailed"] = json!("no")),
        ),
        (
            "toolCalls non-array",
            Box::new(|meta: &mut Value| meta["eval"]["toolCalls"] = json!("x")),
        ),
        (
            "toolCalls element non-object",
            Box::new(|meta: &mut Value| meta["eval"]["toolCalls"] = json!([1])),
        ),
        (
            "score entry name non-string",
            Box::new(|meta: &mut Value| {
                meta["eval"]["scores"] = json!([{ "score": 0.5, "name": 1 }]);
            }),
        ),
        (
            "harness name non-string",
            Box::new(|meta: &mut Value| meta["harness"]["name"] = json!(1)),
        ),
        (
            "run strict unknown key",
            Box::new(|meta: &mut Value| meta["harness"]["run"]["bogus"] = json!(1)),
        ),
        (
            "run missing session",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]
                    .as_object_mut()
                    .unwrap()
                    .remove("session");
            }),
        ),
        (
            "run missing usage",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]
                    .as_object_mut()
                    .unwrap()
                    .remove("usage");
            }),
        ),
        (
            "run missing errors",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]
                    .as_object_mut()
                    .unwrap()
                    .remove("errors");
            }),
        ),
        (
            "artifacts non-object",
            Box::new(|meta: &mut Value| meta["harness"]["run"]["artifacts"] = json!("x")),
        ),
        (
            "traces element non-object",
            Box::new(|meta: &mut Value| meta["harness"]["run"]["traces"] = json!([1])),
        ),
        (
            "usage provider non-string",
            Box::new(|meta: &mut Value| meta["harness"]["run"]["usage"]["provider"] = json!(1)),
        ),
        (
            "usage model non-string",
            Box::new(|meta: &mut Value| meta["harness"]["run"]["usage"]["model"] = json!(1)),
        ),
        (
            "usage outputTokens non-number",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["usage"]["outputTokens"] = json!("x");
            }),
        ),
        (
            "usage reasoningTokens non-number",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["usage"]["reasoningTokens"] = json!("x");
            }),
        ),
        (
            "usage retries non-number",
            Box::new(|meta: &mut Value| meta["harness"]["run"]["usage"]["retries"] = json!("x")),
        ),
        (
            "usage metadata non-object",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["usage"]["metadata"] = json!("x");
            }),
        ),
        (
            "timings totalMs non-number",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["timings"] = json!({ "totalMs": "fast" });
            }),
        ),
        (
            "timings metadata non-object",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["timings"] = json!({ "totalMs": 5, "metadata": "x" });
            }),
        ),
        (
            "session provider non-string",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["provider"] = json!(1);
            }),
        ),
        (
            "session model non-string",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["model"] = json!(1);
            }),
        ),
        (
            "session metadata non-object",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["metadata"] = json!("x");
            }),
        ),
        (
            "message event unknown key",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] =
                    json!([{ "type": "message", "role": "user", "bogus": 1 }]);
            }),
        ),
        (
            "message event missing role",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] = json!([{ "type": "message" }]);
            }),
        ),
        (
            "tool_call id non-string",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] =
                    json!([{ "type": "tool_call", "id": 1, "name": "edit" }]);
            }),
        ),
        (
            "tool_call arguments non-object",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] = json!([{
                    "type": "tool_call", "id": "t1", "name": "edit", "arguments": "x"
                }]);
            }),
        ),
        (
            "tool_call startedAt non-string",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] = json!([{
                    "type": "tool_call", "id": "t1", "name": "edit", "startedAt": 1
                }]);
            }),
        ),
        (
            "tool_call durationMs non-number",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] = json!([{
                    "type": "tool_call", "id": "t1", "name": "edit", "durationMs": "1"
                }]);
            }),
        ),
        (
            "tool_result name non-string",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] = json!([{
                    "type": "tool_result", "toolCallId": "t1", "name": 1
                }]);
            }),
        ),
        (
            "tool_result startedAt non-string",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] = json!([{
                    "type": "tool_result", "toolCallId": "t1", "startedAt": 1
                }]);
            }),
        ),
        (
            "tool_result metadata non-object",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] = json!([{
                    "type": "tool_result", "toolCallId": "t1", "metadata": "x"
                }]);
            }),
        ),
        (
            "event error type non-string",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] = json!([{
                    "type": "tool_result", "toolCallId": "t1",
                    "error": { "message": "boom", "type": 1 }
                }]);
            }),
        ),
    ];
    for (name, mutate) in mutations {
        let meta = meta_with(|blob| mutate(blob));
        assert!(
            read_eval_task_meta(Some(&meta)).is_none(),
            "expected {name} to be rejected"
        );
    }
}

/// One named meta-shape mutation, rejected by the strict upstream schema.
type MetaMutation = (&'static str, Box<dyn FnOnce(&mut Value)>);

#[test]
fn accepts_the_optional_meta_surface() {
    let accepted = meta_with(|meta| {
        meta["eval"]["output"] = json!({ "ok": true });
        meta["eval"]["toolCalls"] = json!([{ "status": "ok", "name": "edit" }]);
        meta["eval"]["scores"][0]["metadata"] = json!({ "why": "fixture" });
        meta["harness"]["run"]["traces"] = json!([{ "name": "run" }]);
        meta["harness"]["run"]["usage"]["reasoningTokens"] = json!(1);
        meta["harness"]["run"]["usage"]["retries"] = json!(2);
        meta["harness"]["run"]["session"]["provider"] = json!("fixture");
        meta["harness"]["run"]["session"]["model"] = json!("model");
        meta["harness"]["run"]["session"]["metadata"] = json!({});
    });
    let meta = read_eval_task_meta(Some(&accepted)).unwrap();
    assert_eq!(
        meta.eval.as_ref().and_then(|eval| eval.avg_score),
        Some(0.5)
    );
}

#[expect(
    clippy::too_many_lines,
    reason = "the matrix is one exhaustive table of meta-shape rejections;               splitting it would hide which arms still stand"
)]
#[test]
fn rejects_malformed_meta_shapes() {
    let mutations: Vec<MetaMutation> = vec![
        (
            "eval unknown key",
            Box::new(|meta: &mut Value| meta["eval"]["bogus"] = json!(1)),
        ),
        (
            "scores non-array",
            Box::new(|meta: &mut Value| meta["eval"]["scores"] = json!("x")),
        ),
        (
            "score entry missing score",
            Box::new(|meta: &mut Value| meta["eval"]["scores"] = json!([{ "name": "j" }])),
        ),
        (
            "score entry name non-string",
            Box::new(|meta: &mut Value| {
                meta["eval"]["scores"] = json!([{ "score": 0.5, "name": 1 }]);
            }),
        ),
        (
            "score entry metadata non-object",
            Box::new(|meta: &mut Value| {
                meta["eval"]["scores"] = json!([{ "score": 0.5, "metadata": "x" }]);
            }),
        ),
        (
            "thresholdFailed non-bool",
            Box::new(|meta: &mut Value| meta["eval"]["thresholdFailed"] = json!("no")),
        ),
        (
            "toolCalls non-array",
            Box::new(|meta: &mut Value| meta["eval"]["toolCalls"] = json!("x")),
        ),
        (
            "toolCalls element non-object",
            Box::new(|meta: &mut Value| meta["eval"]["toolCalls"] = json!([1])),
        ),
        (
            "harness name non-string",
            Box::new(|meta: &mut Value| meta["harness"]["name"] = json!(1)),
        ),
        (
            "run missing usage",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]
                    .as_object_mut()
                    .unwrap()
                    .remove("usage");
            }),
        ),
        (
            "run missing session",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]
                    .as_object_mut()
                    .unwrap()
                    .remove("session");
            }),
        ),
        (
            "run missing errors",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]
                    .as_object_mut()
                    .unwrap()
                    .remove("errors");
            }),
        ),
        (
            "errors element non-object",
            Box::new(|meta: &mut Value| meta["harness"]["run"]["errors"] = json!([1])),
        ),
        (
            "artifacts non-object",
            Box::new(|meta: &mut Value| meta["harness"]["run"]["artifacts"] = json!("x")),
        ),
        (
            "usage provider non-string",
            Box::new(|meta: &mut Value| meta["harness"]["run"]["usage"]["provider"] = json!(1)),
        ),
        (
            "usage metadata non-object",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["usage"]["metadata"] = json!("x");
            }),
        ),
        (
            "timings totalMs null",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["timings"] = json!({ "totalMs": null });
            }),
        ),
        (
            "session provider non-string",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["provider"] = json!(1);
            }),
        ),
        (
            "session events non-array",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] = json!("x");
            }),
        ),
        (
            "message role invalid",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] =
                    json!([{ "type": "message", "role": "tool" }]);
            }),
        ),
        (
            "tool_call missing name",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] = json!([{ "type": "tool_call" }]);
            }),
        ),
        (
            "tool_result missing toolCallId",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] = json!([{ "type": "tool_result" }]);
            }),
        ),
        (
            "event error message missing",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] =
                    json!([{ "type": "tool_result", "toolCallId": "t1", "error": {} }]);
            }),
        ),
        (
            "event error type non-string",
            Box::new(|meta: &mut Value| {
                meta["harness"]["run"]["session"]["events"] = json!([{
                    "type": "tool_result",
                    "toolCallId": "t1",
                    "error": { "message": "boom", "type": 1 }
                }]);
            }),
        ),
    ];
    for (name, mutate) in mutations {
        let meta = meta_with(|blob| mutate(blob));
        assert!(
            read_eval_task_meta(Some(&meta)).is_none(),
            "expected {name} to be rejected"
        );
    }
}

#[test]
fn harness_only_meta_falls_back_to_status_scores() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vitest.json");
    let report = json!({
        "numFailedTests": 0,
        "numPassedTests": 2,
        "numPendingTests": 0,
        "numTodoTests": 0,
        "numTotalTests": 2,
        "startTime": 0,
        "success": true,
        "testResults": [
            {
                "message": "",
                "name": "/repo/example.docs.eval.ts",
                "status": "passed",
                "assertionResults": [
                    {
                        "fullName": "Set > one",
                        "status": "passed",
                        "title": "one",
                        "meta": { "harness": { "name": "without_docs" } }
                    },
                    {
                        "fullName": "Set > two",
                        "status": "failed",
                        "title": "two",
                        "meta": { "harness": { "name": "with_docs" } }
                    }
                ],
            }
        ],
    });
    std::fs::write(&path, serde_json::to_string(&report).unwrap()).unwrap();
    let cases = collect_report_workspace(&read_vitest_json_report_file(&path).unwrap());
    assert_eq!(cases.len(), 2);
    assert_eq!(
        cases[0].eval.as_ref().and_then(|eval| eval.avg_score),
        Some(1.0)
    );
    assert_eq!(
        cases[1].eval.as_ref().and_then(|eval| eval.avg_score),
        Some(0.0)
    );
}
