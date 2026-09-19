//! Port of `packages/evals/test/plan.test.ts` — the 4 portable planning
//! cases, 1:1 against upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use pi_evals::EvalError;
use pi_evals::plan::{
    DiscoveredEvalCase, DocumentationVariant, create_task_plan, parse_discovered_cases,
};
use serde_json::json;

fn error_contains<T>(result: Result<T, EvalError>, needle: &str) -> bool {
    result
        .err()
        .is_some_and(|error| error.message().contains(needle))
}

fn discovered() -> serde_json::Value {
    json!([{ "name": "Add model > adds the model", "file": "evals/models.docs.eval.ts" }])
}

fn case() -> DiscoveredEvalCase {
    DiscoveredEvalCase {
        file: String::from("evals/models.docs.eval.ts"),
        full_name: String::from("Add model > adds the model"),
        eval_set: String::from("Add model"),
        case_id: String::from("adds the model"),
    }
}

fn task(variant: DocumentationVariant, run_number: u32) -> pi_evals::plan::EvalTask {
    pi_evals::plan::EvalTask {
        file: String::from("evals/models.docs.eval.ts"),
        full_name: String::from("Add model > adds the model"),
        eval_set: String::from("Add model"),
        case_id: String::from("adds the model"),
        variant,
        model: String::from("fixture/model"),
        run_number,
    }
}

#[test]
fn derives_stable_case_identity_from_ordinary_vitest_names() {
    assert_eq!(parse_discovered_cases(&discovered()), Ok(vec![case()]));
}

#[test]
fn rejects_ambiguous_and_duplicate_identities() {
    assert!(error_contains(
        parse_discovered_cases(&json!([{ "name": "adds the model", "file": "model.ts" }])),
        "<eval set> > <case>"
    ));
    let duplicated = json!([
        { "name": "Add model > adds the model", "file": "evals/models.docs.eval.ts" },
        { "name": "Add model > adds the model", "file": "evals/models.docs.eval.ts" }
    ]);
    assert!(error_contains(
        parse_discovered_cases(&duplicated),
        "Duplicate eval case identity"
    ));
}

#[test]
fn creates_one_isolated_task_per_case_variant_model_and_repetition() {
    let cases = parse_discovered_cases(&discovered());
    let planned = cases.and_then(|cases| create_task_plan(&cases, "fixture/model", 2));
    assert_eq!(
        planned,
        Ok(vec![
            task(DocumentationVariant::WithoutDocs, 1),
            task(DocumentationVariant::WithDocs, 1),
            task(DocumentationVariant::WithDocs, 2),
            task(DocumentationVariant::WithoutDocs, 2),
        ])
    );
}

#[test]
fn rejects_non_array_input() {
    assert!(error_contains(
        parse_discovered_cases(&json!("not an array")),
        "Discovered eval cases must be an array"
    ));
}

#[test]
fn rejects_items_without_a_string_name_and_file() {
    assert!(error_contains(
        parse_discovered_cases(&json!([{ "name": "Add model > adds the model" }])),
        "Discovered eval case is invalid"
    ));
}

#[test]
fn rejects_invalid_model_identities_and_repetitions() {
    let cases = parse_discovered_cases(&discovered());
    assert!(error_contains(
        cases
            .clone()
            .and_then(|cases| create_task_plan(&cases, "model", 1)),
        "provider and model"
    ));
    assert!(error_contains(
        cases.and_then(|cases| create_task_plan(&cases, "fixture/model", 0)),
        "positive integer"
    ));
}
