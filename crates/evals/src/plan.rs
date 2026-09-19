//! Eval-case discovery and task planning, ported from
//! `packages/evals/src/plan.ts` (zero imports upstream; zero deps here).

use serde_json::Value;

use crate::EvalError;

/// The two documentation variants of a docs-lift experiment, in upstream's
/// declaration order.
pub const DOCUMENTATION_VARIANTS: [DocumentationVariant; 2] = [
    DocumentationVariant::WithoutDocs,
    DocumentationVariant::WithDocs,
];

/// Which documentation set a task runs against: the bundled docs (treatment)
/// or an install without them (control).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DocumentationVariant {
    /// Control arm: the installed runtime has no bundled docs.
    WithoutDocs,
    /// Treatment arm: the installed runtime ships the bundled docs.
    WithDocs,
}

impl DocumentationVariant {
    /// The wire/report spelling used by upstream plans and reports.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::WithoutDocs => "without_docs",
            Self::WithDocs => "with_docs",
        }
    }
}

impl std::fmt::Display for DocumentationVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One eval case discovered in a Vitest eval suite, keyed by its
/// `"<eval set> > <case>"` full name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscoveredEvalCase {
    /// Eval file the case was discovered in, as the discovery pass saw it.
    pub file: String,
    /// The full Vitest test name, `"<eval set> > <case>"`.
    pub full_name: String,
    /// The eval set segment of the full name.
    pub eval_set: String,
    /// The case segment of the full name.
    pub case_id: String,
}

/// One planned run of one case under one variant, model, and repetition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalTask {
    /// Eval file the case was discovered in.
    pub file: String,
    /// The full Vitest test name.
    pub full_name: String,
    /// The eval set segment of the full name.
    pub eval_set: String,
    /// The case segment of the full name.
    pub case_id: String,
    /// Which documentation variant this run executes.
    pub variant: DocumentationVariant,
    /// Model identity, `"<provider>/<model>"`.
    pub model: String,
    /// One-based repetition number; variant order alternates on it.
    pub run_number: u32,
}

/// Derives eval-case identities from discovered Vitest cases.
///
/// `value` is the discovery payload: an array of records each carrying
/// string `name` and `file` fields, where `name` must be exactly
/// `"<eval set> > <case>"` and each `(eval set, case)` pair must be unique.
///
/// # Errors
///
/// Upstream throws `TypeError` when `value` is not an array, when an item
/// lacks a string `name` or `file`, when a name is not exactly two
/// `"<eval set> > <case>"` segments (a third segment is rejected, and empty
/// segments are rejected after trimming), or when a case identity repeats.
pub fn parse_discovered_cases(value: &Value) -> Result<Vec<DiscoveredEvalCase>, EvalError> {
    let items = value
        .as_array()
        .ok_or_else(|| EvalError::new("Discovered eval cases must be an array."))?;
    let mut identities = std::collections::HashSet::new();
    let mut cases = Vec::with_capacity(items.len());
    for item in items {
        let name = item
            .as_object()
            .and_then(|item| item.get("name"))
            .and_then(Value::as_str);
        let file = item
            .as_object()
            .and_then(|item| item.get("file"))
            .and_then(Value::as_str);
        let (Some(name), Some(file)) = (name, file) else {
            return Err(EvalError::new("Discovered eval case is invalid."));
        };
        let mut segments = name.split(" > ");
        let eval_set = segments.next().unwrap_or_default();
        let case_id = segments.next();
        let extra_segments = segments.count();
        let valid = !eval_set.trim().is_empty()
            && case_id.is_some_and(|case_id| !case_id.trim().is_empty())
            && extra_segments == 0;
        if !valid {
            return Err(EvalError::new(format!(
                "Documentation eval must use \"<eval set> > <case>\": {name}"
            )));
        }
        let identity = serde_json::to_string(&[eval_set, case_id.unwrap_or_default()])
            .map_err(|error| EvalError::new(error.to_string()))?;
        if !identities.insert(identity) {
            return Err(EvalError::new(format!(
                "Duplicate eval case identity: {name}"
            )));
        }
        cases.push(DiscoveredEvalCase {
            file: file.to_string(),
            full_name: name.to_string(),
            eval_set: eval_set.to_string(),
            case_id: case_id.unwrap_or_default().to_string(),
        });
    }
    Ok(cases)
}

/// Expands discovered cases into `(case, variant, model, repetition)` tasks.
///
/// Variant order alternates per repetition — odd repetitions run control
/// first, even repetitions run treatment first — so variant effects are not
/// confounded with run order.
///
/// `runs_per_variant` is a `u32`, which covers upstream's
/// `Number.isSafeInteger` check by construction.
///
/// # Errors
///
/// Upstream throws `TypeError` when `model` has no `/` separator, starts or
/// ends with one, or when `runs_per_variant` is below one.
pub fn create_task_plan(
    cases: &[DiscoveredEvalCase],
    model: &str,
    runs_per_variant: u32,
) -> Result<Vec<EvalTask>, EvalError> {
    if !model.contains('/') || model.starts_with('/') || model.ends_with('/') {
        return Err(EvalError::new(
            "Model identity must contain a provider and model.",
        ));
    }
    if runs_per_variant < 1 {
        return Err(EvalError::new(
            "Runs per variant must be a positive integer.",
        ));
    }
    let mut tasks = Vec::new();
    for eval_case in cases {
        for run_number in 1..=runs_per_variant {
            let variants = if run_number % 2 == 1 {
                [
                    DocumentationVariant::WithoutDocs,
                    DocumentationVariant::WithDocs,
                ]
            } else {
                [
                    DocumentationVariant::WithDocs,
                    DocumentationVariant::WithoutDocs,
                ]
            };
            for variant in variants {
                tasks.push(EvalTask {
                    file: eval_case.file.clone(),
                    full_name: eval_case.full_name.clone(),
                    eval_set: eval_case.eval_set.clone(),
                    case_id: eval_case.case_id.clone(),
                    variant,
                    model: model.to_string(),
                    run_number,
                });
            }
        }
    }
    Ok(tasks)
}
