//! The trust leg: TS pi's `trust.json` carried into the Rust pi's trust
//! store.
//!
//! The store is a `Record<absolute-path, bool|null>` of per-directory trust
//! decisions, the same map behind the same path the trust-manager port
//! reads. The values are the whole contract: anything but `true`, `false`,
//! or `null` fails the leg, and the keys ride verbatim (the store re-sorts
//! on its own writes).

use std::path::Path;

use pi_coding_agent::utils::text::strip_bom;
use serde_json::Value;

use crate::discovery::Discovery;
use crate::plan::{OpTarget, PlannedOp};
use crate::report::{ImportReport, TrustItem, TrustOutcome};

/// Run the trust leg against the discovery, appending the item and
/// planning the cross-dir copy.
pub fn run_trust(discovery: &Discovery, report: &mut ImportReport) -> Vec<PlannedOp> {
    let mut ops = Vec::new();
    let source_path = discovery.source.join("trust.json");
    if !source_path.exists() {
        return ops;
    }
    let Some(entries) = validate(&source_path, report) else {
        return ops;
    };
    let target_path = discovery.target.join("trust.json");
    let outcome = if discovery.in_place {
        TrustOutcome::Unchanged { entries }
    } else if target_path.exists() {
        TrustOutcome::Skipped {
            reason: "target already has trust.json".to_string(),
        }
    } else {
        ops.push(PlannedOp::CopyFile {
            from: source_path.clone(),
            to: target_path.clone(),
            targets: vec![OpTarget::Trust],
        });
        TrustOutcome::Written { entries }
    };
    report.trust = Some(TrustItem {
        path: if outcome_is_written(&outcome) {
            target_path.display().to_string()
        } else {
            source_path.display().to_string()
        },
        outcome,
    });
    ops
}

/// Whether the outcome names a write, the report's path-form rule.
const fn outcome_is_written(outcome: &TrustOutcome) -> bool {
    matches!(outcome, TrustOutcome::Written { .. })
}

/// The store's validation: a JSON object whose values are all `true`,
/// `false`, or `null`; the entry count returns, the failures record.
fn validate(source_path: &Path, report: &mut ImportReport) -> Option<u64> {
    let content = match std::fs::read_to_string(source_path) {
        Ok(content) => content,
        Err(error) => {
            let reason = error.to_string();
            report.trust = Some(TrustItem {
                path: source_path.display().to_string(),
                outcome: TrustOutcome::Failed {
                    reason: reason.clone(),
                },
            });
            report.push_failure(source_path.display().to_string(), reason);
            return None;
        }
    };
    let value: Value = match serde_json::from_str(strip_bom(&content)) {
        Ok(value) => value,
        Err(error) => {
            let reason = format!("trust.json is unreadable: {error}");
            report.trust = Some(TrustItem {
                path: source_path.display().to_string(),
                outcome: TrustOutcome::Failed {
                    reason: reason.clone(),
                },
            });
            report.push_failure(source_path.display().to_string(), reason);
            return None;
        }
    };
    let Some(map) = value.as_object() else {
        let reason = "trust.json is not a JSON object".to_string();
        report.trust = Some(TrustItem {
            path: source_path.display().to_string(),
            outcome: TrustOutcome::Failed {
                reason: reason.clone(),
            },
        });
        report.push_failure(source_path.display().to_string(), reason);
        return None;
    };
    for (key, value) in map {
        if !value.is_boolean() && !value.is_null() {
            let reason = format!("trust.json value for {key} must be true, false, or null");
            report.trust = Some(TrustItem {
                path: source_path.display().to_string(),
                outcome: TrustOutcome::Failed {
                    reason: reason.clone(),
                },
            });
            report.push_failure(source_path.display().to_string(), reason);
            return None;
        }
    }
    Some(map.len() as u64)
}
