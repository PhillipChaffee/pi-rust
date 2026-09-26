//! The fork-content policy, ported from upstream
//! `src/harness/session/fork-policy.ts`: the branch-ancestry walk and the
//! reserved-namespace projection every fork backend shares.

use serde_json::Value as JsonValue;

use crate::harness::session::types::{ForkOptions, ForkPosition, SessionError};

/// Where a branch fork's copied values land, upstream's
/// `ForkCurrentStatePlan`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ForkCurrentStatePlan {
    /// Copy the named branch's path; its values project under the copied
    /// branch name with the destination tip, wire `"scope": "branch"`.
    Branch {
        /// The source Branch name copied under.
        branch: String,
        /// The destination branch tip after the copy; `None` at the branch
        /// root, upstream's `destinationTip`.
        destination_tip: Option<String>,
    },
    /// Copy the whole tree, wire `"scope": "tree"`.
    Tree,
}

/// The ancestry walk one branch fork reads, upstream's `selectBranchFork`
/// source object.
///
/// The tip is `None` for an unknown branch and `Some(None)` for a branch
/// whose tip is the root; the parent lookup is `None` for a missing
/// entry; the selector feeds one copied path entry.
pub struct BranchForkSource<'a> {
    /// The source branch's tip.
    pub tip: Option<Option<String>>,
    /// The parent of one entry.
    pub get_parent: &'a dyn Fn(&str) -> Option<Option<String>>,
    /// The selector one copied path entry feeds.
    pub select_entry: &'a mut dyn FnMut(&str),
}

impl std::fmt::Debug for BranchForkSource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BranchForkSource")
            .field("tip", &self.tip)
            .finish_non_exhaustive()
    }
}

/// Walks the source branch's ancestry and selects the copied path, upstream's
/// `selectBranchFork`.
///
/// # Errors
/// A `SessionError::Message` for an unknown source branch, a corrupt
/// ancestry, or a requested entry off the branch.
pub fn select_branch_fork(
    options: &ForkOptions,
    source: &mut BranchForkSource<'_>,
) -> Result<ForkCurrentStatePlan, SessionError> {
    let ForkOptions::Branch {
        branch,
        entry_id,
        position,
        ..
    } = options
    else {
        return Ok(ForkCurrentStatePlan::Tree);
    };
    let Some(tip) = &source.tip else {
        return Err(SessionError::Message(format!(
            "Unknown source branch: {branch}"
        )));
    };
    let requested = entry_id.clone().or_else(|| tip.clone());
    let mut found = requested.is_none();
    let mut destination_tip: Option<String> = None;
    let mut entry_id = tip.clone();
    while let Some(current) = entry_id {
        let Some(parent_id) = (source.get_parent)(&current) else {
            return Err(SessionError::Message(format!(
                "Corrupt source branch: missing parent {current}"
            )));
        };
        if Some(&current) == requested.as_ref() {
            found = true;
            destination_tip = if *position == Some(ForkPosition::Before) {
                parent_id.clone()
            } else {
                Some(current.clone())
            };
            if *position != Some(ForkPosition::Before) {
                (source.select_entry)(&current);
            }
        } else if found {
            (source.select_entry)(&current);
        }
        entry_id = parent_id;
    }
    if !found {
        return Err(SessionError::Message(format!(
            "Fork entry {} is not on source branch {branch:?}",
            requested.as_deref().unwrap_or("null")
        )));
    }
    Ok(ForkCurrentStatePlan::Branch {
        branch: branch.clone(),
        destination_tip,
    })
}

/// One surviving current-state write projected into destination state,
/// upstream's `CommittedValueSetWrite | CommittedListAppendWrite` input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ForkStateWrite {
    /// A surviving scalar value.
    ValueSet {
        /// The source sequence.
        seq: u64,
        /// The addressed namespace.
        namespace: String,
        /// The addressed key.
        key: String,
        /// The stored value.
        value: JsonValue,
    },
    /// A surviving list element.
    ListAppend {
        /// The source sequence.
        seq: u64,
        /// The addressed namespace.
        namespace: String,
        /// The addressed key.
        key: String,
        /// The stored element.
        value: JsonValue,
    },
}

/// Projects one current scalar row or surviving list element into
/// destination state, upstream's `projectForkCurrentStateWrite`.
///
/// Returns `Ok(None)` when the write does not survive the fork scope.
///
/// # Errors
/// A `SessionError::Message` when a surviving write sits in an unknown
/// reserved `pi.` namespace.
pub fn project_fork_current_state_write(
    write: &ForkStateWrite,
    plan: &ForkCurrentStatePlan,
    is_entry_copied: &dyn Fn(&str) -> bool,
) -> Result<Option<ForkStateWrite>, SessionError> {
    let (namespace, key) = match write {
        ForkStateWrite::ValueSet { namespace, key, .. }
        | ForkStateWrite::ListAppend { namespace, key, .. } => (namespace.as_str(), key.as_str()),
    };
    match namespace {
        "pi.session.name" => return Ok(Some(write.clone())),
        "pi.entry.label" => {
            return Ok(is_entry_copied(key).then(|| write.clone()));
        }
        "pi.branch.tip" => match plan {
            ForkCurrentStatePlan::Tree => return Ok(Some(write.clone())),
            ForkCurrentStatePlan::Branch {
                branch,
                destination_tip,
            } => {
                return Ok((key == branch).then(|| {
                    let mut projected = write.clone();
                    match &mut projected {
                        ForkStateWrite::ValueSet { value, .. } => {
                            *value = destination_tip
                                .clone()
                                .map_or(JsonValue::Null, serde_json::Value::String);
                        }
                        ForkStateWrite::ListAppend { .. } => {}
                    }
                    projected
                }));
            }
        },
        "pi.lane.config" => {
            return Ok(
                (matches!(plan, ForkCurrentStatePlan::Tree) || key == plan_branch(plan))
                    .then(|| write.clone()),
            );
        }
        "pi.lane.state" => {
            return Ok(
                if matches!(plan, ForkCurrentStatePlan::Tree) || key == plan_branch(plan) {
                    let mut projected = write.clone();
                    match &mut projected {
                        ForkStateWrite::ValueSet { value, .. }
                        | ForkStateWrite::ListAppend { value, .. } => {
                            *value = IDLE_LANE_STATE_JSON.clone();
                        }
                    }
                    Some(projected)
                } else {
                    None
                },
            );
        }
        "pi.result" => return Ok(None),
        _ => {}
    }
    if namespace.starts_with("pi.op.") || namespace.starts_with("pi.pending.") {
        return Ok(None);
    }
    if namespace == "pi" || namespace.starts_with("pi.") {
        return Err(SessionError::Message(format!(
            "Unknown reserved fork namespace: {namespace}"
        )));
    }
    Ok(matches!(plan, ForkCurrentStatePlan::Tree).then(|| write.clone()))
}

fn plan_branch(plan: &ForkCurrentStatePlan) -> &str {
    match plan {
        ForkCurrentStatePlan::Branch { branch, .. } => branch,
        ForkCurrentStatePlan::Tree => "",
    }
}

/// The idle lane state the fork writes under `pi.lane.state`, upstream's
/// `{ currentOperationId: null, lastOperationId: null, inbox: [] }`.
static IDLE_LANE_STATE_JSON: std::sync::LazyLock<JsonValue> = std::sync::LazyLock::new(|| {
    serde_json::json!({
        "currentOperationId": serde_json::Value::Null,
        "lastOperationId": serde_json::Value::Null,
        "inbox": [],
    })
});

#[cfg(test)]
mod tests;
