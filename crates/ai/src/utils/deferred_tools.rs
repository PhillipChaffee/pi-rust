//! Deferred tool splitting, ported from
//! `packages/ai/src/utils/deferred-tools.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Providers with native deferred tool loading keep not-yet-used tool
//! definitions out of the request until a tool result marks them loaded
//! (`addedToolNames`). [`split_deferred_tools`] divides the current tool set
//! into the immediate definitions the request carries and the deferred ones
//! waiting for their marker.

use crate::types::{AssistantBlock, Context, Message, Tool};

/// A tool-name normalizer, upstream's `ToolNameNormalizer`; OAuth providers
/// canonicalize tool names through it.
pub type ToolNameNormalizer<'a> = dyn Fn(&str) -> String + 'a;

/// Split the context's tools into immediate and deferred definitions.
///
/// Deduplication keys on the normalized name, last definition wins. A tool
/// is deferred when a tool result marked it loaded and no earlier assistant
/// message has called it.
#[must_use]
pub fn split_deferred_tools(
    context: &Context,
    enabled: bool,
    normalize_name: &ToolNameNormalizer<'_>,
) -> SplitDeferredTools {
    let mut unique_tools: Vec<(String, Tool)> = Vec::new();
    if let Some(tools) = &context.tools {
        for tool in tools {
            let normalized = normalize_name(&tool.name);
            if let Some(slot) = unique_tools
                .iter_mut()
                .find(|(name, _)| *name == normalized)
            {
                slot.1 = tool.clone();
            } else {
                unique_tools.push((normalized, tool.clone()));
            }
        }
    }
    if !enabled {
        return SplitDeferredTools {
            immediate: unique_tools.into_iter().map(|(_, tool)| tool).collect(),
            deferred: Vec::new(),
        };
    }

    let mut deferred_names = std::collections::BTreeSet::new();
    let mut used_names = std::collections::BTreeSet::new();
    for message in &context.messages {
        match message {
            Message::Assistant(assistant) => {
                for block in &assistant.content {
                    if let AssistantBlock::ToolCall(tool_call) = block {
                        used_names.insert(normalize_name(&tool_call.name));
                    }
                }
            }
            Message::ToolResult(result) => {
                for name in result.added_tool_names.iter().flatten() {
                    let normalized = normalize_name(name);
                    if !used_names.contains(&normalized) {
                        deferred_names.insert(normalized);
                    }
                }
            }
            Message::User(_) => {}
        }
    }

    let mut immediate = Vec::new();
    let mut deferred = Vec::new();
    for (name, tool) in unique_tools {
        if deferred_names.contains(&name) {
            deferred.push((name, tool));
        } else {
            immediate.push(tool);
        }
    }
    SplitDeferredTools {
        immediate,
        deferred,
    }
}

/// The split tool set: definitions the request carries immediately and the
/// deferred definitions waiting for their load marker, both in the context's
/// definition order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SplitDeferredTools {
    /// Definitions the request carries up front.
    pub immediate: Vec<Tool>,
    /// Definitions a later tool result loads, keyed by normalized name.
    pub deferred: Vec<(String, Tool)>,
}
