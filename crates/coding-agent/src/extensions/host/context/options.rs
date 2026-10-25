//! The two context values the domain types do not serialize themselves:
//! the scoped-model list and the system-prompt options ([ADR 0010]).
//!
//! [ADR 0010]: ../../../../../../docs/adr/0010-extension-host-wire-protocol.md

use serde_json::{Map, Value, json};

use super::super::payload::descriptors::source_info_to_wire;
use super::HostCallError;

/// Maps the scoped models upstream's `scopedModels` carries.
pub(super) fn scoped_models_to_wire(
    models: &[crate::extensions::reasons::ScopedModel],
) -> Result<Value, HostCallError> {
    let mut wire = Vec::with_capacity(models.len());
    for scoped in models {
        wire.push(json!({
            "model": model_to_wire(&scoped.model)?,
            "thinkingLevel": scoped.thinking_level,
        }));
    }
    Ok(Value::Array(wire))
}

/// Maps the system-prompt options upstream's `getSystemPromptOptions`
/// returns.
pub(super) fn system_prompt_options_to_wire(
    options: &crate::system_prompt::BuildSystemPromptOptions,
) -> Value {
    let mut wire = Map::new();
    if let Some(custom_prompt) = &options.custom_prompt {
        wire.insert("customPrompt".to_string(), json!(custom_prompt));
    }
    if let Some(selected_tools) = &options.selected_tools {
        wire.insert("selectedTools".to_string(), json!(selected_tools));
    }
    if let Some(tool_snippets) = &options.tool_snippets {
        wire.insert("toolSnippets".to_string(), json!(tool_snippets));
    }
    if let Some(guidelines) = &options.prompt_guidelines {
        wire.insert("promptGuidelines".to_string(), json!(guidelines));
    }
    if let Some(append) = &options.append_system_prompt {
        wire.insert("appendSystemPrompt".to_string(), json!(append));
    }
    wire.insert("cwd".to_string(), json!(options.cwd));
    if let Some(context_files) = &options.context_files {
        let files = context_files
            .iter()
            .map(|file| json!({ "path": file.path, "content": file.content }))
            .collect();
        wire.insert("contextFiles".to_string(), Value::Array(files));
    }
    if let Some(skills) = &options.skills {
        let skills = skills.iter().map(skill_to_wire).collect::<Vec<_>>();
        wire.insert("skills".to_string(), Value::Array(skills));
    }
    Value::Object(wire)
}

fn model_to_wire(model: &pi_ai::types::Model) -> Result<Value, HostCallError> {
    serde_json::to_value(model)
        .map_err(|error| HostCallError::new("invalid_params", error.to_string()))
}

fn skill_to_wire(skill: &crate::skills::Skill) -> Value {
    json!({
        "name": skill.name,
        "description": skill.description,
        "filePath": skill.file_path,
        "baseDir": skill.base_dir,
        "sourceInfo": source_info_to_wire(&skill.source_info),
        "disableModelInvocation": skill.disable_model_invocation,
    })
}
