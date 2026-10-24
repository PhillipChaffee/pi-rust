//! The descriptor wire mappings: the declarative [`ToolDefinition`] surface
//! a `register_tool` frame carries (the executor and argument shim are
//! wire calls, values never cross), the command and flag registration
//! descriptors, and the provider registration's data fields (the stream,
//! refresh, and OAuth closures stay host-side; ADR 0007 reserves native
//! `Provider` objects to compile-ins). The `ExtensionContext` snapshot
//! lives here too, upstream's camelCase.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::extensions::types::{RenderShell, ToolDefinition};
use pi_agent_core::types::ToolExecutionMode;

use super::{
    WireError, fields, optional_bool, optional_string, optional_string_array, payload_error,
    required_bool, required_string, required_u64, required_value, serde_from_wire, serde_to_wire,
};

/// The declarative tool body a `register_tool` frame carries, upstream's
/// `ToolDefinition` without the function members.
///
/// The wire cannot carry closures, so [`ToolDeclaration`] holds the fields
/// the registering side announces; the executor and the argument shim ride
/// #170's call scope instead.
#[derive(Clone, Debug, PartialEq)]
#[expect(
    clippy::derive_partial_eq_without_eq,
    reason = "the parameters member rides serde_json::Value, which is PartialEq-only"
)]
pub struct ToolDeclaration {
    /// Tool name (used in LLM tool calls).
    pub name: String,
    /// Human-readable label for UI.
    pub label: String,
    /// Description for LLM.
    pub description: String,
    /// Optional one-line snippet for the Available tools system-prompt
    /// section; custom tools are omitted from that section when absent.
    pub prompt_snippet: Option<String>,
    /// Optional guideline bullets appended to the default system prompt
    /// Guidelines section when this tool is active.
    pub prompt_guidelines: Option<Vec<String>>,
    /// Parameter schema upstream's TypeBox emits.
    pub parameters: Value,
    /// Optional provider-side constrained sampling request; `false` and the
    /// full config both ride the wire untagged.
    pub constrained_sampling: Option<pi_ai::types::ConstrainedSamplingSetting>,
    /// Whether the standard tool row shell renders or the tool renders its
    /// own framing.
    pub render_shell: Option<RenderShell>,
    /// Per-tool execution mode override; absent, the default applies.
    pub execution_mode: Option<ToolExecutionMode>,
}

/// Maps the declarative surface of a domain tool into its `register_tool`
/// value.
///
/// The [`ToolDefinition`] executor and argument shim stay out: on the wire
/// they are calls the host opens, not values it stores.
///
/// # Errors
///
/// Returns [`WireError`] when the parameter schema or the constrained
/// sampling setting stops short of the wire's JSON.
pub fn tool_definition_to_wire(definition: &ToolDefinition) -> Result<Value, WireError> {
    declaration_to_wire(&ToolDeclaration {
        name: definition.name.clone(),
        label: definition.label.clone(),
        description: definition.description.clone(),
        prompt_snippet: definition.prompt_snippet.clone(),
        prompt_guidelines: definition.prompt_guidelines.clone(),
        parameters: definition.parameters.clone(),
        constrained_sampling: definition.constrained_sampling.clone(),
        render_shell: definition.render_shell,
        execution_mode: definition.execution_mode,
    })
}

/// Reads the declarative tool body of a `register_tool` value.
///
/// # Errors
///
/// Returns [`WireError`] when a required field is missing or mistyped.
pub fn tool_declaration_from_wire(tool: &Value) -> Result<ToolDeclaration, WireError> {
    let fields = fields(tool, "register_tool")?;
    let prompt_snippet = optional_string(fields, "promptSnippet", "register_tool")?;
    let prompt_guidelines = optional_string_array(fields, "promptGuidelines", "register_tool")?;
    let constrained_sampling = match fields.get("constrainedSampling") {
        None | Some(Value::Null) => None,
        Some(value) => Some(serde_from_wire(value.clone(), "constrainedSampling")?),
    };
    let render_shell = match fields.get("renderShell") {
        None | Some(Value::Null) => None,
        Some(shell) => match shell.as_str() {
            Some("default") => Some(RenderShell::Default),
            Some("self") => Some(RenderShell::SelfRender),
            _ => {
                return Err(payload_error(
                    "register_tool's renderShell must be \"default\" or \"self\"",
                ));
            }
        },
    };
    let execution_mode = match fields.get("executionMode") {
        None | Some(Value::Null) => None,
        Some(mode) => Some(serde_from_wire(mode.clone(), "executionMode")?),
    };
    Ok(ToolDeclaration {
        name: required_string(fields, "name", "register_tool")?,
        label: required_string(fields, "label", "register_tool")?,
        description: required_string(fields, "description", "register_tool")?,
        prompt_snippet,
        prompt_guidelines,
        parameters: required_value(fields, "parameters", "register_tool")?.clone(),
        constrained_sampling,
        render_shell,
        execution_mode,
    })
}

fn declaration_to_wire(declaration: &ToolDeclaration) -> Result<Value, WireError> {
    let mut wire = json!({
        "name": declaration.name,
        "label": declaration.label,
        "description": declaration.description,
        "parameters": declaration.parameters,
    });
    if let Some(prompt_snippet) = &declaration.prompt_snippet {
        wire["promptSnippet"] = json!(prompt_snippet);
    }
    if let Some(prompt_guidelines) = &declaration.prompt_guidelines {
        wire["promptGuidelines"] = json!(prompt_guidelines);
    }
    if let Some(constrained_sampling) = &declaration.constrained_sampling {
        wire["constrainedSampling"] = serde_to_wire(constrained_sampling, "constrainedSampling")?;
    }
    if let Some(render_shell) = declaration.render_shell {
        wire["renderShell"] = json!(match render_shell {
            RenderShell::Default => "default",
            RenderShell::SelfRender => "self",
        });
    }
    if let Some(execution_mode) = declaration.execution_mode {
        wire["executionMode"] = serde_to_wire(&execution_mode, "executionMode")?;
    }
    Ok(wire)
}

/// One registered command's descriptor, upstream's `RegisteredCommand`
/// without the handlers: the child announces the name and description, the
/// host stamps provenance and derives the invocation name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandDeclaration {
    /// The registered name.
    pub name: String,
    /// The command description.
    pub description: Option<String>,
}

/// The registration descriptor a `register_command` frame carries.
///
/// # Errors
///
/// Returns [`WireError`] when the name is missing or mistyped.
pub fn command_to_wire(command: &CommandDeclaration) -> Result<Value, WireError> {
    let mut wire = json!({ "name": command.name });
    if let Some(description) = &command.description {
        wire["description"] = json!(description);
    }
    Ok(wire)
}

/// Reads the command descriptor of a `register_command` value.
///
/// # Errors
///
/// Returns [`WireError`] when the name is missing or mistyped.
pub fn command_declaration_from_wire(command: &Value) -> Result<CommandDeclaration, WireError> {
    let fields = fields(command, "register_command")?;
    Ok(CommandDeclaration {
        name: required_string(fields, "name", "register_command")?,
        description: optional_string(fields, "description", "register_command")?,
    })
}

/// One registered CLI flag's descriptor, upstream's `ExtensionFlag` without
/// the `extensionPath` the host stamps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlagDeclaration {
    /// The flag name, without the leading dashes.
    pub name: String,
    /// The flag description.
    pub description: Option<String>,
    /// Whether the flag takes a boolean or a string value.
    pub flag_type: crate::extensions::runtime::FlagType,
    /// The default value, upstream's `default`.
    pub default: Option<crate::extensions::runtime::FlagValue>,
}

/// The registration descriptor a `register_flag` frame carries.
///
/// # Errors
///
/// Returns [`WireError`] when the name or value type is missing or mistyped.
pub fn flag_to_wire(flag: &FlagDeclaration) -> Result<Value, WireError> {
    let mut wire = json!({
        "name": flag.name,
        "type": match flag.flag_type {
            crate::extensions::runtime::FlagType::Boolean => "boolean",
            crate::extensions::runtime::FlagType::String => "string",
        },
    });
    if let Some(description) = &flag.description {
        wire["description"] = json!(description);
    }
    match &flag.default {
        Some(crate::extensions::runtime::FlagValue::Bool(value)) => {
            wire["default"] = json!(value);
        }
        Some(crate::extensions::runtime::FlagValue::Text(value)) => {
            wire["default"] = json!(value);
        }
        None => {}
    }
    Ok(wire)
}

/// Reads the flag descriptor of a `register_flag` value.
///
/// # Errors
///
/// Returns [`WireError`] when the name or value type is missing or mistyped,
/// or the default does not match the flag's own value type.
pub fn flag_declaration_from_wire(flag: &Value) -> Result<FlagDeclaration, WireError> {
    let fields = fields(flag, "register_flag")?;
    let flag_type = match required_string(fields, "type", "register_flag")?.as_str() {
        "boolean" => crate::extensions::runtime::FlagType::Boolean,
        "string" => crate::extensions::runtime::FlagType::String,
        other => {
            return Err(payload_error(&format!(
                "register_flag's type `{other}` is unknown"
            )));
        }
    };
    let default = match fields.get("default") {
        None | Some(Value::Null) => None,
        Some(Value::Bool(value)) => Some(crate::extensions::runtime::FlagValue::Bool(*value)),
        Some(Value::String(value)) => {
            Some(crate::extensions::runtime::FlagValue::Text(value.clone()))
        }
        Some(_) => {
            return Err(payload_error(
                "register_flag's default must be a boolean or a string",
            ));
        }
    };
    if let (Some(default), crate::extensions::runtime::FlagType::Boolean) = (&default, flag_type)
        && !matches!(default, crate::extensions::runtime::FlagValue::Bool(_))
    {
        return Err(payload_error("a boolean flag's default must be a boolean"));
    }
    Ok(FlagDeclaration {
        name: required_string(fields, "name", "register_flag")?,
        description: optional_string(fields, "description", "register_flag")?,
        flag_type,
        default,
    })
}

/// Writes the provider registration's data fields, upstream's
/// `(name, config)` overload.
///
/// The stream override, refresh, and OAuth closures stay host-side —
/// compile-ins register them in-process, wire children's calls ride #170's
/// dispatch — and the native `Provider` overload stays behind them per
/// ADR 0007. The wire carries everything else: display name, base URL, API
/// key template, API, headers, bearer flag, and the model list.
///
/// # Errors
///
/// Returns [`WireError`] when a model member stops short of the wire's
/// JSON.
pub fn provider_to_wire(
    name: &str,
    config: &crate::provider_composer::ProviderConfigInput,
) -> Result<Value, WireError> {
    let mut config_wire = json!({});
    if let Some(display) = &config.name {
        config_wire["name"] = json!(display);
    }
    if let Some(base_url) = &config.base_url {
        config_wire["baseUrl"] = json!(base_url);
    }
    if let Some(api_key) = &config.api_key {
        config_wire["apiKey"] = json!(api_key);
    }
    if let Some(api) = &config.api {
        config_wire["api"] = serde_to_wire(api, "api")?;
    }
    if let Some(headers) = &config.headers {
        config_wire["headers"] = json!(headers);
    }
    if let Some(auth_header) = config.auth_header {
        config_wire["authHeader"] = json!(auth_header);
    }
    if let Some(models) = &config.models {
        let mut models_wire = Vec::new();
        for model in models {
            models_wire.push(provider_model_to_wire(model)?);
        }
        config_wire["models"] = Value::from(models_wire);
    }
    Ok(json!({ "name": name, "config": config_wire }))
}

fn provider_model_to_wire(
    model: &crate::provider_composer::ProviderModelInput,
) -> Result<Value, WireError> {
    let mut wire = json!({
        "id": model.id,
        "name": model.name,
        "reasoning": model.reasoning,
        "input": serde_to_wire(&model.input, "model input modalities")?,
        "cost": serde_to_wire(&model.cost, "model cost")?,
        "contextWindow": model.context_window,
        "maxTokens": model.max_tokens,
    });
    if let Some(api) = &model.api {
        wire["api"] = serde_to_wire(api, "model api")?;
    }
    if let Some(base_url) = &model.base_url {
        wire["baseUrl"] = json!(base_url);
    }
    if let Some(thinking_level_map) = &model.thinking_level_map {
        wire["thinkingLevelMap"] = serde_to_wire(thinking_level_map, "thinkingLevelMap")?;
    }
    if let Some(sampling_params) = &model.sampling_params {
        wire["samplingParams"] = json!(sampling_params);
    }
    if let Some(headers) = &model.headers {
        wire["headers"] = json!(headers);
    }
    if let Some(compat) = &model.compat {
        wire["compat"] = serde_to_wire(compat, "compat")?;
    }
    Ok(wire)
}

fn provider_model_from_wire(
    model: &Value,
) -> Result<crate::provider_composer::ProviderModelInput, WireError> {
    let fields = fields(model, "provider_model")?;
    Ok(crate::provider_composer::ProviderModelInput {
        id: required_string(fields, "id", "provider_model")?,
        name: required_string(fields, "name", "provider_model")?,
        api: match fields.get("api") {
            None | Some(Value::Null) => None,
            Some(api) => Some(serde_from_wire(api.clone(), "api")?),
        },
        base_url: optional_string(fields, "baseUrl", "provider_model")?,
        reasoning: required_bool(fields, "reasoning", "provider_model")?,
        thinking_level_map: match fields.get("thinkingLevelMap") {
            None | Some(Value::Null) => None,
            Some(map) => Some(serde_from_wire(map.clone(), "thinkingLevelMap")?),
        },
        input: serde_from_wire(
            required_value(fields, "input", "provider_model")?.clone(),
            "model input modalities",
        )?,
        cost: serde_from_wire(
            required_value(fields, "cost", "provider_model")?.clone(),
            "cost",
        )?,
        context_window: required_u64(fields, "contextWindow", "provider_model")?,
        max_tokens: required_u64(fields, "maxTokens", "provider_model")?,
        sampling_params: match fields.get("samplingParams") {
            None | Some(Value::Null) => None,
            Some(sampling_params) => {
                Some(serde_from_wire(sampling_params.clone(), "samplingParams")?)
            }
        },
        headers: match fields.get("headers") {
            None | Some(Value::Null) => None,
            Some(headers) => Some(serde_from_wire(headers.clone(), "model headers")?),
        },
        compat: match fields.get("compat") {
            None | Some(Value::Null) => None,
            Some(compat) => Some(serde_from_wire(compat.clone(), "compat")?),
        },
    })
}

/// Reads the provider registration descriptor of a `register_provider`
/// value: the registration name and the data config, with the closure
/// members all absent.
///
/// # Errors
///
/// Returns [`WireError`] when the name, the config object, or a model field
/// is missing or mistyped.
pub fn provider_from_wire(
    provider: &Value,
) -> Result<(String, crate::provider_composer::ProviderConfigInput), WireError> {
    let provider_fields = fields(provider, "register_provider")?;
    let name = required_string(provider_fields, "name", "register_provider")?;
    let config = required_value(provider_fields, "config", "register_provider")?;
    let config_fields = fields(config, "provider_config")?;
    let models = match config_fields.get("models") {
        None | Some(Value::Null) => None,
        Some(models) => {
            let mut resolved = Vec::new();
            for model in models
                .as_array()
                .ok_or_else(|| payload_error("register_provider's models must be an array"))?
            {
                resolved.push(provider_model_from_wire(model)?);
            }
            Some(resolved)
        }
    };
    Ok((
        name,
        crate::provider_composer::ProviderConfigInput {
            name: optional_string(config_fields, "name", "provider_config")?,
            base_url: optional_string(config_fields, "baseUrl", "provider_config")?,
            api_key: optional_string(config_fields, "apiKey", "provider_config")?,
            api: match config_fields.get("api") {
                None | Some(Value::Null) => None,
                Some(api) => Some(serde_from_wire(api.clone(), "provider api")?),
            },
            // The stream override, refresh, and OAuth closures stay
            // host-side; a wire child's registrations carry data only.
            stream_simple: None,
            headers: match config_fields.get("headers") {
                None | Some(Value::Null) => None,
                Some(headers) => Some(serde_from_wire(headers.clone(), "provider headers")?),
            },
            auth_header: optional_bool(config_fields, "authHeader", "provider_config")?,
            oauth: None,
            models,
            refresh_models: None,
        },
    ))
}

/// The serializable [`ExtensionContext`](crate::extensions::ExtensionContext)
/// snapshot a `call_tool` frame carries.
///
/// The snapshot holds the cwd, the model, the thinking level, and the
/// session identifiers the bash spawn environment exposes; the interactive
/// surface reads through the context-call methods instead.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextSnapshot {
    /// The cwd override tool path resolution falls back from.
    pub cwd: String,
    /// The current model, absent when none is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<pi_ai::types::Model>,
    /// The current thinking level, absent when none is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<pi_agent_core::types::ThinkingLevel>,
    /// The session id the spawn environment exposes as `PI_SESSION_ID`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// The session file the spawn environment exposes as
    /// `PI_SESSION_FILE`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_file: Option<String>,
}

impl ContextSnapshot {
    /// Reads the snapshot off the context each dispatch delivers.
    #[must_use]
    pub fn from_extension_context(context: &dyn crate::extensions::ExtensionContext) -> Self {
        Self {
            cwd: context.cwd().to_string(),
            model: context.model().cloned(),
            thinking_level: context.thinking_level(),
            session_id: context.session_id(),
            session_file: context.session_file(),
        }
    }
}

/// Writes the context snapshot for a `call_tool` frame.
///
/// # Errors
///
/// Returns [`WireError`] when the snapshot's model stops short of the
/// wire's JSON.
pub fn context_snapshot_to_wire(snapshot: &ContextSnapshot) -> Result<Value, WireError> {
    serde_to_wire(snapshot, "context_snapshot")
}

/// Reads the context snapshot of a `call_tool` value.
///
/// # Errors
///
/// Returns [`WireError`] when the cwd or a member is mistyped.
pub fn context_snapshot_from_wire(snapshot: &Value) -> Result<ContextSnapshot, WireError> {
    serde_from_wire(snapshot.clone(), "context_snapshot")
}

/// Writes one resource's provenance, upstream's `SourceInfo` interface.
pub(super) fn source_info_to_wire(info: &crate::source_info::SourceInfo) -> Value {
    let mut wire = json!({
        "path": info.path,
        "source": info.source,
        "scope": match info.scope {
            crate::source_info::SourceScope::User => "user",
            crate::source_info::SourceScope::Project => "project",
            crate::source_info::SourceScope::Temporary => "temporary",
        },
        "origin": match info.origin {
            crate::source_info::SourceOrigin::Package => "package",
            crate::source_info::SourceOrigin::TopLevel => "topLevel",
        },
    });
    if let Some(base_dir) = &info.base_dir {
        wire["baseDir"] = json!(base_dir);
    }
    wire
}

pub(super) fn source_info_from_wire(
    value: &Value,
) -> Result<crate::source_info::SourceInfo, WireError> {
    let fields = fields(value, "source_info")?;
    let scope = match required_string(fields, "scope", "source_info")?.as_str() {
        "user" => crate::source_info::SourceScope::User,
        "project" => crate::source_info::SourceScope::Project,
        "temporary" => crate::source_info::SourceScope::Temporary,
        other => {
            return Err(payload_error(&format!(
                "source_info scope `{other}` is unknown"
            )));
        }
    };
    let origin = match required_string(fields, "origin", "source_info")?.as_str() {
        "package" => crate::source_info::SourceOrigin::Package,
        "topLevel" => crate::source_info::SourceOrigin::TopLevel,
        other => {
            return Err(payload_error(&format!(
                "source_info origin `{other}` is unknown"
            )));
        }
    };
    Ok(crate::source_info::SourceInfo {
        path: required_string(fields, "path", "source_info")?,
        source: required_string(fields, "source", "source_info")?,
        scope,
        origin,
        base_dir: optional_string(fields, "baseDir", "source_info")?,
    })
}
