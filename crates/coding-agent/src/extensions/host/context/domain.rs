//! The `session.*` and `models.*` context namespaces, the session-manager
//! and model-registry reads a wire child reaches ([ADR 0010]).
//!
//! The names mirror the TypeScript accessors the in-process context carries
//! (`getSessionId`, `getAll`, `find`), so a child written against upstream's
//! context reads the same shapes.
//!
//! [ADR 0010]: ../../../../../../docs/adr/0010-extension-host-wire-protocol.md

use serde_json::{Value, json};

use crate::extensions::runtime::HandlerContext;

use super::HostCallError;

/// Whether the method belongs to the two domain namespaces.
pub(super) fn is_domain_method(method: &str) -> bool {
    method.starts_with("session.") || method.starts_with("models.")
}

/// Answer one `session.*` or `models.*` read.
///
/// # Errors
///
/// Returns an in-scope [`HostCallError`] for an unknown name, malformed
/// parameters, or a stale context.
pub(super) fn call_domain(
    context: &HandlerContext,
    method: &str,
    params: Option<&Value>,
) -> Result<Value, HostCallError> {
    match method {
        "session.getSessionId" => context
            .session_manager()
            .map(|manager| Value::from(manager.session_id))
            .map_err(HostCallError::stale),
        "session.getSessionFile" => context
            .session_manager()
            .map(|manager| {
                manager
                    .session_file
                    .as_deref()
                    .map_or(Value::Null, Value::from)
            })
            .map_err(HostCallError::stale),
        "session.getSessionName" => context
            .session_manager()
            .map(|manager| manager.get_session_name().map_or(Value::Null, Value::from))
            .map_err(HostCallError::stale),
        "session.getSessionDir" => context
            .session_manager()
            .map(|manager| Value::from(manager.session_dir))
            .map_err(HostCallError::stale),
        "session.getCwd" => context
            .session_manager()
            .map(|manager| Value::from(manager.cwd))
            .map_err(HostCallError::stale),
        "models.getAll" => context
            .model_registry()
            .map_err(HostCallError::stale)
            .and_then(|registry| json_value(registry.get_all())),
        "models.getAvailable" => context
            .model_registry()
            .map_err(HostCallError::stale)
            .and_then(|registry| json_value(registry.get_available())),
        "models.find" => {
            let provider = required_param_string(params, "provider")?;
            let model = required_param_string(params, "model")?;
            context
                .model_registry()
                .map(|registry| {
                    registry
                        .find(&provider, &model)
                        .map_or(Value::Null, |model| json!(model))
                })
                .map_err(HostCallError::stale)
        }
        other => Err(HostCallError::unknown_method(other)),
    }
}

fn json_value<T: serde::Serialize>(value: T) -> Result<Value, HostCallError> {
    serde_json::to_value(value)
        .map_err(|error| HostCallError::invalid_params(format!("the value is not JSON: {error}")))
}

fn params_object(
    params: Option<&Value>,
) -> Result<Option<&serde_json::Map<String, Value>>, HostCallError> {
    match params {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(object)) => Ok(Some(object)),
        Some(_) => Err(HostCallError::invalid_params(
            "the context method's params must be an object",
        )),
    }
}

fn required_param_string(params: Option<&Value>, key: &str) -> Result<String, HostCallError> {
    match params_object(params)?.and_then(|object| object.get(key)) {
        Some(Value::String(text)) => Ok(text.clone()),
        _ => Err(HostCallError::invalid_params(format!(
            "the context method needs a string `{key}`"
        ))),
    }
}

pub(super) fn optional_param_string(
    params: Option<&Value>,
    key: &str,
) -> Result<Option<String>, HostCallError> {
    match params_object(params)?.and_then(|object| object.get(key)) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(_) => Err(HostCallError::invalid_params(format!(
            "the context method's `{key}` must be a string"
        ))),
    }
}
