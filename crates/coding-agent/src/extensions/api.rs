//! The inline-factory `pi` surface and the loader runtime it shares, the
//! port of upstream's creator trio in `src/core/extensions/loader.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Each factory receives an [`ExtensionApi`] over one [`Extension`] record
//! and one shared [`ExtensionRuntime`].
//!
//! Registration methods write the record; `registerProvider` queues into
//! the runtime so a later bind pass can flush once the model registry
//! exists. The factory runs to a result: an error (upstream's throw)
//! discards the API's pending state and fails the load.
//!
//! The interactive action methods (`sendMessage`, `setModel`, …) throw
//! "not initialized" upstream during load and delegate post-bind; those
//! ride their consumers' tickets. The event-bus surface and the session
//! replace/reload subscriptions ride theirs. Upstream's
//! `registerNativeProvider` queue rides the composer's native-provider
//! ticket; the named-provider queue is wired here.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pi_ai::types::BoxedFuture;

use crate::extensions::runtime::{
    CommandHandler, Extension, ExtensionFlag, ExtensionHandler, FlagType, FlagValue,
    RegisteredTool, ResolvedCommand,
};
use crate::extensions::types::ToolDefinition;
use crate::provider_composer::ProviderConfigInput;
use crate::source_info::{
    SourceOrigin, SourceScope, SyntheticSourceOptions, create_synthetic_source_info,
};

/// The inline factory, upstream's `ExtensionFactory`: receives the API,
/// returns an error in place of upstream's throw.
pub type ExtensionFactory =
    Arc<dyn Fn(Arc<ExtensionApi>) -> BoxedFuture<'static, Result<(), String>> + Send + Sync>;

/// One inline-factory input, upstream's `InlineExtension` union: a bare
/// factory labels as `<inline:N>`, a named one as `<inline:name>` and may
/// hide itself from user-facing lists.
pub struct InlineExtension {
    /// The display name, `None` for the positional `<inline:N>` label.
    pub name: Option<String>,
    /// Whether the extension stays out of the startup Extensions list.
    pub hidden: bool,
    /// The factory.
    pub factory: ExtensionFactory,
}

impl std::fmt::Debug for InlineExtension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The factory does not debug; the display surface does.
        f.debug_struct("InlineExtension")
            .field("name", &self.name)
            .field("hidden", &self.hidden)
            .finish_non_exhaustive()
    }
}

/// One queued provider registration, upstream's
/// `runtime.registerProvider(name, config, extensionPath)`.
///
/// Upstream's free-form `ProviderConfig` object restates to the port's
/// filled [`ProviderConfigInput`].
#[derive(Clone, Debug)]
pub struct PendingProviderRegistration {
    /// The provider name.
    pub name: String,
    /// The provider configuration.
    pub config: ProviderConfigInput,
    /// The extension that registered it.
    pub extension_path: String,
}

/// The post-bind provider registration the runtime routes registrations
/// to directly, upstream's `runtime.registerProvider` replacement after
/// `bindCore` flushes the queue.
pub type BoundProviderActions = Arc<crate::extensions::runner::ProviderActions>;

/// The loader runtime shared by every extension API in one load pass,
/// upstream's `ExtensionRuntime`.
///
/// Carries the flag values, the queued provider registrations, and the
/// session-replacement staleness state. Providers queue until a bind pass
/// drains the queue ([`Self::take_pending_provider_registrations`]).
#[derive(Clone, Default)]
pub struct ExtensionRuntime {
    /// The first staleness message wins; set by session replacement or
    /// reload, upstream's `state.staleMessage`.
    stale_message: Arc<Mutex<Option<String>>>,
    /// The flag defaults and values, upstream's `flagValues`.
    flag_values: Arc<Mutex<HashMap<String, FlagValue>>>,
    /// Provider registrations queued until the bind pass, upstream's
    /// `pendingProviderRegistrations`.
    pending_provider_registrations: Arc<Mutex<Vec<Arc<PendingProviderRegistration>>>>,
    /// The post-bind provider registration, upstream's bindCore
    /// `runtime.registerProvider` replacement; absent pre-bind.
    bound_provider_actions: Arc<Mutex<Option<BoundProviderActions>>>,
}

impl ExtensionRuntime {
    /// Build the runtime with empty state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Assert stale-free, upstream's `assertActive`.
    ///
    /// # Errors
    /// When the runtime was invalidated by session replacement or reload.
    pub fn assert_active(&self) -> Result<(), String> {
        let stale = self
            .stale_message
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        stale
            .as_ref()
            .map_or(Ok(()), |message| Err(message.clone()))
    }
    /// Mark the runtime stale, upstream's `invalidate`; the first message
    /// wins and later calls are no-ops. Without a message the port carries
    /// upstream's guidance sentence.
    pub fn invalidate(&self, message: Option<String>) {
        let mut stale = self
            .stale_message
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if stale.is_some() {
            return;
        }
        *stale = Some(message.unwrap_or_else(|| {
            "This extension ctx is stale after session replacement or reload. Do not use a \
             captured pi or command ctx after ctx.newSession(), ctx.fork(), \
             ctx.switchSession(), or ctx.reload(). For newSession, fork, and switchSession, \
             move post-replacement work into withSession and use the ctx passed to \
             withSession. For reload, do not use the old ctx after await ctx.reload()."
                .to_string()
        }));
    }

    /// Write one flag value, upstream's `runtime.flagValues.set`. The
    /// seeding guard lives with the commit path.
    pub fn set_flag_value(&self, name: &str, value: FlagValue) {
        self.flag_values
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(name.to_string(), value);
    }

    /// Read one flag value, upstream's `runtime.flagValues.get`.
    #[must_use]
    pub fn flag_value(&self, name: &str) -> Option<FlagValue> {
        self.flag_values
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(name)
            .cloned()
    }

    /// The whole flag-value snapshot, the runner's `getFlagValues` read;
    /// alphabetical because the port flattens upstream's insertion map.
    #[must_use]
    pub fn flag_values(&self) -> std::collections::BTreeMap<String, FlagValue> {
        let map = self
            .flag_values
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        map.iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }

    /// Arm the post-bind provider registration, upstream's bindCore
    /// replacing `runtime.registerProvider`; `None` clears the armed
    /// actions and re-queues registrations.
    pub fn bind_provider_actions(&self, actions: Option<BoundProviderActions>) {
        *self
            .bound_provider_actions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = actions;
    }

    /// The armed post-bind provider registration, the branch the API's
    /// register/unregister reads.
    #[must_use]
    pub fn bound_provider_actions(&self) -> Option<BoundProviderActions> {
        self.bound_provider_actions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Queue one provider registration until the bind pass, upstream's
    /// `runtime.registerProvider` default.
    pub fn register_provider(&self, registration: PendingProviderRegistration) {
        self.pending_provider_registrations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Arc::new(registration));
    }

    /// Drop every queued registration for the name, upstream's
    /// `runtime.unregisterProvider` queue filters.
    pub fn unregister_provider(&self, name: &str) {
        self.pending_provider_registrations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|queued| queued.name != name);
    }

    /// Drain the queue for the bind pass; removing entries keeps a second
    /// drain from double-registering.
    pub fn take_pending_provider_registrations(&self) -> Vec<Arc<PendingProviderRegistration>> {
        self.pending_provider_registrations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .drain(..)
            .collect()
    }
}

impl std::fmt::Debug for ExtensionRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stale = self
            .stale_message
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some();
        let flags = self
            .flag_values
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len();
        let queued = self
            .pending_provider_registrations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len();
        let bound = self
            .bound_provider_actions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some();
        f.debug_struct("ExtensionRuntime")
            .field("stale", &stale)
            .field("flag_values", &flags)
            .field("pending_provider_registrations", &queued)
            .field("bound_provider_actions", &bound)
            .finish()
    }
}

/// The source-label split of an inline path, upstream's `createExtension`:
/// a `<...>` label reads its pre-colon segment (empty reads as
/// "temporary"), anything else reads as "local".
fn inline_source_label(extension_path: &str) -> String {
    if extension_path.starts_with('<') && extension_path.ends_with('>') {
        let inner = &extension_path[1..extension_path.len() - 1];
        let label = inner.split(':').next().unwrap_or("");
        if label.is_empty() {
            "temporary".to_string()
        } else {
            label.to_string()
        }
    } else {
        "local".to_string()
    }
}
impl std::fmt::Debug for ExtensionApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = *self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        f.debug_struct("ExtensionApi")
            .field("path", &self.extension.path)
            .field("cwd", &self.cwd)
            .field("state", &state)
            .field("runtime", &self.runtime)
            .field(
                "pending_flag_values",
                &self
                    .pending_flag_values
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .len(),
            )
            .finish_non_exhaustive()
    }
}

/// Create one `Extension` record for an inline load, upstream's
/// `createExtension`: the record path and resolved path read the factory
/// label, provenance derives from it, no base dir for the `<...>` labels.
fn create_extension(extension_path: &str) -> Arc<Extension> {
    let source = inline_source_label(extension_path);
    let base_dir = if extension_path.starts_with('<') && extension_path.ends_with('>') {
        None
    } else {
        std::path::Path::new(extension_path)
            .parent()
            .map(|parent| parent.to_string_lossy().into_owned())
    };
    let source_info = create_synthetic_source_info(
        extension_path,
        &SyntheticSourceOptions {
            source,
            scope: Some(SourceScope::Temporary),
            origin: Some(SourceOrigin::TopLevel),
            base_dir,
        },
    );
    Arc::new(Extension::new(
        extension_path.to_string(),
        extension_path.to_string(),
        false,
        source_info,
    ))
}

/// Run one inline factory and initialize its record, upstream's
/// `initializeExtension` plus `loadExtensionFromFactory`: an error result
/// discards the API and fails the load.
///
/// # Errors
/// When the factory failed or the runtime went stale before commit.
async fn initialize_extension(
    factory: &ExtensionFactory,
    extension: Arc<Extension>,
    runtime: &ExtensionRuntime,
    cwd: String,
) -> Result<Arc<Extension>, String> {
    let api = Arc::new(ExtensionApi::new(
        Arc::clone(&extension),
        runtime.clone(),
        cwd,
    ));
    let factory_result = (factory)(Arc::clone(&api)).await;
    match factory_result {
        Ok(()) => {
            if let Err(stale) = api.commit() {
                api.discard();
                return Err(stale);
            }
            Ok(extension)
        }
        Err(message) => {
            api.discard();
            Err(message)
        }
    }
}

/// Load one inline factory into its record, upstream's
/// `loadExtensionFromFactory`: the record path reads the factory label.
///
/// # Errors
/// When the factory failed or the runtime went stale before commit.
pub async fn load_extension_from_factory(
    factory: &ExtensionFactory,
    runtime: &ExtensionRuntime,
    cwd: &str,
    extension_path: &str,
) -> Result<Arc<Extension>, String> {
    let extension = create_extension(extension_path);
    initialize_extension(factory, extension, runtime, cwd.to_string()).await
}

/// The API lifecycle state, upstream's `load.state`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ApiState {
    /// The factory is running; runtime writes queue.
    #[default]
    Loading,
    /// The factory committed; runtime writes apply.
    Active,
    /// The factory failed; the API rejects every call.
    Failed,
}

/// The `pi` object one factory receives, upstream's `createExtensionAPI`:
/// registration methods write the extension record, flag reads fall from
/// the runtime flags back to the pending seeds.
pub struct ExtensionApi {
    /// The record the registrations write.
    extension: Arc<Extension>,
    /// The shared loader runtime.
    runtime: ExtensionRuntime,
    /// The cwd the API methods resolve against; the `exec` method rides
    /// its consumers' ticket and reads this.
    cwd: String,
    /// The lifecycle state.
    state: Mutex<ApiState>,
    /// The flag defaults seeded pre-commit, upstream's `pendingFlagValues`.
    pending_flag_values: Mutex<HashMap<String, FlagValue>>,
}

impl ExtensionApi {
    /// Build the API over an empty record, upstream's `createExtensionAPI`
    /// entry state.
    #[must_use]
    pub fn new(extension: Arc<Extension>, runtime: ExtensionRuntime, cwd: String) -> Self {
        Self {
            extension,
            runtime,
            cwd,
            state: Mutex::new(ApiState::Loading),
            pending_flag_values: Mutex::new(HashMap::new()),
        }
    }

    /// The extension path, the `extension.path` diagnostic surface.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.extension.path
    }

    /// The record the registrations write, the loader layer's read side.
    #[must_use]
    pub const fn extension(&self) -> &Arc<Extension> {
        &self.extension
    }

    /// The cwd the API methods resolve against.
    #[must_use]
    pub fn cwd(&self) -> &str {
        &self.cwd
    }

    /// The loader's commit, upstream's `load.commit`: flush the pending
    /// flag seeds into the runtime and mark the API active. Not a factory
    /// method; the loading path owns it.
    ///
    /// # Errors
    /// When the runtime was invalidated before the commit.
    pub fn commit(&self) -> Result<(), String> {
        if !self.is_loading() {
            return Ok(());
        }
        self.runtime.assert_active()?;
        self.seed_pending_flags();
        self.mark_active();
        Ok(())
    }

    /// The loader's discard, upstream's `load.discard`: mark the API
    /// failed and drop the pending seeds.
    pub fn discard(&self) {
        if self.is_loading() {
            self.mark_failed_without_seeds();
        }
    }

    /// Whether the factory is still running, the `load.state` read the
    /// lifecycle transitions check.
    fn is_loading(&self) -> bool {
        *self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            == ApiState::Loading
    }

    /// Flush the pending flag seeds into the runtime first-come, then drop
    /// the seeds, upstream's commit body between the state checks.
    fn seed_pending_flags(&self) {
        let pending = self
            .pending_flag_values
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (name, value) in pending.iter() {
            if self.runtime.flag_value(name).is_none() {
                self.runtime.set_flag_value(name, value.clone());
            }
        }
        drop(pending);
        self.pending_flag_values
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }

    /// Mark the API active, the post-flush commit half.
    fn mark_active(&self) {
        *self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = ApiState::Active;
    }

    /// Mark the API failed and drop the pending seeds, without touching an
    /// already-transitioned state.
    fn mark_failed_without_seeds(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *state == ApiState::Loading {
            *state = ApiState::Failed;
        }
        drop(state);
        self.pending_flag_values
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
    /// Assert the API may act, upstream's `assertActive`: a failed load
    /// deactivates the API before any runtime check.
    ///
    /// # Errors
    /// When the load failed or the runtime went stale.
    pub fn assert_active(&self) -> Result<(), String> {
        if *self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            == ApiState::Failed
        {
            return Err(format!(
                "Extension \"{}\" failed to load and its API is no longer active.",
                self.extension.path
            ));
        }
        self.runtime.assert_active()
    }

    /// Register one event handler, upstream's `pi.on`.
    ///
    /// # Errors
    /// When the API may not act.
    pub fn on(&self, channel: &str, handler: ExtensionHandler) -> Result<(), String> {
        self.assert_active()?;
        self.extension.on(channel, handler);
        Ok(())
    }

    /// Register one tool, upstream's `pi.registerTool`; the parameter
    /// schema must be a JSON object.
    ///
    /// # Errors
    /// When the API may not act, or the schema is not an object.
    pub fn register_tool(&self, tool: ToolDefinition) -> Result<(), String> {
        self.assert_active()?;
        if !tool.parameters.is_object() {
            return Err(format!(
                "Tool \"{}\" registered by extension \"{}\" must define an object parameter \
                 schema.",
                tool.name, self.extension.path
            ));
        }
        self.extension.register_tool(RegisteredTool {
            definition: tool,
            source_info: self.extension.source_info.clone(),
        });
        Ok(())
    }
    /// Register one command, upstream's `pi.registerCommand`; the
    /// invocation name resolves later against colliding registrations.
    ///
    /// # Errors
    /// When the API may not act.
    pub fn register_command(
        &self,
        name: String,
        description: Option<String>,
        handler: CommandHandler,
    ) -> Result<(), String> {
        self.assert_active()?;
        self.extension.register_command(ResolvedCommand {
            name: name.clone(),
            invocation_name: name,
            description,
            source_info: self.extension.source_info.clone(),
            handler,
        });
        Ok(())
    }
    /// Register one CLI flag, upstream's `pi.registerFlag`: a default must
    /// carry the declared value type, and the default seeds the runtime
    /// value first-come.
    ///
    /// # Errors
    /// When the API may not act, or the default type mismatches.
    pub fn register_flag(
        &self,
        name: String,
        description: Option<String>,
        flag_type: FlagType,
        default: Option<FlagValue>,
    ) -> Result<(), String> {
        self.assert_active()?;
        if let Some(value) = &default {
            let expected = match flag_type {
                FlagType::Boolean => "boolean",
                FlagType::String => "string",
            };
            let got = match value {
                FlagValue::Bool(_) => "boolean",
                FlagValue::Text(_) => "string",
            };
            if expected != got {
                return Err(format!(
                    "Invalid default for flag \"{name}\": expected {expected}, got {got}"
                ));
            }
        }
        self.extension.register_flag(ExtensionFlag {
            name: name.clone(),
            description,
            flag_type,
            default: default.clone(),
            extension_path: self.extension.path.clone(),
        });
        if let Some(value) = default
            && self.runtime.flag_value(&name).is_none()
        {
            self.pending_flag_values
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .entry(name)
                .or_insert(value);
        }
        Ok(())
    }
    /// Read one flag this extension registered, upstream's `pi.getFlag`:
    /// the runtime value first, then the pending seed, `None` for an
    /// unregistered name.
    ///
    /// # Errors
    /// When the API may not act.
    pub fn get_flag(&self, name: &str) -> Result<Option<FlagValue>, String> {
        self.assert_active()?;
        if !self.extension.has_flag(name) {
            return Ok(None);
        }
        let pending = self
            .pending_flag_values
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(self
            .runtime
            .flag_value(name)
            .or_else(|| pending.get(name).cloned()))
    }

    /// Queue one provider registration for the bind pass, upstream's
    /// `pi.registerProvider` named branch. Post-bind the registration
    /// registers directly through the armed provider actions.
    ///
    /// # Errors
    /// When the API may not act, or the post-bind registration fails.
    pub fn register_provider(
        &self,
        name: String,
        config: ProviderConfigInput,
    ) -> Result<(), String> {
        self.assert_active()?;
        if let Some(provider_actions) = self.runtime.bound_provider_actions() {
            return (provider_actions.register_provider)(name, config);
        }
        self.runtime.register_provider(PendingProviderRegistration {
            name,
            config,
            extension_path: self.extension.path.clone(),
        });
        Ok(())
    }

    /// Drop every queued registration for the name, upstream's
    /// `pi.unregisterProvider`. Post-bind the removal registers directly
    /// through the armed provider actions.
    ///
    /// # Errors
    /// When the API may not act, or the post-bind removal fails.
    pub fn unregister_provider(&self, name: &str) -> Result<(), String> {
        self.assert_active()?;
        if let Some(provider_actions) = self.runtime.bound_provider_actions() {
            return (provider_actions.unregister_provider)(name.to_string());
        }
        self.runtime.unregister_provider(name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "the tests pin outcomes; an unexpected result panics the test by design"
    )]
    use std::sync::Arc;

    use pi_agent_core::types::{AgentToolError, AgentToolResult};

    use super::{ExtensionApi, ExtensionFactory, ExtensionRuntime};
    use crate::extensions::events::ExtensionEvent;
    use crate::extensions::results::HandlerResult;
    use crate::extensions::runtime::{CommandHandler, ExtensionHandler, FlagType, FlagValue};
    use crate::extensions::types::ToolDefinition;
    use crate::provider_composer::ProviderConfigInput;
    use crate::source_info::SourceInfo;

    fn source_info(path: &str) -> SourceInfo {
        crate::source_info::create_synthetic_source_info(
            path,
            &crate::source_info::SyntheticSourceOptions::default(),
        )
    }

    fn api_with(extension_path: &str, runtime: &ExtensionRuntime) -> Arc<ExtensionApi> {
        let extension = crate::extensions::runtime::Extension::new(
            extension_path.to_string(),
            extension_path.to_string(),
            false,
            source_info(extension_path),
        );
        Arc::new(ExtensionApi::new(
            Arc::new(extension),
            runtime.clone(),
            "/tmp/project".to_string(),
        ))
    }

    fn noop_handler() -> ExtensionHandler {
        Arc::new(|_event: ExtensionEvent, _ctx| Box::pin(async { HandlerResult::None }))
    }

    fn noop_command() -> CommandHandler {
        Arc::new(|_args: String, _ctx| Box::pin(async {}))
    }

    fn simple_tool(name: &str, parameters: serde_json::Value) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            label: name.to_string(),
            description: "test".to_string(),
            prompt_snippet: None,
            prompt_guidelines: None,
            parameters,
            constrained_sampling: None,
            render_shell: None,
            prepare_arguments: None,
            execution_mode: None,
            execute: Arc::new(
                |_call: &str,
                 _params: &serde_json::Value,
                 _signal: Option<&pi_agent_core::harness::context::AbortSignal>,
                 _update: Option<pi_agent_core::harness::types::AgentHarnessToolUpdateCallback>,
                 _context: Option<&dyn crate::extensions::ExtensionContext>| {
                    Box::pin(async {
                        Ok::<AgentToolResult, AgentToolError>(AgentToolResult {
                            content: Vec::new(),
                            details: serde_json::json!({}),
                            usage: None,
                            added_tool_names: None,
                            terminate: None,
                        })
                    })
                },
            ),
        }
    }
    #[test]
    fn flag_and_tool_registrations_write_the_record() {
        let runtime = ExtensionRuntime::new();
        let api = api_with("/ext/test.ts", &runtime);

        api.register_tool(simple_tool("find", serde_json::json!({ "type": "object" })))
            .expect("the tool registers");
        api.on("session_start", noop_handler())
            .expect("the handler registers");
        api.register_command(
            "ship".to_string(),
            Some("ships it".to_string()),
            noop_command(),
        )
        .expect("the command registers");
        api.register_flag(
            "port".to_string(),
            Some("the port".to_string()),
            FlagType::String,
            Some(FlagValue::Text("8080".to_string())),
        )
        .expect("the flag registers");

        // Pre-commit, the flag reads as the pending seed; the runtime is empty.
        assert_eq!(
            api.get_flag("port").expect("flag read"),
            Some(FlagValue::Text("8080".to_string()))
        );
        assert!(runtime.flag_value("port").is_none());
        let record = api.extension();
        assert_eq!(
            record
                .tools
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len(),
            1
        );
        assert_eq!(
            record
                .commands
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len(),
            1
        );
        assert!(record.has_flag("port"));

        api.commit().expect("the commit flushes");
        // The flag seed moved to the runtime.
        assert_eq!(
            runtime.flag_value("port"),
            Some(FlagValue::Text("8080".to_string()))
        );
        // getFlag keeps reading after the commit.
        assert_eq!(
            api.get_flag("port").expect("flag read"),
            Some(FlagValue::Text("8080".to_string()))
        );
        // An unregistered name reads as none.
        assert_eq!(api.get_flag("unregistered").expect("flag read"), None);
    }

    #[test]
    fn flag_default_type_mismatch_is_rejected() {
        let runtime = ExtensionRuntime::new();
        let api = api_with("/ext/test.ts", &runtime);
        let error = api
            .register_flag(
                "verbose".to_string(),
                None,
                FlagType::Boolean,
                Some(FlagValue::Text("yes".to_string())),
            )
            .expect_err("the mismatch rejects");
        assert_eq!(
            error,
            "Invalid default for flag \"verbose\": expected boolean, got string"
        );
    }

    #[test]
    fn non_object_tool_schema_is_rejected() {
        let runtime = ExtensionRuntime::new();
        let api = api_with("/ext/test.ts", &runtime);
        let error = api
            .register_tool(simple_tool("broken", serde_json::json!(["array"])))
            .expect_err("the schema rejects");
        assert_eq!(
            error,
            "Tool \"broken\" registered by extension \"/ext/test.ts\" must define an object \
             parameter schema."
        );
    }

    #[test]
    fn failed_load_discards_the_api() {
        let runtime = ExtensionRuntime::new();
        let api = api_with("/ext/test.ts", &runtime);
        api.discard();

        let error = api
            .on("session_start", noop_handler())
            .expect_err("the failed API rejects");
        assert_eq!(
            error,
            "Extension \"/ext/test.ts\" failed to load and its API is no longer active."
        );
        // The runtime staleness rides beneath the failed-API state.
        runtime.invalidate(None);
        assert!(runtime.assert_active().is_err());
        // Post-failure lifecycle calls are no-ops (upstream's state guard).
        api.commit().expect("the commit is a no-op");
        api.discard();

        let inline = super::InlineExtension {
            name: Some("demo".to_string()),
            hidden: true,
            factory: api_factory_noop(),
        };
        assert!(format!("{inline:?}").contains("demo"));
        assert!(format!("{runtime:?}").contains("stale: true"));
    }

    fn api_factory_noop() -> ExtensionFactory {
        Arc::new(|_api: Arc<ExtensionApi>| Box::pin(async { Ok(()) }))
    }

    #[tokio::test]
    async fn registered_callables_execute_through_the_record() {
        let runtime = ExtensionRuntime::new();
        let api = api_with("/ext/dispatch.ts", &runtime);
        assert_eq!(api.path(), "/ext/dispatch.ts");
        assert_eq!(api.cwd(), "/tmp/project");
        api.on("session_start", noop_handler())
            .expect("the handler registers");
        api.register_command("demo".to_string(), None, noop_command())
            .expect("the command registers");
        api.register_tool(simple_tool(
            "probe",
            serde_json::json!({ "type": "object" }),
        ))
        .expect("the tool registers");
        let record = api.extension();

        let handler = record
            .handlers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)["session_start"][0]
            .clone();
        let handler_outcome = (handler)(ExtensionEvent::AgentStart, dispatch_context().await).await;
        assert!(matches!(handler_outcome, HandlerResult::None));

        let command = record
            .commands
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)[0]
            .clone();
        (command.handler)("args".to_string(), dispatch_context().await).await;

        let tool = record
            .tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)[0]
            .clone();
        let executed = (tool.definition.execute)(
            "call-1",
            &serde_json::json!({}),
            None,
            None,
            Some(&crate::extensions::types::CwdContext {
                cwd: "/tmp".to_string(),
            }),
        )
        .await;
        assert!(executed.is_ok(), "the tool executes through its record");

        // The factory runs through its load pass lookalike.
        let factory = api_factory_noop();
        let fresh = api_with("/ext/fresh.ts", &runtime);
        (factory)(Arc::clone(&fresh))
            .await
            .expect("the factory runs");
    }

    #[test]
    fn stale_runtime_marks_every_api_inactive() {
        let runtime = ExtensionRuntime::new();
        let api = api_with("/ext/test.ts", &runtime);
        runtime.invalidate(None);

        let error = api
            .on("session_start", noop_handler())
            .expect_err("the stale runtime rejects");
        assert!(error.contains("stale after session replacement"), "{error}");
        // The first message wins.
        runtime.invalidate(Some("custom".to_string()));
        let error = api
            .on("session_start", noop_handler())
            .expect_err("still stale");
        assert!(!error.contains("custom"));
    }

    async fn dispatch_context() -> crate::extensions::runtime::HandlerContext {
        use crate::extensions::runtime::{ContextActions, HandlerContext};
        use crate::model_registry::ModelRegistry;
        use crate::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
        use crate::session_manager::SessionManager;
        use pi_agent_core::types::ThinkingLevel;
        use pi_ai::auth::credential_store::InMemoryCredentialStore;
        let manager = SessionManager::new("/tmp/project", "/tmp/sessions", None, false, None, None)
            .expect("the in-memory manager constructs");
        let registry_runtime = ModelRuntime::create(CreateModelRuntimeOptions {
            credentials: Some(Arc::new(InMemoryCredentialStore::default())),
            models_path: Some(None),
            allow_model_network: false,
            refresh_on_create: Some(false),
            ..CreateModelRuntimeOptions::default()
        })
        .await
        .expect("the runtime constructs offline");
        HandlerContext::new(
            "/tmp/project".to_string(),
            crate::extensions::reasons::ExtensionMode::Print,
            None,
            manager,
            ModelRegistry::new(registry_runtime),
            Arc::new(|| Ok(())),
            Arc::new(ContextActions {
                get_thinking_level: Some(Arc::new(|| ThinkingLevel::Off)),
                ..ContextActions::default()
            }),
        )
    }

    #[test]
    fn provider_queue_remembers_and_forgets_registrations() {
        let runtime = ExtensionRuntime::new();
        let api = api_with("/ext/test.ts", &runtime);
        api.register_provider("custom".to_string(), ProviderConfigInput::default())
            .expect("queued");
        api.register_provider("other".to_string(), ProviderConfigInput::default())
            .expect("queued");
        api.unregister_provider("custom").expect("dropped");
        let queued = runtime.take_pending_provider_registrations();
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].name, "other");
        assert_eq!(queued[0].extension_path, "/ext/test.ts");
        assert!(runtime.take_pending_provider_registrations().is_empty());

        // A second registration survives the earlier re-drain.
        api.register_provider("custom".to_string(), ProviderConfigInput::default())
            .expect("queued");
        assert_eq!(runtime.take_pending_provider_registrations().len(), 1);
        assert!(format!("{api:?}").contains("/ext/test.ts"));
    }
    #[tokio::test]
    async fn load_extension_from_factory_runs_commit_and_lifecycle() {
        let runtime = ExtensionRuntime::new();
        let factory: ExtensionFactory = Arc::new(|api: Arc<ExtensionApi>| {
            Box::pin(async move {
                assert_eq!(api.path(), "<inline:demo>");
                api.register_flag(
                    "demos".to_string(),
                    None,
                    FlagType::Boolean,
                    Some(FlagValue::Bool(true)),
                )?;
                api.register_tool(simple_tool(
                    "custom",
                    serde_json::json!({ "type": "object", "properties": {} }),
                ))?;
                api.on("session_start", noop_handler())?;
                api.register_command("demo".to_string(), None, noop_command())?;
                Ok(())
            })
        });
        let extension = crate::extensions::api::load_extension_from_factory(
            &factory,
            &runtime,
            "/tmp/project",
            "<inline:demo>",
        )
        .await
        .expect("the factory runs");
        assert!(
            !extension.hidden(),
            "the loader decides hidden, not initialize"
        );
        assert_eq!(extension.path, "<inline:demo>");
        assert!(
            extension
                .handlers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .contains_key("session_start")
        );
        assert_eq!(runtime.flag_value("demos"), Some(FlagValue::Bool(true)));

        // The hidden mark drives the inline loader, the factory does not.
        extension.set_hidden(false);
        assert!(!extension.hidden());
    }

    #[tokio::test]
    async fn failing_factory_fails_the_load_and_discards_the_api() {
        let runtime = ExtensionRuntime::new();
        let factory: ExtensionFactory = Arc::new(|_api: Arc<ExtensionApi>| {
            Box::pin(async { Err("the factory refused".to_string()) })
        });
        let error = crate::extensions::api::load_extension_from_factory(
            &factory,
            &runtime,
            "/tmp/project",
            "<inline:1>",
        )
        .await
        .expect_err("the load fails");
        assert_eq!(error, "the factory refused");
    }

    #[tokio::test]
    async fn uninitialized_commit_paths_an_error_through_the_loader() {
        let runtime = ExtensionRuntime::new();
        runtime.invalidate(Some("stale".to_string()));
        let factory: ExtensionFactory =
            Arc::new(|_api: Arc<ExtensionApi>| Box::pin(async { Ok(()) }));
        let error = crate::extensions::api::load_extension_from_factory(
            &factory,
            &runtime,
            "/tmp/project",
            "<inline:2>",
        )
        .await
        .expect_err("the stale runtime fails the commit");
        assert!(error.contains("stale"), "{error}");
    }

    #[tokio::test]
    async fn post_bind_provider_registration_registers_directly() {
        let runtime = ExtensionRuntime::new();
        let extension = crate::extensions::runtime::Extension::new(
            "/ext/post-bind.ts".to_string(),
            "/ext/post-bind.ts".to_string(),
            false,
            source_info("/ext/post-bind.ts"),
        );

        // Pre-bind the registration queues.
        let api = Arc::new(ExtensionApi::new(
            {
                let extension = Arc::new(extension);
                Arc::clone(&extension)
            },
            runtime.clone(),
            "/tmp/project".to_string(),
        ));
        api.register_provider("queued".to_string(), ProviderConfigInput::default())
            .expect("the queue accepts");
        assert_eq!(
            runtime.take_pending_provider_registrations().len(),
            1,
            "the pre-bind path queues"
        );

        // Post-bind the registrations route through the armed actions and
        // return their failures; the queue stays drained.
        runtime.bind_provider_actions(Some(Arc::new(crate::extensions::runner::ProviderActions {
            register_provider: Arc::new(|name: String, _config: ProviderConfigInput| {
                Err(format!("cannot register {name}"))
            }),
            unregister_provider: Arc::new(|_name: String| Ok(())),
        })));
        assert_eq!(
            api.register_provider("bound".to_string(), ProviderConfigInput::default())
                .expect_err("the post-bind failure propagates"),
            "cannot register bound".to_string()
        );
        assert!(runtime.take_pending_provider_registrations().is_empty());
        assert_eq!(
            api.unregister_provider("queued")
                .expect("the post-bind drop"),
            ()
        );

        // Unarming re-queues: the reset returns the immediate path.
        runtime.bind_provider_actions(None);
        api.register_provider("after-reset".to_string(), ProviderConfigInput::default())
            .expect("the reset path queues");
        assert_eq!(runtime.take_pending_provider_registrations().len(), 1);
        assert!(
            format!("{runtime:?}").contains("bound_provider_actions: false"),
            "{}",
            format!("{runtime:?}")
        );
    }

    #[test]
    fn flag_values_snapshot_reads_alphabetically() {
        let runtime = ExtensionRuntime::new();
        runtime.set_flag_value("zeta", FlagValue::Bool(true));
        runtime.set_flag_value("alpha", FlagValue::Text("1".to_string()));
        let values = runtime.flag_values();
        assert_eq!(
            values.keys().cloned().collect::<Vec<_>>(),
            vec!["alpha".to_string(), "zeta".to_string()]
        );
    }
}
