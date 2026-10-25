//! The spawn gate: verified child processes over piped stdio ([ADR 0010]).
//!
//! [`ExtensionProcess::spawn`] verifies an installed package's receipt,
//! launches the entry binary with the session cwd, and hands its stdio to
//! [`HostSession::connect`] under a handshake deadline.
//! [`ExtensionLoader`] owns the loader cache: one live process per resolved
//! path, reused across a reload's pre-trust and final passes, retired when a
//! path leaves the set, and invalidated at reload start.
//!
//! [ADR 0010]: ../../../../../docs/adr/0010-extension-host-wire-protocol.md

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::Join;
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::time::timeout;

use crate::extensions::api::{ExtensionRuntime, PendingProviderRegistration};
use crate::extensions::runner::ExtensionErrorListener;
use crate::extensions::runtime::{Extension, ExtensionError, HandlerContext};
use crate::package_manager::{INSTALL_RECEIPT_FILE, InstallReceipt, hash_package_files};
use crate::resource_loader::{
    ExtensionEntry, ExtensionLoadError, ExtensionLoadResult, resolve_default,
    resolve_extension_load_path,
};
use crate::source_info::{SyntheticSourceOptions, create_synthetic_source_info};

use super::EXTENSION_PROTOCOL_VERSION;
use super::session::{ErrorListener, HostConnectOptions, HostSession, SharedContext};

#[cfg(test)]
mod tests;

/// How long the child has to finish its handshake.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

/// The graceful-close budget; the child is killed once it passes.
pub const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);

/// One child session over the process's piped stdio.
type ChildSession = HostSession<Join<ChildStdout, ChildStdin>>;

/// A spawn-gate failure; the loader reports the message as a load error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpawnError {
    message: String,
}

impl SpawnError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The failure message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for SpawnError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for SpawnError {}

/// The ingredients one spawn needs beyond the connect options.
pub struct SpawnOptions {
    /// The child's working directory, the session cwd.
    pub cwd: String,
    /// The handshake and registration options.
    pub connect: HostConnectOptions,
    handshake_timeout: Duration,
    shutdown_grace: Duration,
}

impl SpawnOptions {
    /// The options at the production deadlines.
    #[must_use]
    pub const fn new(cwd: String, connect: HostConnectOptions) -> Self {
        Self {
            cwd,
            connect,
            handshake_timeout: HANDSHAKE_TIMEOUT,
            shutdown_grace: SHUTDOWN_GRACE,
        }
    }
}

impl fmt::Debug for SpawnOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SpawnOptions")
            .field("cwd", &self.cwd)
            .field("connect", &self.connect)
            .finish_non_exhaustive()
    }
}

/// One live extension child and the session over its stdio.
struct ExtensionProcess {
    child: Child,
    session: ChildSession,
    providers: Vec<PendingProviderRegistration>,
    grace: Duration,
}

impl ExtensionProcess {
    async fn spawn(options: SpawnOptions) -> Result<Self, SpawnError> {
        verify_install_receipt(Path::new(&options.connect.resolved_path))?;
        let mut child = Command::new(&options.connect.resolved_path)
            .current_dir(&options.cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // The child's diagnostics stream through verbatim; the wire
            // never parses stderr.
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| {
                SpawnError::new(format!(
                    "failed to spawn extension {}: {error}",
                    options.connect.resolved_path
                ))
            })?;
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            return Err(SpawnError::new(
                "the extension child exposes no piped stdio",
            ));
        };
        let mut session = match timeout(
            options.handshake_timeout,
            HostSession::connect(tokio::io::join(stdout, stdin), options.connect),
        )
        .await
        {
            Ok(Ok(session)) => session,
            Ok(Err(error)) => return Err(SpawnError::new(error.message().to_string())),
            Err(_) => {
                return Err(SpawnError::new(format!(
                    "the extension did not complete its handshake within {}ms",
                    options.handshake_timeout.as_millis()
                )));
            }
        };
        let providers = session.take_provider_registrations();
        Ok(Self {
            child,
            session,
            providers,
            grace: options.shutdown_grace,
        })
    }

    fn extension(&self) -> Arc<Extension> {
        self.session.extension()
    }

    fn take_providers(&mut self) -> Vec<PendingProviderRegistration> {
        std::mem::take(&mut self.providers)
    }

    /// Close stdin (the child sees end of stream), wait out the grace, then
    /// kill. Drop alone kills immediately.
    async fn shutdown(&mut self) {
        let _ = self.session.close().await;
        if timeout(self.grace, self.child.wait()).await.is_err() {
            let _ = self.child.start_kill();
            let _ = self.child.wait().await;
        }
    }
}

/// Verify an installed package's receipt before spawning its extension.
///
/// Discovery walks up from the entry binary: the nearest ancestor holding
/// `pi-package-install.json` is the package root an install wrote. A path
/// with no such ancestor is an explicit local path and loads in place
/// without a receipt. A receipt fails closed when it does not parse, records
/// no protocol version, records a different version, or its per-file sha256
/// map no longer matches the tree.
fn verify_install_receipt(extension_path: &Path) -> Result<(), SpawnError> {
    let Some(root) = install_root(extension_path) else {
        return Ok(());
    };
    let receipt_path = root.join(INSTALL_RECEIPT_FILE);
    let content = std::fs::read_to_string(&receipt_path).map_err(|error| {
        SpawnError::new(format!(
            "failed to read the install receipt at {}: {error}",
            receipt_path.display()
        ))
    })?;
    let value: serde_json::Value = serde_json::from_str(&content).map_err(|error| {
        SpawnError::new(format!(
            "the install receipt at {} is not JSON: {error}",
            receipt_path.display()
        ))
    })?;
    let receipt = InstallReceipt::from_value(&value).map_err(|error| {
        SpawnError::new(format!(
            "the install receipt at {} is invalid: {}",
            receipt_path.display(),
            error
        ))
    })?;
    let recorded = receipt.protocol_version.ok_or_else(|| {
        SpawnError::new(format!(
            "the install receipt at {} records no extension protocol version",
            receipt_path.display()
        ))
    })?;
    if recorded != EXTENSION_PROTOCOL_VERSION {
        return Err(SpawnError::new(format!(
            "the install receipt records extension protocol version {recorded}, the host speaks {EXTENSION_PROTOCOL_VERSION}"
        )));
    }
    let actual = hash_package_files(&root);
    for (relative, digest) in &receipt.files {
        match actual.get(relative) {
            None => {
                return Err(SpawnError::new(format!(
                    "installed extension file is missing: {relative}"
                )));
            }
            Some(found) if found != digest => {
                return Err(SpawnError::new(format!(
                    "installed extension file changed since install: {relative}"
                )));
            }
            Some(_) => {}
        }
    }
    if let Some(relative) = actual
        .keys()
        .find(|relative| !receipt.files.contains_key(*relative))
    {
        return Err(SpawnError::new(format!(
            "installed extension carries a file the receipt does not list: {relative}"
        )));
    }
    Ok(())
}

/// The nearest ancestor of the entry binary holding an install receipt.
fn install_root(extension_path: &Path) -> Option<PathBuf> {
    let mut current = extension_path.parent()?;
    loop {
        if current.join(INSTALL_RECEIPT_FILE).is_file() {
            return Some(current.to_path_buf());
        }
        current = current.parent()?;
    }
}

/// The loader-owned session cache, one live child per resolved path.
pub struct ExtensionLoader {
    context: SharedContext,
    listener: Arc<Mutex<Option<ErrorListener>>>,
    cache: Mutex<Cache>,
    handshake_timeout: Duration,
    shutdown_grace: Duration,
}

#[derive(Default)]
struct Cache {
    cwd: Option<String>,
    sessions: HashMap<String, CachedSession>,
}

struct CachedSession {
    path: String,
    process: ExtensionProcess,
}

impl fmt::Debug for ExtensionLoader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sessions = lock(&self.cache).sessions.len();
        formatter
            .debug_struct("ExtensionLoader")
            .field("sessions", &sessions)
            .finish_non_exhaustive()
    }
}

impl Default for ExtensionLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtensionLoader {
    /// The empty cache with no bound context and no error listener.
    #[must_use]
    pub fn new() -> Self {
        Self {
            context: SharedContext::default(),
            listener: Arc::new(Mutex::new(None)),
            cache: Mutex::new(Cache::default()),
            handshake_timeout: HANDSHAKE_TIMEOUT,
            shutdown_grace: SHUTDOWN_GRACE,
        }
    }

    /// Load the requested paths, spawning what the cache lacks.
    ///
    /// Provider registrations queue once, into the runtime of the pass that
    /// spawned the session: a later pass reuses the same session and the
    /// loader hands it the pre-trust runtime.
    pub async fn load(
        &self,
        paths: &[String],
        cwd: &str,
        runtime: Option<ExtensionRuntime>,
    ) -> ExtensionLoadResult {
        let runtime = runtime.unwrap_or_default();
        let resolved_cwd = resolve_default(cwd);
        let entries: Vec<(String, String)> = paths
            .iter()
            .map(|path| {
                (
                    path.clone(),
                    resolve_extension_load_path(path, &resolved_cwd),
                )
            })
            .collect();

        let retired = {
            let mut cache = lock(&self.cache);
            if cache.cwd.as_deref() == Some(resolved_cwd.as_str()) {
                Vec::new()
            } else {
                cache.cwd = Some(resolved_cwd.clone());
                cache.sessions.drain().map(|(_, session)| session).collect()
            }
        };
        self.retire(retired).await;

        let mut failures: HashMap<String, String> = HashMap::new();
        for (path, resolved) in &entries {
            if lock(&self.cache).sessions.contains_key(resolved) {
                continue;
            }
            match self.spawn(path, resolved, &resolved_cwd).await {
                Ok(mut process) => {
                    for registration in process.take_providers() {
                        runtime.register_provider(registration);
                    }
                    lock(&self.cache).sessions.insert(
                        resolved.clone(),
                        CachedSession {
                            path: path.clone(),
                            process,
                        },
                    );
                }
                Err(error) => {
                    failures.insert(resolved.clone(), error.message().to_string());
                }
            }
        }

        let mut extensions = Vec::new();
        let mut errors = Vec::new();
        let cache = lock(&self.cache);
        for (path, resolved) in &entries {
            if let Some(cached) = cache.sessions.get(resolved) {
                let record = cached.process.extension();
                extensions.push(ExtensionEntry {
                    path: cached.path.clone(),
                    resolved_path: resolved.clone(),
                    hidden: false,
                    source_info: Some(record.source_info.clone()),
                    record: Some(record),
                });
            } else {
                errors.push(ExtensionLoadError {
                    path: path.clone(),
                    error: failures
                        .get(resolved)
                        .cloned()
                        .unwrap_or_else(|| "the extension did not load".to_string()),
                });
            }
        }
        ExtensionLoadResult {
            extensions,
            errors,
            runtime,
        }
    }

    /// Shut down every cached session and drop the cached cwd, the reload
    /// boundary.
    pub async fn invalidate(&self) {
        let retired = {
            let mut cache = lock(&self.cache);
            cache.cwd = None;
            cache.sessions.drain().map(|(_, session)| session).collect()
        };
        self.retire(retired).await;
    }

    /// Shut down the sessions whose paths left the final set.
    pub async fn retain(&self, paths: &[String], cwd: &str) {
        let resolved_cwd = resolve_default(cwd);
        let wanted: HashSet<String> = paths
            .iter()
            .map(|path| resolve_extension_load_path(path, &resolved_cwd))
            .collect();
        let retired = {
            let mut cache = lock(&self.cache);
            if cache.cwd.as_deref() == Some(resolved_cwd.as_str()) {
                cache
                    .sessions
                    .extract_if(|resolved, _| !wanted.contains(resolved))
                    .map(|(_, session)| session)
                    .collect()
            } else {
                Vec::new()
            }
        };
        self.retire(retired).await;
    }

    /// Bind the running session's context and dispatch-error listener; the
    /// spawned sessions read both through the shared slots.
    pub fn bind(&self, context: HandlerContext, listener: Option<ExtensionErrorListener>) {
        *lock(&self.context) = Some(context);
        *lock(&self.listener) = listener;
    }

    /// The relay the spawned sessions report through: whatever listener the
    /// owning runner binds, or nothing before it exists.
    fn relay(&self) -> ErrorListener {
        let slot = Arc::clone(&self.listener);
        Arc::new(move |error: &ExtensionError| {
            let listener = lock(&slot).clone();
            if let Some(listener) = listener {
                listener(error);
            }
        })
    }

    async fn spawn(
        &self,
        path: &str,
        resolved: &str,
        cwd: &str,
    ) -> Result<ExtensionProcess, SpawnError> {
        let mut options = SpawnOptions::new(
            cwd.to_string(),
            HostConnectOptions {
                path: path.to_string(),
                resolved_path: resolved.to_string(),
                source_info: create_synthetic_source_info(
                    resolved,
                    &SyntheticSourceOptions {
                        source: "extension".to_string(),
                        ..SyntheticSourceOptions::default()
                    },
                ),
                context: Arc::clone(&self.context),
                ui: None,
                error_listener: Some(self.relay()),
            },
        );
        options.handshake_timeout = self.handshake_timeout;
        options.shutdown_grace = self.shutdown_grace;
        ExtensionProcess::spawn(options).await
    }

    async fn retire(&self, sessions: Vec<CachedSession>) {
        for mut cached in sessions {
            cached.process.shutdown().await;
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
