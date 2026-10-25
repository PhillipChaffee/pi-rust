//! The session's shared state, handshake, and registration pass.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::{Mutex, mpsc};

use crate::extensions::runtime::{ExtensionError, HandlerContext};

use super::super::context::{HostCallError, UiCallHandler};
use super::super::link::Link;
use super::super::payload::events::EVENT_CHANNELS;
use super::super::{EXTENSION_PROTOCOL_VERSION, ExtensionIdentity, Frame, WireError};
use super::ErrorListener;
use super::{HostConnectOptions, SharedContext};

/// The abort a signal watcher reports to the running call scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SignalAbort {
    pub(super) call: u64,
    pub(super) signal: u64,
}

/// The shared session state the proxy closures capture.
pub(super) struct SessionCore<T> {
    pub(super) link: Mutex<Link<T>>,
    pub(super) context: SharedContext,
    pub(super) ui: Option<Arc<dyn UiCallHandler>>,
    error_listener: Option<ErrorListener>,
    path: String,
    next_call: AtomicU64,
    dead: AtomicBool,
    dead_message: StdMutex<Option<String>>,
    pub(super) abort_tx: mpsc::UnboundedSender<SignalAbort>,
    pub(super) abort_rx: Mutex<mpsc::UnboundedReceiver<SignalAbort>>,
}

impl<T> SessionCore<T>
where
    T: AsyncRead + AsyncWrite + Unpin,
{
    pub(super) fn new(io: T, options: &HostConnectOptions) -> Result<Self, WireError> {
        let (abort_tx, abort_rx) = mpsc::unbounded_channel();
        Ok(Self {
            link: Mutex::new(Link::new(io)?),
            context: options.context.clone(),
            ui: options.ui.clone(),
            error_listener: options.error_listener.clone(),
            path: options.path.clone(),
            next_call: AtomicU64::new(0),
            dead: AtomicBool::new(false),
            dead_message: StdMutex::new(None),
            abort_tx,
            abort_rx: Mutex::new(abort_rx),
        })
    }

    pub(super) async fn send(&self, frame: &Frame) -> Result<(), WireError> {
        self.link.lock().await.write_frame(frame).await
    }

    pub(super) async fn receive(&self) -> Result<Option<Frame>, WireError> {
        self.link.lock().await.read_frame().await
    }

    pub(super) async fn close(&self) -> Result<(), WireError> {
        self.link.lock().await.close().await
    }

    /// The handler context the owning session bound, absent before it
    /// builds its runner.
    pub(super) fn context(&self) -> Option<HandlerContext> {
        self.context
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub(super) fn is_dead(&self) -> bool {
        self.dead.load(Ordering::SeqCst)
    }

    pub(super) fn next_call(&self) -> u64 {
        self.next_call.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Mark the session dead once; the caller reports the failure, so a
    /// crash yields one diagnostic rather than one per layer.
    pub(super) fn fail(&self, _event: &str, error: HostCallError) -> HostCallError {
        if !self.dead.swap(true, Ordering::SeqCst) {
            self.set_dead_message(&format!("{}: {}", error.code(), error.message()));
        }
        error
    }

    pub(super) fn dead_failure(&self) -> HostCallError {
        let message = self
            .dead_message
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .unwrap_or_else(|| "the extension session is closed".to_string());
        HostCallError::new("extension_exited", message)
    }

    fn set_dead_message(&self, message: &str) {
        let mut stored = self
            .dead_message
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if stored.is_none() {
            *stored = Some(message.to_string());
        }
    }

    pub(super) fn report(&self, event: &str, error: &HostCallError, stack: Option<String>) {
        if let Some(listener) = &self.error_listener {
            listener(&ExtensionError {
                extension_path: self.path.clone(),
                event: event.to_string(),
                error: format!("{}: {}", error.code(), error.message()),
                stack,
            });
        }
    }
}

pub(super) async fn handshake<T>(core: &SessionCore<T>) -> Result<ExtensionIdentity, WireError>
where
    T: AsyncRead + AsyncWrite + Unpin,
{
    let Some(frame) = core.receive().await? else {
        return Err(WireError::new(
            "the extension closed the link before its hello",
        ));
    };
    let Frame::Hello {
        protocol_version,
        extension,
    } = frame
    else {
        let message = "the extension's first frame must be its hello";
        let _ = core
            .send(&Frame::HelloError {
                code: "handshake".to_string(),
                message: message.to_string(),
            })
            .await;
        return Err(WireError::new(message));
    };
    if protocol_version != EXTENSION_PROTOCOL_VERSION {
        let message = format!(
            "the extension speaks wire version {protocol_version}, the host {EXTENSION_PROTOCOL_VERSION}"
        );
        let _ = core
            .send(&Frame::HelloError {
                code: "protocol_version".to_string(),
                message: message.clone(),
            })
            .await;
        return Err(WireError::new(message));
    }
    core.send(&Frame::Welcome {
        protocol_version: EXTENSION_PROTOCOL_VERSION,
    })
    .await?;
    Ok(extension)
}

/// The declarations the registration pass collected before `ready`.
pub(super) struct Registrations {
    pub(super) events: Vec<String>,
    pub(super) tools: Vec<super::super::payload::descriptors::ToolDeclaration>,
    pub(super) commands: Vec<super::super::payload::descriptors::CommandDeclaration>,
    pub(super) flags: Vec<super::super::payload::descriptors::FlagDeclaration>,
    pub(super) providers: Vec<crate::provider_composer::ProviderConfigInput>,
    pub(super) provider_names: Vec<String>,
}

pub(super) async fn read_registrations<T>(core: &SessionCore<T>) -> Result<Registrations, WireError>
where
    T: AsyncRead + AsyncWrite + Unpin,
{
    let mut registrations = Registrations {
        events: Vec::new(),
        tools: Vec::new(),
        commands: Vec::new(),
        flags: Vec::new(),
        providers: Vec::new(),
        provider_names: Vec::new(),
    };
    loop {
        let Some(frame) = core.receive().await? else {
            return Err(WireError::new(
                "the extension closed the link before it was ready",
            ));
        };
        match frame {
            Frame::Ready => return Ok(registrations),
            Frame::RegisterEvent { channel } => {
                if !EVENT_CHANNELS.contains(&channel.as_str()) {
                    return Err(WireError::new(format!(
                        "the extension registered an unknown event channel {channel:?}"
                    )));
                }
                registrations.events.push(channel);
            }
            Frame::RegisterTool { tool } => registrations
                .tools
                .push(super::super::payload::descriptors::tool_declaration_from_wire(&tool)?),
            Frame::RegisterCommand { command } => registrations
                .commands
                .push(super::super::payload::descriptors::command_declaration_from_wire(&command)?),
            Frame::RegisterFlag { flag } => registrations
                .flags
                .push(super::super::payload::descriptors::flag_declaration_from_wire(&flag)?),
            Frame::RegisterProvider { provider } => {
                let (name, config) =
                    super::super::payload::descriptors::provider_from_wire(&provider)?;
                registrations.providers.push(config);
                registrations.provider_names.push(name);
            }
            other => {
                return Err(WireError::new(format!(
                    "the extension sent an unexpected frame before ready: {other:?}"
                )));
            }
        }
    }
}
