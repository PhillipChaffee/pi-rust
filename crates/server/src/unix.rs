//! The Unix-domain socket listener transport, ported from upstream
//! `src/transports/unix/` (the `./unix` subpath export).
//!
//! Upstream binds an owned scratch socket, hard-links it onto the public
//! route, and cleans up by inode identity so a replaced path is never
//! unlinked; writes serialize through a promise tail behind a pending-byte
//! cap, and a close queues its final chunk behind every pending write. The
//! port restates the bind dance on `std::os::unix::net` plus tokio's
//! hard-link and metadata calls, and the connection on one driver task per
//! socket: reads and ordered writes multiplex over a command channel, `send`
//! rejects once queued-and-unwritten bytes would exceed the cap, and a close
//! shuts the write half down after the pending writes and waits — bounded by
//! the graceful window — for the peer to close.
//!
//! Windows is out of scope for this effort (map ticket "Decide the Rust
//! stack"), so the module compiles on Unix targets only.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::path::Path;
use std::rc::{Rc, Weak};
use std::time::Duration;

use pi_chord::future::{LocalBoxFuture, boxed};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::oneshot;

use crate::connection::{ByteConnection, ByteConnectionAcceptor, ByteConnectionHandler};
use crate::errors::Failure;
use crate::latch::CloseLatch as UnixCloseLatch;
use crate::latch::Latch;
use crate::listener::ServerListener;
use crate::server::Server;
use crate::types::{ServerHost, ServerOptions, ready};

/// The socket permissions a fresh listener grants, upstream's
/// `DEFAULT_SOCKET_MODE`.
const DEFAULT_SOCKET_MODE: u32 = 0o600;
/// The graceful-close window, upstream's
/// `DEFAULT_GRACEFUL_CLOSE_TIMEOUT_MS`.
const DEFAULT_GRACEFUL_CLOSE_TIMEOUT_MS: u64 = 5_000;
/// The framed-byte ceiling, upstream's `MAX_UINT32` option guard.
const MAX_UINT32: usize = u32::MAX as usize;
/// The upper bound a timeout may take, upstream's `MAX_TIMER_DELAY_MS`.
const MAX_TIMER_DELAY_MS: u64 = 2_147_483_647;
/// How long the stale-socket liveness probe waits before treating a busy
/// path as live, upstream's `SOCKET_PROBE_TIMEOUT_MS`.
const SOCKET_PROBE_TIMEOUT_MS: u64 = 1_000;

/// The options one Unix listener opens with, upstream's
/// `UnixListenerOptions`.
#[derive(Clone)]
pub struct UnixListenerOptions {
    /// The socket path to bind.
    pub path: String,
    /// Socket filesystem permissions; defaults to owner read/write only
    /// (0o600), upstream's `mode`.
    pub mode: Option<u32>,
    /// Maximum framed bytes queued per connection before a slow peer is
    /// disconnected, upstream's `maxPendingBytes`.
    pub max_pending_bytes: Option<usize>,
    /// The graceful-close window, upstream's `gracefulCloseTimeoutMs`.
    pub graceful_close_timeout_ms: Option<u64>,
    /// Used to derive and validate `max_pending_bytes`; must match the
    /// server when customized, upstream's `maxFrameLength`.
    pub max_frame_length: Option<usize>,
    /// Reports transport errors outside the connection lifecycle, upstream's
    /// `onError`.
    pub on_error: Option<crate::types::ErrorObserver>,
}

impl std::fmt::Debug for UnixListenerOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("UnixListenerOptions")
            .field("path", &self.path)
            .field("mode", &self.mode)
            .field("max_pending_bytes", &self.max_pending_bytes)
            .field("graceful_close_timeout_ms", &self.graceful_close_timeout_ms)
            .field("max_frame_length", &self.max_frame_length)
            .finish_non_exhaustive()
    }
}

/// The composed server-plus-Unix-listener options, upstream's
/// `UnixServerOptions` (`ServerOptions` minus `listeners`, plus
/// `UnixListenerOptions`).
#[derive(Clone)]
pub struct UnixServerOptions {
    /// Stable logical server identity, upstream's `serverId`.
    pub server_id: String,
    /// The socket path to bind, upstream's `path`.
    pub path: String,
    /// Socket filesystem permissions, upstream's `mode`.
    pub mode: Option<u32>,
    /// The per-connection pending-byte cap, upstream's `maxPendingBytes`.
    pub max_pending_bytes: Option<usize>,
    /// The graceful-close window, upstream's `gracefulCloseTimeoutMs`.
    pub graceful_close_timeout_ms: Option<u64>,
    /// The framed-byte ceiling, upstream's `maxFrameLength`.
    pub max_frame_length: Option<usize>,
    /// The handshake budget, upstream's `handshakeTimeoutMs`.
    pub handshake_timeout_ms: Option<u64>,
    /// Runs when the connected-presentation count changes, upstream's
    /// `onConnectionCountChanged`.
    pub on_connection_count_changed: Option<Rc<dyn Fn(usize)>>,
    /// Reports transport and server errors, upstream's `onError`.
    pub on_error: Option<crate::types::ErrorObserver>,
}

impl std::fmt::Debug for UnixServerOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("UnixServerOptions")
            .field("server_id", &self.server_id)
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

/// Derive the local Unix socket path for one logical server identity,
/// upstream's `getUnixSocketPath`.
///
/// # Errors
/// A non-canonical `server_id`, upstream's `TypeError`.
pub fn get_unix_socket_path(server_id: &str, server_directory: &str) -> Result<String, Failure> {
    if !pi_protocol::is_server_id(server_id) {
        return Err(Failure::message(
            "Unix serverId must be a canonical lowercase UUIDv4",
        ));
    }
    Ok(Path::new(server_directory)
        .join(format!("{server_id}.sock"))
        .to_string_lossy()
        .into_owned())
}

/// Creates fresh Unix-domain socket listeners, upstream's
/// `createUnixListener`.
///
/// # Errors
/// An empty path, an out-of-range mode or timeout, or a `maxPendingBytes`
/// below one maximum frame, upstream's `TypeError`s.
pub fn create_unix_listener(
    options: UnixListenerOptions,
) -> Result<Rc<dyn ServerListener>, Failure> {
    let resolved = resolve_unix_listener_options(&options)?;
    Ok(Rc::new_cyclic(|listener| UnixListener {
        self_rc: listener.clone(),
        path: resolved.path,
        mode: resolved.mode,
        max_pending_bytes: resolved.max_pending_bytes,
        graceful_close_timeout_ms: resolved.graceful_close_timeout_ms,
        on_error: options.on_error,
        connections: RefCell::new(HashMap::new()),
        acceptor: RefCell::new(None),
        accept_task: RefCell::new(None),
        socket_identity: RefCell::new(None),
        owned_bind_path: RefCell::new(None),
        closing: Cell::new(false),
        close_latch: RefCell::new(None),
    }))
}

/// Composes a [`Server`] with one Unix-domain socket listener, upstream's
/// `createUnixServer`.
///
/// # Errors
/// Whatever the listener's or the server's option validation raises.
pub fn create_unix_server<H: ServerHost + 'static>(
    host: Rc<H>,
    options: UnixServerOptions,
) -> Result<Server<H>, Failure> {
    let listener = create_unix_listener(UnixListenerOptions {
        path: options.path,
        mode: options.mode,
        max_pending_bytes: options.max_pending_bytes,
        graceful_close_timeout_ms: options.graceful_close_timeout_ms,
        max_frame_length: options.max_frame_length,
        on_error: options.on_error.clone(),
    })?;
    Server::new(
        host,
        ServerOptions {
            listeners: vec![listener],
            server_id: options.server_id,
            max_frame_length: options.max_frame_length,
            handshake_timeout_ms: options.handshake_timeout_ms,
            on_connection_count_changed: options.on_connection_count_changed,
            on_error: options.on_error,
        },
    )
}

struct ResolvedUnixListenerOptions {
    path: String,
    mode: u32,
    max_pending_bytes: usize,
    graceful_close_timeout_ms: u64,
}

fn resolve_unix_listener_options(
    options: &UnixListenerOptions,
) -> Result<ResolvedUnixListenerOptions, Failure> {
    if options.path.is_empty() {
        return Err(Failure::message(
            "Server Unix socket path must not be empty",
        ));
    }
    let mode = options.mode.unwrap_or(DEFAULT_SOCKET_MODE);
    if mode > 0o777 {
        return Err(Failure::message(
            "Server Unix socket mode must be an integer between 0 and 0o777",
        ));
    }
    let max_frame_length = options
        .max_frame_length
        .unwrap_or(pi_protocol::DEFAULT_MAX_FRAME_LENGTH);
    if max_frame_length == 0 || max_frame_length > MAX_UINT32 {
        return Err(Failure::message(format!(
            "Server maxFrameLength must be an integer between 1 and {MAX_UINT32}"
        )));
    }
    let max_pending_bytes = options.max_pending_bytes.unwrap_or(max_frame_length * 4);
    if max_pending_bytes < max_frame_length + 4 {
        return Err(Failure::message(
            "Server maxPendingBytes must be a safe integer at least maxFrameLength + 4",
        ));
    }
    let graceful_close_timeout_ms = options
        .graceful_close_timeout_ms
        .unwrap_or(DEFAULT_GRACEFUL_CLOSE_TIMEOUT_MS);
    if graceful_close_timeout_ms == 0 || graceful_close_timeout_ms > MAX_TIMER_DELAY_MS {
        return Err(Failure::message(format!(
            "Server gracefulCloseTimeoutMs must be an integer between 1 and {MAX_TIMER_DELAY_MS}"
        )));
    }
    Ok(ResolvedUnixListenerOptions {
        path: options.path.clone(),
        mode,
        max_pending_bytes,
        graceful_close_timeout_ms,
    })
}

/// The socket file identity the cleanup guards on, upstream's
/// `FileIdentity`.
#[derive(Clone, Copy)]
struct FileIdentity {
    dev: u64,
    ino: u64,
}

/// The Unix-domain socket listener, upstream's `UnixListener`.
struct UnixListener {
    /// The weak self-reference the trait methods resolve their spawned
    /// continuations through.
    self_rc: Weak<Self>,
    path: String,
    mode: u32,
    max_pending_bytes: usize,
    graceful_close_timeout_ms: u64,
    on_error: Option<crate::types::ErrorObserver>,
    connections: RefCell<HashMap<usize, Rc<UnixByteConnection>>>,
    acceptor: RefCell<Option<ByteConnectionAcceptor>>,
    accept_task: RefCell<Option<tokio::task::JoinHandle<()>>>,
    socket_identity: RefCell<Option<FileIdentity>>,
    owned_bind_path: RefCell<Option<String>>,
    closing: Cell<bool>,
    close_latch: RefCell<Option<UnixCloseLatch>>,
}

impl UnixListener {
    fn report_error(&self, error: &Failure) {
        if let Some(on_error) = &self.on_error {
            on_error(error);
        }
    }

    fn accept_socket(self: &Rc<Self>, socket: tokio::net::UnixStream) {
        if self.closing.get() {
            return;
        }
        let connection = Rc::new(UnixByteConnection::new(
            socket,
            self.graceful_close_timeout_ms,
            self.max_pending_bytes,
        ));
        let key = Rc::as_ptr(&connection) as usize;
        self.connections
            .borrow_mut()
            .insert(key, Rc::clone(&connection));
        let Some(accept) = self.acceptor.borrow().clone() else {
            self.connections.borrow_mut().remove(&key);
            return;
        };
        let connection_trait: Rc<dyn ByteConnection> = connection.clone();
        let handler = accept(connection_trait);
        let registry_on_close = {
            let listener = Rc::downgrade(self);
            let inner_close = Rc::clone(&handler.on_close);
            Rc::new(move || {
                if let Some(listener) = listener.upgrade() {
                    listener.connections.borrow_mut().remove(&key);
                }
                inner_close();
            })
        };
        let registry_handler = ByteConnectionHandler {
            on_data: Rc::clone(&handler.on_data),
            on_close: registry_on_close,
            on_error: Rc::clone(&handler.on_error),
        };
        connection.drive(registry_handler);
    }

    async fn close_server_and_cleanup(self: &Rc<Self>) -> Result<(), Failure> {
        if let Some(task) = self.accept_task.borrow_mut().take() {
            task.abort();
        }
        // Remove an unpublished startup bind path before the public route.
        let owned = self.owned_bind_path.borrow_mut().take();
        if let Some(owned) = owned {
            remove_path(&owned).await;
        }
        self.cleanup_owned_socket().await
    }

    async fn cleanup_owned_socket(self: &Rc<Self>) -> Result<(), Failure> {
        let Some(identity) = self.socket_identity.borrow_mut().take() else {
            return Ok(());
        };
        let current = match tokio::fs::symlink_metadata(&self.path).await {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(io_failure(&error)),
        };
        if !current.file_type().is_socket()
            || current.dev() != identity.dev
            || current.ino() != identity.ino
        {
            return Ok(());
        }
        let preserved = sibling_path(&self.path, &format!("cleanup-{}", short_uuid()));
        if let Err(error) = tokio::fs::rename(&self.path, &preserved).await {
            if error.kind() == std::io::ErrorKind::NotFound {
                return Ok(());
            }
            return Err(io_failure(&error));
        }
        let moved = tokio::fs::symlink_metadata(&preserved)
            .await
            .map_err(|error| io_failure(&error))?;
        if moved.file_type().is_socket()
            && moved.dev() == identity.dev
            && moved.ino() == identity.ino
        {
            remove_path(&preserved).await;
            return Ok(());
        }
        match tokio::fs::symlink_metadata(&self.path).await {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                tokio::fs::rename(&preserved, &self.path)
                    .await
                    .map_err(|error| io_failure(&error))?;
            }
            Err(error) => return Err(io_failure(&error)),
        }
        Err(Failure::message(format!(
            "Unix listener path changed during cleanup; preserved replacement at {preserved}"
        )))
    }

    async fn close_internal(self: &Rc<Self>) -> Result<(), Failure> {
        let server_closed = if self.accept_task.borrow().is_some() {
            let this = Rc::clone(self);
            tokio::task::spawn_local(async move { this.close_server_and_cleanup().await })
        } else {
            let this = Rc::clone(self);
            tokio::task::spawn_local(async move { this.cleanup_owned_socket().await })
        };
        let connections: Vec<Rc<UnixByteConnection>> =
            self.connections.borrow().values().cloned().collect();
        let mut handles = Vec::with_capacity(connections.len());
        for connection in connections {
            let connection = Rc::clone(&connection);
            handles.push(tokio::task::spawn_local(async move {
                connection.close(None).await
            }));
        }
        for handle in handles {
            let _ = handle.await;
        }
        let result = server_closed
            .await
            .unwrap_or_else(|join| Err(Failure::message(join.to_string())));
        let owned = self.owned_bind_path.borrow_mut().take();
        if let Some(owned) = owned {
            remove_path(&owned).await;
        }
        self.connections.borrow_mut().clear();
        result
    }
}

impl ServerListener for UnixListener {
    fn start(&self, accept: ByteConnectionAcceptor) -> LocalBoxFuture<Result<(), Failure>> {
        if self.accept_task.borrow().is_some() {
            return ready(Err(Failure::message("Unix listener is already started")));
        }
        if self.closing.get() {
            return ready(Err(Failure::message("Unix listener is closing or closed")));
        }
        *self.acceptor.borrow_mut() = Some(accept);
        let Some(this) = self.self_rc.upgrade() else {
            return ready(Err(Failure::message("Unix listener is gone")));
        };
        boxed(async move {
            let result = this.start_internal().await;
            if let Err(error) = result {
                // A cleanup failure replaces the original error, upstream's
                // catch block rethrowing the awaited cleanup rejection.
                return match this.close_server_and_cleanup().await {
                    Ok(()) => Err(error),
                    Err(cleanup) => Err(cleanup),
                };
            }
            Ok(())
        })
    }

    fn close(&self) -> LocalBoxFuture<Result<(), Failure>> {
        if let Some(existing) = self.close_latch.borrow().clone() {
            return boxed(async move { existing.wait().await });
        }
        self.closing.set(true);
        let latch = Rc::new(Latch::new());
        *self.close_latch.borrow_mut() = Some(Rc::clone(&latch));
        let Some(this) = self.self_rc.upgrade() else {
            return ready(Ok(()));
        };
        let spawn_latch = Rc::clone(&latch);
        tokio::task::spawn_local(async move {
            let result = this.close_internal().await;
            spawn_latch.settle(result);
        });
        boxed(async move { latch.wait().await })
    }
}

impl UnixListener {
    async fn start_internal(self: &Rc<Self>) -> Result<(), Failure> {
        let owned = get_owned_bind_path(&self.path);
        if let Some(parent) = Path::new(&self.path).parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|error| io_failure(&error))?;
            // Best effort: upstream passes mode 0o700 to the recursive
            // mkdir; the leaf parent is the only directory the socket's
            // reachability depends on.
            let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
        }
        remove_stale_socket(&self.path).await?;
        remove_stale_socket(&owned).await?;
        *self.owned_bind_path.borrow_mut() = Some(owned.clone());
        let std_listener =
            std::os::unix::net::UnixListener::bind(&owned).map_err(|error| io_failure(&error))?;
        std_listener
            .set_nonblocking(true)
            .map_err(|error| io_failure(&error))?;
        let metadata = std::fs::symlink_metadata(&owned).map_err(|error| io_failure(&error))?;
        if !metadata.file_type().is_socket() {
            return Err(Failure::message(format!(
                "Unix listener path is not a socket after binding: {owned}"
            )));
        }
        *self.socket_identity.borrow_mut() = Some(FileIdentity {
            dev: metadata.dev(),
            ino: metadata.ino(),
        });
        tokio::fs::hard_link(&owned, &self.path)
            .await
            .map_err(|error| io_failure(&error))?;
        set_socket_mode(&self.path, self.mode).await?;
        remove_path(&owned).await;
        let tokio_listener =
            tokio::net::UnixListener::from_std(std_listener).map_err(|error| io_failure(&error))?;
        let task = self.spawn_accept_task(tokio_listener);
        *self.accept_task.borrow_mut() = Some(task);
        Ok(())
    }

    fn spawn_accept_task(
        self: &Rc<Self>,
        listener: tokio::net::UnixListener,
    ) -> tokio::task::JoinHandle<()> {
        let this = Rc::clone(self);
        tokio::task::spawn_local(async move {
            loop {
                match listener.accept().await {
                    Ok((socket, _address)) => this.accept_socket(socket),
                    Err(error) => this.report_error(&io_failure(&error)),
                }
            }
        })
    }
}

/// One command the connection handle drives the socket task with.
enum UnixConnectionCommand {
    /// Write one chunk in arrival order, settling the sender's completion.
    Send {
        chunk: Vec<u8>,
        done: oneshot::Sender<Result<(), Failure>>,
    },
    /// End the socket behind every pending write, optionally emitting the
    /// final chunk last, and settle once the socket is closed.
    Close {
        final_chunk: Option<Vec<u8>>,
        done: oneshot::Sender<()>,
    },
}

/// One connection's shared accounting, upstream's `UnixByteConnection`
/// private fields.
struct UnixConnectionState {
    closed: Cell<bool>,
    closing: Cell<bool>,
    pending_bytes: Cell<usize>,
    max_pending_bytes: usize,
    graceful_close_timeout_ms: u64,
    close_latch: RefCell<Option<Rc<Latch<()>>>>,
}

/// One connected Unix-socket byte connection, upstream's `UnixByteConnection`
/// (`@internal`, exported for transport-level verification).
pub struct UnixByteConnection {
    state: Rc<UnixConnectionState>,
    commands: tokio::sync::mpsc::UnboundedSender<UnixConnectionCommand>,
    commands_rx: RefCell<Option<tokio::sync::mpsc::UnboundedReceiver<UnixConnectionCommand>>>,
    stream: RefCell<Option<tokio::net::UnixStream>>,
}

impl std::fmt::Debug for UnixByteConnection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("UnixByteConnection")
            .field("closed", &self.state.closed.get())
            .finish_non_exhaustive()
    }
}

impl UnixByteConnection {
    /// Builds one connection over `stream`, upstream's
    /// `new UnixByteConnection(socket, ...)`; the caller drives it through
    /// [`UnixByteConnection::drive`].
    pub fn new(
        stream: tokio::net::UnixStream,
        graceful_close_timeout_ms: u64,
        max_pending_bytes: usize,
    ) -> Self {
        let (commands, receiver) = tokio::sync::mpsc::unbounded_channel();
        Self {
            state: Rc::new(UnixConnectionState {
                closed: Cell::new(false),
                closing: Cell::new(false),
                pending_bytes: Cell::new(0),
                max_pending_bytes,
                graceful_close_timeout_ms,
                close_latch: RefCell::new(None),
            }),
            commands,
            commands_rx: RefCell::new(Some(receiver)),
            stream: RefCell::new(Some(stream)),
        }
    }

    /// Marks the transport closed and settles any pending close, upstream's
    /// `markClosed`.
    pub fn mark_closed(&self) {
        if self.state.closed.get() {
            return;
        }
        self.state.closed.set(true);
        self.state.closing.set(true);
        if let Some(latch) = self.state.close_latch.borrow_mut().take() {
            latch.settle(());
        }
    }

    /// Drives the socket: reads feed the handler, commands write in order,
    /// and the close command ends the write half behind every pending write.
    pub fn drive(self: &Rc<Self>, handler: ByteConnectionHandler) {
        let Some(stream) = self.stream.borrow_mut().take() else {
            return;
        };
        let Some(mut commands) = self.commands_rx.borrow_mut().take() else {
            return;
        };
        let (mut read_half, mut write_half) = stream.into_split();
        let state = Rc::clone(&self.state);
        let connection = Rc::clone(self);
        tokio::task::spawn_local(async move {
            let mut buffer = vec![0u8; 64 * 1024];
            'outer: loop {
                tokio::select! {
                    read = read_half.read(&mut buffer) => match read {
                        Ok(0) => {
                            connection.mark_closed();
                            (handler.on_close)();
                            break 'outer;
                        }
                        Ok(read) => (handler.on_data)(&buffer[..read]),
                        Err(error) => {
                            (handler.on_error)(&io_failure(&error));
                            connection.mark_closed();
                            (handler.on_close)();
                            break 'outer;
                        }
                    },
                    command = commands.recv() => match command {
                        None => break 'outer,
                        Some(UnixConnectionCommand::Send { chunk, done }) => {
                            let written = write_half.write_all(&chunk).await;
                            state
                                .pending_bytes
                                .set(state.pending_bytes.get() - chunk.len());
                            let failed = written.is_err();
                            let _ = done.send(written.map_err(|error| io_failure(&error)));
                            if failed {
                                // A failed write surfaces as a socket error,
                                // upstream's 'error' event firing before the
                                // 'close' the write callback rides.
                                (handler.on_error)(&Failure::message(
                                    "Unix connection closed during write",
                                ));
                                connection.mark_closed();
                                (handler.on_close)();
                                break 'outer;
                            }
                        }
                        Some(UnixConnectionCommand::Close { final_chunk, done }) => {
                            if let Some(final_chunk) = final_chunk {
                                let _ = write_half.write_all(&final_chunk).await;
                            }
                            let _ = write_half.shutdown().await;
                            let deadline = tokio::time::Instant::now()
                                + Duration::from_millis(state.graceful_close_timeout_ms);
                            loop {
                                tokio::select! {
                                    read = read_half.read(&mut buffer) => match read {
                                        Ok(0) | Err(_) => {
                                            connection.mark_closed();
                                            (handler.on_close)();
                                            let _ = done.send(());
                                            break 'outer;
                                        }
                                        Ok(read) => (handler.on_data)(&buffer[..read]),
                                    },
                                    () = tokio::time::sleep_until(deadline) => {
                                        connection.mark_closed();
                                        (handler.on_close)();
                                        let _ = done.send(());
                                        break 'outer;
                                    }
                                }
                            }
                        }
                    },
                }
            }
        });
    }
}

impl ByteConnection for UnixByteConnection {
    fn closed(&self) -> bool {
        self.state.closed.get()
    }

    fn send(&self, chunk: Vec<u8>) -> LocalBoxFuture<Result<(), Failure>> {
        if self.state.closed.get() || self.state.closing.get() {
            return ready(Err(Failure::message("Unix connection is closed")));
        }
        let pending = self.state.pending_bytes.get();
        if pending + chunk.len() > self.state.max_pending_bytes {
            return ready(Err(Failure::message(
                "Unix connection exceeded its pending byte limit",
            )));
        }
        self.state.pending_bytes.set(pending + chunk.len());
        let (done, receiver) = oneshot::channel();
        let _ = self
            .commands
            .send(UnixConnectionCommand::Send { chunk, done });
        boxed(async move {
            receiver
                .await
                .unwrap_or_else(|_| Err(Failure::message("Unix connection is closed")))
        })
    }

    fn close(&self, final_chunk: Option<Vec<u8>>) -> LocalBoxFuture<Result<(), Failure>> {
        if self.state.closed.get() {
            self.mark_closed();
            return ready(Ok(()));
        }
        if let Some(existing) = self.state.close_latch.borrow().clone() {
            return boxed(async move {
                existing.wait().await;
                Ok(())
            });
        }
        self.state.closing.set(true);
        let latch = Rc::new(Latch::new());
        *self.state.close_latch.borrow_mut() = Some(Rc::clone(&latch));
        let (done, receiver) = oneshot::channel();
        let _ = self
            .commands
            .send(UnixConnectionCommand::Close { final_chunk, done });
        boxed(async move {
            // The driver settles the command once the socket is closed, the
            // same resolve the socket's 'close' event gives upstream's
            // closePromise; mark_closed racing the driver settles the latch
            // first and the command lands right behind it.
            let _ = receiver.await;
            latch.wait().await;
            Ok(())
        })
    }
}

/// Derives the owned bind path one public route binds through, upstream's
/// `getOwnedBindPath`.
fn get_owned_bind_path(path: &str) -> String {
    let digest = Sha256::digest(path.as_bytes());
    let hex = format!("{digest:x}");
    sibling_path(path, &format!("bind-{}", &hex[..8]))
}

/// The sibling path one directory entry gets, upstream's
/// `join(dirname(path), name)`.
fn sibling_path(path: &str, name: &str) -> String {
    Path::new(path)
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(name)
        .to_string_lossy()
        .into_owned()
}

/// The six-character scratch suffix, upstream's
/// `randomUUID().slice(0, 6)`.
fn short_uuid() -> String {
    let uuid = uuid::Uuid::new_v4().to_string();
    uuid[..6].to_string()
}

/// Removes a stale socket at `path`, upstream's `removeStaleSocket`.
async fn remove_stale_socket(path: &str) -> Result<(), Failure> {
    let original = match tokio::fs::symlink_metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(io_failure(&error)),
    };
    if !original.file_type().is_socket() {
        return Err(Failure::message(format!(
            "Refusing to remove non-socket Unix listener path: {path}"
        )));
    }
    if is_socket_live(path).await? {
        return Err(Failure::message(format!(
            "Unix listener is already running: {path}"
        )));
    }
    let preserved = sibling_path(path, &format!("stale-{}", short_uuid()));
    if let Err(error) = tokio::fs::rename(path, &preserved).await {
        if error.kind() == std::io::ErrorKind::NotFound {
            return Ok(());
        }
        return Err(io_failure(&error));
    }
    let current = tokio::fs::symlink_metadata(&preserved)
        .await
        .map_err(|error| io_failure(&error))?;
    if !current.file_type().is_socket()
        || current.dev() != original.dev()
        || current.ino() != original.ino()
    {
        match tokio::fs::symlink_metadata(path).await {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                tokio::fs::rename(&preserved, path)
                    .await
                    .map_err(|error| io_failure(&error))?;
            }
            Err(error) => return Err(io_failure(&error)),
        }
        return Err(Failure::message(format!(
            "Unix listener path changed while checking for a stale socket: {path}"
        )));
    }
    remove_path(&preserved).await;
    Ok(())
}

/// Unlinks `path`, tolerating a missing entry, upstream's `removePath`.
async fn remove_path(path: &str) {
    let _ = tokio::fs::remove_file(path).await;
}

/// Whether a live endpoint answers at `path`, upstream's `isSocketLive`: a
/// connect inside the probe budget means live, the refused family means
/// stale, and a budget expiry assumes live like upstream's timer does.
async fn is_socket_live(path: &str) -> Result<bool, Failure> {
    match tokio::time::timeout(
        Duration::from_millis(SOCKET_PROBE_TIMEOUT_MS),
        tokio::net::UnixStream::connect(path),
    )
    .await
    {
        Ok(Ok(_stream)) => Ok(true),
        Ok(Err(error)) => match error.kind() {
            std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::NotFound
            | std::io::ErrorKind::BrokenPipe
            | std::io::ErrorKind::ConnectionReset => Ok(false),
            _ => Err(io_failure(&error)),
        },
        Err(_elapsed) => Ok(true),
    }
}

/// Applies the socket's filesystem permissions, upstream's `setSocketMode`
/// with its ENOSYS/ENOTSUP tolerance.
async fn set_socket_mode(path: &str, mode: u32) -> Result<(), Failure> {
    let permissions = std::fs::Permissions::from_mode(mode);
    match tokio::fs::set_permissions(path, permissions).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::Unsupported => Ok(()),
        Err(error) => Err(io_failure(&error)),
    }
}

/// Wraps an `io::Error` as an opaque failure carrying its message, the
/// errno-carrying `Error` instances upstream surfaces.
fn io_failure(error: &std::io::Error) -> Failure {
    Failure::message(error.to_string())
}
