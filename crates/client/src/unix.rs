//! The Unix-domain socket transports and discovery, ported from upstream
//! `src/unix.ts` (the `./unix` subpath export).
//!
//! Upstream drives a Node `Socket` and serializes writes through a promise
//! tail with a pending-byte cap; the port restates the shape on tokio's
//! Unix stream: one local task per connection drives reads and ordered
//! writes, `send` rejects once the queued-and-unwritten bytes would
//! exceed the cap, and a close shuts the socket down, failing every
//! in-flight write. Windows is out of scope for this effort, so the module
//! compiles on Unix targets only.

use std::cell::{Cell, RefCell};
use std::os::unix::fs::FileTypeExt;
use std::path::Path;
use std::rc::Rc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use pi_chord::future::{LocalBoxFuture, boxed};
use pi_protocol::DEFAULT_MAX_FRAME_LENGTH;

use tokio::sync::oneshot;
use tokio::time::Duration;

use crate::client::Client;
use crate::errors::{ClientError, is_error_code};
use crate::types::ClientOptions;
use crate::{ByteTransport, ByteTransportFactory, ByteTransportHandlers};

/// The default per-connection and per-handshake budget, upstream's
/// `DEFAULT_DISCOVERY_TIMEOUT_MS`.
const DEFAULT_DISCOVERY_TIMEOUT_MS: u64 = 1_000;

/// The upper bound a timeout may take, upstream's `MAX_TIMER_DELAY_MS`
/// Node-timer bound restated for the tokio clock.
const MAX_TIMER_DELAY_MS: u64 = 2_147_483_647;

/// The suffix discovery matches server-addressed sockets by, upstream's
/// `UNIX_SOCKET_SUFFIX`.
const UNIX_SOCKET_SUFFIX: &str = ".sock";

/// The probe concurrency cap, upstream's
/// `MAX_CONCURRENT_DISCOVERY_PROBES`.
const MAX_CONCURRENT_DISCOVERY_PROBES: usize = 16;

/// The default pending-byte cap: four maximum frames, upstream's
/// `DEFAULT_MAX_FRAME_LENGTH * 4`.
const DEFAULT_MAX_PENDING_BYTES: usize = DEFAULT_MAX_FRAME_LENGTH * 4;

/// The options one Unix transport opens with, upstream's
/// `UnixTransportOptions`.
#[derive(Debug, Clone)]
pub struct UnixTransportOptions {
    /// The socket path to connect to.
    pub path: String,
    /// The pending-byte cap; defaults to four maximum frames.
    pub max_pending_bytes: Option<usize>,
}

/// One discovered server route, upstream's `UnixServerRoute`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnixServerRoute {
    /// The logical server the endpoint answered the handshake with.
    pub server_id: pi_protocol::ServerId,
    /// The socket path the endpoint listens on.
    pub path: String,
}

/// The options one discovery run probes with, upstream's
/// `DiscoverUnixServersOptions`.
#[derive(Debug, Clone)]
pub struct DiscoverUnixServersOptions {
    /// Directory containing server-addressed Unix sockets.
    pub directory: String,
    /// Maximum time for each connection and handshake; defaults to 1,000
    /// ms.
    pub timeout_ms: Option<u64>,
}

/// Discover reachable local servers by probing server-addressed Unix
/// sockets, in server-id order.
///
/// Stale endpoints, refused sockets, and protocol mismatches are omitted;
/// unexpected filesystem failures surface.
///
/// # Errors
/// Upstream throws a `TypeError` for a `timeoutMs` outside
/// `1..=MAX_TIMER_DELAY_MS`, and propagates filesystem failures other than
/// a missing directory; the port returns the same failures.
pub async fn discover_unix_servers(
    options: DiscoverUnixServersOptions,
) -> Result<Vec<UnixServerRoute>, ClientError> {
    let timeout_ms = options.timeout_ms.unwrap_or(DEFAULT_DISCOVERY_TIMEOUT_MS);
    if timeout_ms == 0 || timeout_ms > MAX_TIMER_DELAY_MS {
        return Err(ClientError::other(format!(
            "Unix discovery timeoutMs must be an integer between 1 and {MAX_TIMER_DELAY_MS}"
        )));
    }

    let names = match tokio::fs::read_dir(&options.directory).await {
        Ok(mut entries) => {
            let mut names = Vec::new();
            loop {
                match entries.next_entry().await {
                    Ok(Some(entry)) => names.push(entry.file_name().to_string_lossy().into_owned()),
                    Ok(None) => break,
                    Err(error) => return Err(io_error(&error)),
                }
            }
            names
        }
        Err(error) if is_enoent(&error) => return Ok(Vec::new()),
        Err(error) => return Err(io_error(&error)),
    };

    let candidates: Rc<Vec<UnixServerRoute>> = Rc::new(
        names
            .into_iter()
            .filter_map(|name| {
                name.strip_suffix(UNIX_SOCKET_SUFFIX).and_then(|server_id| {
                    Some(UnixServerRoute {
                        server_id: pi_protocol::ServerId::new(server_id)?,
                        path: Path::new(&options.directory)
                            .join(&name)
                            .to_string_lossy()
                            .into_owned(),
                    })
                })
            })
            .collect(),
    );
    let next = Rc::new(Cell::new(0usize));
    let routes_rc = Rc::new(RefCell::new(Vec::new()));
    let failure_rc: Rc<RefCell<Option<ClientError>>> = Rc::new(RefCell::new(None));
    let failure = Rc::clone(&failure_rc);
    let workers = MAX_CONCURRENT_DISCOVERY_PROBES.min(candidates.len());
    let mut probes = Vec::with_capacity(workers);
    for _ in 0..workers {
        probes.push(tokio::task::spawn_local({
            let next = Rc::clone(&next);
            let routes = Rc::clone(&routes_rc);
            let failure = Rc::clone(&failure);
            let candidates = Rc::clone(&candidates);
            async move {
                loop {
                    if failure.borrow().is_some() {
                        return;
                    }
                    let Some(candidate) = candidates.get(next.get()).cloned() else {
                        return;
                    };
                    next.set(next.get() + 1);
                    match tokio::fs::symlink_metadata(&candidate.path).await {
                        // A socket can disappear between readdir and lstat
                        // during normal server shutdown.
                        Ok(metadata) if metadata.file_type().is_socket() => {}
                        Ok(_) => continue,
                        Err(error) if is_enoent(&error) => continue,
                        Err(error) => {
                            if failure.borrow().is_none() {
                                *failure.borrow_mut() = Some(io_error(&error));
                            }
                            return;
                        }
                    }
                    match probe_unix_server(&candidate, timeout_ms).await {
                        Ok(Some(route)) => routes.borrow_mut().push(route),
                        Ok(None) => {}
                        Err(error) => {
                            if failure.borrow().is_none() {
                                *failure.borrow_mut() = Some(error);
                            }
                        }
                    }
                }
            }
        }));
    }
    for probe in probes {
        if let Err(error) = probe.await
            && failure.borrow().is_none()
        {
            *failure.borrow_mut() = Some(ClientError::other(error.to_string()));
        }
    }
    if let Some(error) = failure_rc.borrow_mut().take() {
        return Err(error);
    }
    let mut routes = routes_rc.borrow_mut().clone();
    routes.sort_by(|left, right| left.server_id.as_str().cmp(right.server_id.as_str()));
    Ok(routes)
}

/// Creates fresh Unix-domain socket transports for client connection
/// attempts, upstream's `createUnixTransportFactory`.
///
/// # Errors
/// Upstream throws a `TypeError` for an empty path or a non-positive
/// `maxPendingBytes`; the port returns the same failures.
pub fn create_unix_transport_factory(
    options: UnixTransportOptions,
) -> Result<ByteTransportFactory, ClientError> {
    let max_pending_bytes = validate_unix_transport_options(&options)?;
    let path = options.path;
    Ok(Rc::new(move |handlers| {
        connect_unix_socket(path.clone(), max_pending_bytes, handlers)
    }))
}

fn validate_unix_transport_options(options: &UnixTransportOptions) -> Result<usize, ClientError> {
    if options.path.is_empty() {
        return Err(ClientError::other("Unix transport path must not be empty"));
    }
    let max_pending_bytes = options
        .max_pending_bytes
        .unwrap_or(DEFAULT_MAX_PENDING_BYTES);
    if max_pending_bytes == 0 {
        return Err(ClientError::other(
            "Unix transport maxPendingBytes must be a positive safe integer",
        ));
    }
    Ok(max_pending_bytes)
}

/// One command the transport handle drives the reader task with.
enum UnixCommand {
    /// Write one chunk in arrival order, settling the sender's completion.
    Send {
        chunk: Vec<u8>,
        done: oneshot::Sender<Result<(), ClientError>>,
    },
    /// Shut the socket down and end the reader task.
    Close,
}

/// The transport handle's shared accounting, upstream's
/// `UnixByteTransport` private fields.
struct UnixTransportState {
    closed: Cell<bool>,
    pending_bytes: Cell<usize>,
    max_pending_bytes: usize,
}

/// One connected Unix-socket transport, upstream's `UnixByteTransport`.
struct UnixByteTransport {
    state: Rc<UnixTransportState>,
    commands: tokio::sync::mpsc::UnboundedSender<UnixCommand>,
}

impl ByteTransport for UnixByteTransport {
    fn send(&self, chunk: Vec<u8>) -> LocalBoxFuture<Result<(), ClientError>> {
        if self.state.closed.get() {
            return boxed(std::future::ready(Err(ClientError::other(
                "Unix transport is closed",
            ))));
        }
        let pending = self.state.pending_bytes.get();
        if pending + chunk.len() > self.state.max_pending_bytes {
            return boxed(std::future::ready(Err(ClientError::other(
                "Unix transport exceeded its pending byte limit",
            ))));
        }
        self.state.pending_bytes.set(pending + chunk.len());
        let (done, receiver) = oneshot::channel();
        let _ = self.commands.send(UnixCommand::Send { chunk, done });
        boxed(async move {
            receiver
                .await
                .unwrap_or_else(|_| Err(ClientError::other("Unix transport closed during write")))
        })
    }

    fn close(&self) {
        if self.state.closed.get() {
            return;
        }
        self.state.closed.set(true);
        let _ = self.commands.send(UnixCommand::Close);
    }
}

/// Connects one Unix socket and starts its reader task, upstream's
/// `connectUnixSocket`.
fn connect_unix_socket(
    path: String,
    max_pending_bytes: usize,
    handlers: ByteTransportHandlers,
) -> LocalBoxFuture<Result<Rc<dyn ByteTransport>, ClientError>> {
    boxed(async move {
        let stream = match tokio::net::UnixStream::connect(&path).await {
            Ok(stream) => stream,
            Err(error) => return Err(io_error(&error)),
        };
        let state = Rc::new(UnixTransportState {
            closed: Cell::new(false),
            pending_bytes: Cell::new(0),
            max_pending_bytes,
        });
        let (commands, mut commands_rx) = tokio::sync::mpsc::unbounded_channel();
        let (mut read_half, mut write_half) = stream.into_split();
        tokio::task::spawn_local({
            let state = Rc::clone(&state);
            async move {
                let mut chunk = vec![0u8; 64 * 1024];
                loop {
                    tokio::select! {
                        read = read_half.read(&mut chunk) => match read {
                            Ok(0) => {
                                if !state.closed.get() {
                                    (handlers.on_close)();
                                }
                                break;
                            }
                            Ok(read) => (handlers.on_data)(&chunk[..read]),
                            Err(error) => {
                                if !state.closed.get() {
                                    (handlers.on_error)(&io_error(&error));
                                }
                                break;
                            }
                        },
                        command = commands_rx.recv() => match command {
                            None => break,
                            Some(UnixCommand::Close) => {
                                state.closed.set(true);
                                let _ = write_half.shutdown().await;
                                while let Ok(command) = commands_rx.try_recv() {
                                    fail_write(command, "Unix transport closed during write");
                                }
                                break;
                            }
                            Some(command @ UnixCommand::Send { .. }) => {
                                write_command(command, &state, &mut write_half).await;
                            }
                        },
                    }
                }
            }
        });
        let transport: Rc<dyn ByteTransport> = Rc::new(UnixByteTransport { state, commands });
        Ok(transport)
    })
}

/// Writes one queued chunk and settles its completion, decrementing the
/// pending-byte accounting either way.
async fn write_command(
    command: UnixCommand,
    state: &Rc<UnixTransportState>,
    write_half: &mut tokio::net::unix::OwnedWriteHalf,
) {
    let UnixCommand::Send { chunk, done } = command else {
        return;
    };
    let settled = match write_half.write_all(&chunk).await {
        Ok(()) => Ok(()),
        Err(error) => Err(io_error(&error)),
    };
    state
        .pending_bytes
        .set(state.pending_bytes.get() - chunk.len());
    let _ = done.send(settled);
}

fn fail_write(command: UnixCommand, message: &str) {
    if let UnixCommand::Send { done, .. } = command {
        let _ = done.send(Err(ClientError::other(message)));
    }
}

/// Probes one candidate route with a full handshake, upstream's
/// `probeUnixServer`: reachable servers answer, stale or non-protocol
/// endpoints are omitted, and everything else surfaces.
async fn probe_unix_server(
    route: &UnixServerRoute,
    timeout_ms: u64,
) -> Result<Option<UnixServerRoute>, ClientError> {
    let factory = create_unix_transport_factory(UnixTransportOptions {
        path: route.path.clone(),
        max_pending_bytes: None,
    })?;
    let client = Client::new(ClientOptions {
        server_id: route.server_id.as_str().to_string(),
        transport_factory: factory,
        max_frame_length: None,
        on_listener_error: None,
    })?;
    let outcome = tokio::select! {
        result = client.connect() => result,
        () = tokio::time::sleep(Duration::from_millis(timeout_ms)) => {
            client.dispose();
            // Missing/refused sockets are stale or shutting down. Protocol
            // failures mean the endpoint is not the advertised server.
            // Both are safe to omit.
            return Ok(None);
        }
    };
    client.dispose();
    match outcome {
        Ok(_) => Ok(Some(route.clone())),
        Err(error) if probe_omittable(&error) => Ok(None),
        Err(error) => Err(error),
    }
}

/// Whether a probe failure is safe to omit, upstream's omission list:
/// timeouts, protocol validation failures, version rejections, clean
/// transport closes, and the connection-refusal errno family.
fn probe_omittable(error: &ClientError) -> bool {
    match error {
        ClientError::Server(failure) => failure.code == "version",
        ClientError::Protocol(_) | ClientError::Disconnected { cause: None, .. } => true,
        ClientError::Disconnected {
            cause: Some(cause), ..
        } => probe_omittable(cause),
        ClientError::Other { .. } | ClientError::Disposed => {
            ["ENOENT", "ECONNREFUSED", "ECONNRESET", "EPIPE", "ETIMEDOUT"]
                .iter()
                .any(|code| is_error_code(error, code))
        }
    }
}

fn is_enoent(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::NotFound
}

fn io_error(error: &std::io::Error) -> ClientError {
    ClientError::from_io(error)
}
