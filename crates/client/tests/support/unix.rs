//! Unix-socket fixtures for the client's Unix suites: a minimal handshake
//! server standing in for the `pi-server` crate (upstream's suite imports
//! `../../server/src`, which this port reaches through an equivalent
//! in-crate stub), silent-socket listeners for the probe-cap cases, and the
//! raw per-socket servers the transport tests drive.

#![cfg(unix)]
#![allow(
    clippy::panic,
    reason = "test assertions panic at the failing case only; the restriction lint targets production code"
)]
#![allow(
    clippy::expect_used,
    reason = "test helpers settle results the case's own assertions would reject"
)]
#![allow(
    dead_code,
    reason = "the fixtures are shared across the suites; each test binary links its own slice"
)]
#![allow(
    unreachable_pub,
    reason = "the fixture module is shared across test binaries, never exported"
)]

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::rc::Rc;

use pi_protocol::{
    ClientMessage, ClientMessageDecoder, FrameDecoderOptions, ServerHello, ServerId, ServerMessage,
};

use pi_chord::future::{LocalBoxFuture, boxed};
use pi_client::{ByteTransport, ClientError};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::net::UnixStream;

/// Builds a canonical fixture server id from a small ordinal, upstream's
/// `serverId(value)` helper.
#[must_use]
pub fn server_id(value: u64) -> String {
    format!("00000000-0000-4000-8000-{value:012x}")
}

/// Reads one decoded batch of client messages off the socket, `None` once
/// the peer closed or the stream failed, the read loop every fixture
/// server shares.
async fn read_batch(
    read_half: &mut tokio::net::unix::OwnedReadHalf,
    decoder: &mut ClientMessageDecoder,
    chunk: &mut [u8],
) -> Option<Vec<ClientMessage>> {
    let read = match read_half.read(chunk).await {
        Ok(0) | Err(_) => return None,
        Ok(read) => read,
    };
    decoder.push(&chunk[..read]).ok()
}

/// One running handshake-answering server, upstream's `startServer` helper:
/// a listener bound at the file id's path that answers handshakes with the
/// reported identity.
pub struct TestUnixServer {
    connections: Rc<Cell<usize>>,
}

impl TestUnixServer {
    /// Binds the socket and starts the accept loop, upstream's
    /// `startServer`.
    pub fn start(directory: &Path, file_server_id: &str, reported_server_id: &str) -> Rc<Self> {
        let path = directory.join(format!("{file_server_id}.sock"));
        let listener = UnixListener::bind(&path).expect("the fixture socket binds");
        let connections = Rc::new(Cell::new(0));
        tokio::task::spawn_local({
            let reported_server_id = reported_server_id.to_string();
            let connections = Rc::clone(&connections);
            async move {
                loop {
                    let Ok((stream, _)) = listener.accept().await else {
                        break;
                    };
                    connections.set(connections.get() + 1);
                    tokio::task::spawn_local(serve_handshake(
                        stream,
                        reported_server_id.clone(),
                        Rc::clone(&connections),
                    ));
                }
            }
        });
        Rc::new(Self { connections })
    }

    /// The live connection count, for fixture assertions.
    #[must_use]
    pub fn connections(&self) -> usize {
        self.connections.get()
    }
}

/// Answers one client connection's handshake with the reported identity,
/// the surface the discovery probes exercise.
async fn serve_handshake(stream: UnixStream, server_id: String, connections: Rc<Cell<usize>>) {
    let (mut read_half, mut write_half) = stream.into_split();
    let Ok(mut decoder) = ClientMessageDecoder::new(FrameDecoderOptions::default()) else {
        return;
    };
    let mut chunk = vec![0u8; 64 * 1024];
    let mut handshaken = false;
    while let Some(messages) = read_batch(&mut read_half, &mut decoder, &mut chunk).await {
        for message in messages {
            if matches!(message, ClientMessage::Hello(_)) && !handshaken {
                handshaken = true;
                let frame = super::server_frame(&ServerMessage::Hello(ServerHello {
                    server_id: ServerId::new(&server_id).expect("fixture ids are canonical"),
                }));
                if write_half.write_all(&frame).await.is_err() {
                    connections.set(connections.get().saturating_sub(1));
                    return;
                }
            }
        }
    }
    connections.set(connections.get().saturating_sub(1));
}

/// Answers every client connection's handshake with a typed hello error,
/// the endpoint shape the probe-omission boundary cases drive: version
/// rejections are omitted, every other code surfaces.
pub struct HelloErrorServer {}

impl HelloErrorServer {
    /// Binds the socket and starts the hello-error accept loop.
    pub fn start(directory: &Path, file_server_id: &str, code: &str, message: &str) -> Rc<Self> {
        Self::start_delayed(directory, file_server_id, code, message, 0)
    }

    /// Binds the socket and starts the hello-error accept loop, holding each
    /// connection for `delay_ms` before answering.
    pub fn start_delayed(
        directory: &Path,
        file_server_id: &str,
        code: &str,
        message: &str,
        delay_ms: u64,
    ) -> Rc<Self> {
        let path = directory.join(format!("{file_server_id}.sock"));
        let listener = UnixListener::bind(&path).expect("the fixture socket binds");
        let code = code.to_string();
        let message = message.to_string();
        tokio::task::spawn_local(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                tokio::task::spawn_local(serve_hello_error(
                    stream,
                    code.clone(),
                    message.clone(),
                    delay_ms,
                ));
            }
        });
        Rc::new(Self {})
    }
}

/// Answers one connection's handshake with the typed hello error, the
/// rejection the discovery probes classify as omittable or surfaced.
async fn serve_hello_error(stream: UnixStream, code: String, message: String, delay_ms: u64) {
    if delay_ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
    }
    let (mut read_half, mut write_half) = stream.into_split();
    let Ok(mut decoder) = ClientMessageDecoder::new(FrameDecoderOptions::default()) else {
        return;
    };
    let mut chunk = vec![0u8; 64 * 1024];
    let mut errored = false;
    while let Some(messages) = read_batch(&mut read_half, &mut decoder, &mut chunk).await {
        for incoming in messages {
            if matches!(incoming, ClientMessage::Hello(_)) && !errored {
                errored = true;
                let frame = super::server_frame(&ServerMessage::HelloError(
                    pi_protocol::ServerHelloError {
                        error: pi_protocol::ProtocolError {
                            code: code.clone(),
                            message: message.clone(),
                        },
                    },
                ));
                let _ = write_half.write_all(&frame).await;
            }
        }
    }
}

/// A listener that accepts and holds connections without ever answering,
/// upstream's `startSilentSocket`, with live counts for the probe-cap
/// assertions.
pub struct SilentSocket {}

impl SilentSocket {
    /// Binds and starts the accept loop.
    pub fn start(path: &Path, counts: Option<Rc<ProbeCounts>>) -> Rc<Self> {
        let listener = UnixListener::bind(path).expect("the silent socket binds");
        tokio::task::spawn_local(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                if let Some(counts) = &counts {
                    counts.accept();
                }
                let counts = counts.clone();
                tokio::task::spawn_local(async move {
                    let mut stream = stream;
                    let mut buffer = [0u8; 64];
                    // Held open, never answered, until the probe destroys
                    // its end.
                    loop {
                        match stream.read(&mut buffer).await {
                            Ok(0) | Err(_) => break,
                            Ok(_) => {}
                        }
                    }
                    if let Some(counts) = counts {
                        counts.release();
                    }
                });
            }
        });
        Rc::new(Self {})
    }
}

/// The live-connection counts the probe-cap test polls, upstream's
/// `connections = { active, maximum, total }` object.
#[derive(Debug, Default)]
pub struct ProbeCounts {
    /// The connections held right now.
    pub active: Cell<usize>,
    /// The high-water mark.
    pub maximum: Cell<usize>,
    /// Every accept so far.
    pub total: Cell<usize>,
}

impl ProbeCounts {
    fn accept(self: &Rc<Self>) {
        self.total.set(self.total.get() + 1);
        let active = self.active.get() + 1;
        self.active.set(active);
        if active > self.maximum.get() {
            self.maximum.set(active);
        }
    }

    fn release(self: &Rc<Self>) {
        self.active.set(self.active.get().saturating_sub(1));
    }
}

/// Polls until `condition` holds or the deadline passes, upstream's
/// `expect.poll` restated over the tokio clock.
pub async fn poll_until(condition: impl Fn() -> bool, deadline_ms: u64) -> bool {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(deadline_ms);
    loop {
        if (condition)() {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

/// A raw per-connection handler for the transport tests: handshakes
/// byte-by-byte, records received calls, answers with split-frame
/// responses or a truncated final frame, upstream's inline `createServer`
/// handler in `unix-transport.test.ts`.
pub async fn serve_raw(
    stream: UnixStream,
    server_id: &str,
    received_members: Rc<RefCell<Vec<String>>>,
    truncate_final_frame: bool,
) {
    let (mut read_half, mut write_half) = stream.into_split();
    let Ok(mut decoder) = ClientMessageDecoder::new(FrameDecoderOptions::default()) else {
        return;
    };
    let mut chunk = vec![0u8; 64 * 1024];
    while let Some(messages) = read_batch(&mut read_half, &mut decoder, &mut chunk).await {
        for message in messages {
            match message {
                ClientMessage::Hello(_) => {
                    let frame = super::server_frame(&ServerMessage::Hello(ServerHello {
                        server_id: ServerId::new(server_id).expect("canonical"),
                    }));
                    for byte in frame {
                        if write_half.write_all(&[byte]).await.is_err() {
                            return;
                        }
                    }
                }
                ClientMessage::Cancel(_) => {}
                ClientMessage::Request(envelope) => {
                    let call = pi_chord::services::wire::parse_service_call(&envelope.call)
                        .expect("a valid call arrives");
                    received_members
                        .borrow_mut()
                        .push(format!("{}.{}", call.service_id, call.member));
                    if truncate_final_frame {
                        let _ = write_half.write_all(&[0, 0, 0, 2, 1]).await;
                        let _ = write_half.shutdown().await;
                        return;
                    }
                    let frame = super::server_frame(&ServerMessage::Response(
                        pi_protocol::ResponseEnvelope::Success(pi_protocol::ResponseSuccess {
                            id: envelope.id.clone(),
                            result: Some(pi_chord::types::JsonValue::Array(vec![])),
                        }),
                    ));
                    let split = frame.len() / 2;
                    let _ = write_half.write_all(&frame[..split]).await;
                    let _ = write_half.write_all(&frame[split..]).await;
                }
            }
        }
    }
}

/// Probes the directory's sockets and returns the discovered routes,
/// upstream's `discovery` helper, boxed so the local-task suites can spawn
/// it.
pub fn discovery(
    directory: &Path,
    timeout_ms: Option<u64>,
) -> LocalBoxFuture<Result<Vec<pi_client::unix::UnixServerRoute>, ClientError>> {
    let directory = directory.to_string_lossy().into_owned();
    boxed(async move {
        pi_client::unix::discover_unix_servers(pi_client::unix::DiscoverUnixServersOptions {
            directory,
            timeout_ms,
        })
        .await
    })
}

/// Builds the Unix transport factory over `path` with the given pending
/// cap, the factory construction every Unix transport case drives.
#[must_use]
pub fn unix_transport_factory(
    path: &Path,
    max_pending_bytes: Option<usize>,
) -> pi_client::ByteTransportFactory {
    pi_client::unix::create_unix_transport_factory(pi_client::unix::UnixTransportOptions {
        path: path.to_string_lossy().into_owned(),
        max_pending_bytes,
    })
    .expect("the options validate")
}

/// Binds a silent socket at `path` and builds the Unix transport factory
/// over it, the fixture the pending-cap and close cases drive.
#[must_use]
pub fn silent_socket_factory(
    path: &Path,
    max_pending_bytes: Option<usize>,
) -> pi_client::ByteTransportFactory {
    SilentSocket::start(path, None);
    unix_transport_factory(path, max_pending_bytes)
}

/// Connects one transport through `factory`, the await every Unix
/// transport case performs.
pub async fn connect_transport(factory: &pi_client::ByteTransportFactory) -> Rc<dyn ByteTransport> {
    factory(super::noop_handlers())
        .await
        .expect("the transport connects")
}

/// Binds a raw Unix server answering every connection through
/// [`serve_raw`], and builds the client over a Unix transport pointed at
/// it; returns the client, the temp directory holding the socket, and the
/// members the server received.
pub fn raw_unix_client(
    server_id: &str,
    truncate_final_frame: bool,
) -> (
    pi_client::Client,
    tempfile::TempDir,
    Rc<RefCell<Vec<String>>>,
) {
    let temp = tempfile::tempdir().expect("temp directory");
    let path = temp.path().join("pi.sock");
    let received_members = Rc::new(RefCell::new(Vec::new()));
    let listener = UnixListener::bind(&path).expect("the socket binds");
    tokio::task::spawn_local({
        let received_members = Rc::clone(&received_members);
        let server_id = server_id.to_string();
        async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                tokio::task::spawn_local({
                    let server_id = server_id.clone();
                    let received_members = received_members.clone();
                    async move {
                        serve_raw(stream, &server_id, received_members, truncate_final_frame).await;
                    }
                });
            }
        }
    });
    let factory = unix_transport_factory(&path, None);
    let client = pi_client::Client::new(super::client_options(factory, server_id))
        .expect("the identity is canonical");
    (client, temp, received_members)
}
