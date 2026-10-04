//! The wire-level test client, ported from upstream `src/testing/client.ts`.
//!
//! It decodes what the server sends, records every message, and resolves
//! `next(predicate)` waiters in arrival order — upstream's `ServerMessage`
//! waiter set. `connectUnixTestClient` drives a real Unix socket and is
//! Unix-only like the transports it exercises.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_chord::future::{LocalBoxFuture, boxed};
use pi_chord::services::wire::service_call_to_json;
use pi_chord::types::JsonValue;
use pi_protocol::{
    AttachmentEnvelope, ClientHello, ClientMessage, FrameDecoderOptions, RequestEnvelope,
    ResponseEnvelope, RpcTarget, ServerId, ServerMessage, ServerMessageDecoder, ServerTarget,
    SessionTarget, encode_client_message,
};

use crate::latch::{Deferred, Latch};
use crate::types::ready;

/// The send surface the wire client drives, upstream's `WireChannel`.
pub trait WireChannel {
    /// Writes one chunk in order, upstream's `send`.
    fn send(&self, chunk: Vec<u8>) -> LocalBoxFuture<()>;

    /// Writes one chunk split at `split_at`, upstream's `sendFragmented`.
    fn send_fragmented(&self, chunk: Vec<u8>, split_at: usize) -> LocalBoxFuture<()>;

    /// Closes the channel, upstream's `close`.
    fn close(&self) -> LocalBoxFuture<()>;
}

/// The result one `next` waiter settles with, upstream's
/// `Promise<ServerMessage>` rejections carrying an `Error` message.
pub type WireResult = Result<ServerMessage, String>;

/// One pending `next` waiter, upstream's `MessageWaiter`.
struct TestWaiter {
    predicate: Rc<dyn Fn(&ServerMessage) -> bool>,
    done: Rc<Latch<WireResult>>,
}

/// The message-level client the conformance and protocol cases drive,
/// upstream's `ProtocolTestClient`.
pub struct ProtocolTestClient {
    channel: Rc<dyn WireChannel>,
    decoder: RefCell<ServerMessageDecoder>,
    messages: RefCell<Vec<ServerMessage>>,
    waiters: RefCell<Vec<TestWaiter>>,
    closed_deferred: Deferred<()>,
    closed_value: Cell<bool>,
    request_sequence: Cell<u64>,
    attachment: RefCell<Option<(String, String)>>,
}

impl std::fmt::Debug for ProtocolTestClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ProtocolTestClient").finish()
    }
}

impl ProtocolTestClient {
    /// Builds the client over `channel`, upstream's
    /// `new ProtocolTestClient(channel)`.
    ///
    /// # Panics
    /// Never for the default decoder options, which the default frame
    /// ceiling satisfies.
    pub fn new(channel: Rc<dyn WireChannel>) -> Self {
        Self {
            channel,
            decoder: RefCell::new(
                ServerMessageDecoder::new(FrameDecoderOptions::default())
                    .expect("the default frame ceiling is valid"),
            ),
            messages: RefCell::new(Vec::new()),
            waiters: RefCell::new(Vec::new()),
            closed_deferred: Deferred::new(),
            closed_value: Cell::new(false),
            request_sequence: Cell::new(0),
            attachment: RefCell::new(None),
        }
    }

    /// Whether the wire closed, upstream's `closed`.
    pub const fn closed(&self) -> bool {
        self.closed_value.get()
    }

    /// Sends the client hello and waits for the handshake answer, upstream's
    /// `hello(version)`.
    ///
    /// # Errors
    /// A closed wire, upstream's waiter rejection.
    pub async fn hello(&self, version: pi_chord::types::JsonNumber) -> WireResult {
        let response = self.next(predicate_fn(|message: &ServerMessage| {
            matches!(
                message,
                ServerMessage::Hello(_) | ServerMessage::HelloError(_)
            )
        }));
        self.send_message(&ClientMessage::Hello(ClientHello { version }))
            .await;
        response.await
    }

    /// Sends one routed request and waits for its response envelope,
    /// upstream's `requestService(target, call, id?)`.
    ///
    /// # Errors
    /// A closed wire, or a non-response message matching the correlation id.
    pub async fn request_service(
        &self,
        target: RpcTarget,
        call: pi_chord::types::ServiceCall,
        id: Option<String>,
    ) -> Result<ResponseEnvelope, String> {
        let id = id.unwrap_or_else(|| {
            let next = self.request_sequence.get() + 1;
            self.request_sequence.set(next);
            format!("request-{next}")
        });
        let correlation = id.clone();
        let response = self.next(predicate_fn(move |message: &ServerMessage| match message {
            ServerMessage::Response(ResponseEnvelope::Success(success)) => {
                success.id == correlation
            }
            ServerMessage::Response(ResponseEnvelope::Failure(failure)) => {
                failure.id == correlation
            }
            _ => false,
        }));
        self.send_message(&ClientMessage::Request(RequestEnvelope {
            id,
            target,
            call: service_call_to_json(&call),
        }))
        .await;
        match response.await? {
            ServerMessage::Response(envelope) => Ok(envelope),
            other => Err(format!("expected a response envelope, got {other:?}")),
        }
    }

    /// Runs the session-management attach call, upstream's `attach`.
    ///
    /// # Errors
    /// A closed wire, or a non-canonical server id.
    pub async fn attach(
        &self,
        server_id: &str,
        session_id: &str,
    ) -> Result<ResponseEnvelope, String> {
        self.request_service(
            server_target(server_id),
            pi_chord::types::ServiceCall {
                service_id: "pi.session-management".to_string(),
                instance: None,
                member: "attach".to_string(),
                args: vec![JsonValue::Str(session_id.to_string())],
            },
            None,
        )
        .await
    }

    /// Runs one session-scoped call against the tracked attachment, upstream's
    /// `requestSessionService`; a stale or absent attachment routes with the
    /// `missing-attachment` placeholder id.
    ///
    /// # Errors
    /// A closed wire, or a non-canonical server id.
    pub async fn request_session_service(
        &self,
        server_id: &str,
        session_id: &str,
        call: pi_chord::types::ServiceCall,
        id: Option<String>,
    ) -> Result<ResponseEnvelope, String> {
        let Some(server) = ServerId::new(server_id) else {
            return Err("server id must be a canonical lowercase UUIDv4".to_string());
        };
        let attachment = self.attachment.borrow().clone();
        let target = match attachment {
            Some((attached_session, attachment_id)) if attached_session == session_id => {
                RpcTarget::Session(SessionTarget {
                    server_id: server.clone(),
                    session_id: session_id.to_string(),
                    attachment_id,
                })
            }
            _ => RpcTarget::Session(SessionTarget {
                server_id: server.clone(),
                session_id: session_id.to_string(),
                attachment_id: "missing-attachment".to_string(),
            }),
        };
        self.request_service(target, call, id).await
    }

    /// Encodes and sends one client message, upstream's `sendMessage`.
    ///
    /// # Panics
    /// When the driving case sends a schema-invalid message, upstream's
    /// throw.
    pub async fn send_message(&self, message: &ClientMessage) {
        let frame = encode_client_message(message, FrameDecoderOptions::default())
            .expect("the driving case sends a valid message");
        self.channel.send(frame).await;
    }

    /// Sends raw bytes, upstream's `sendBytes`.
    pub async fn send_bytes(&self, chunk: Vec<u8>) {
        self.channel.send(chunk).await;
    }

    /// Encodes one message and sends it split at `split_at`, upstream's
    /// `sendFragmentedMessage`.
    ///
    /// # Panics
    /// When the driving case sends a schema-invalid message, upstream's
    /// throw.
    pub async fn send_fragmented_message(&self, message: &ClientMessage, split_at: usize) {
        let frame = encode_client_message(message, FrameDecoderOptions::default())
            .expect("the driving case sends a valid message");
        self.channel.send_fragmented(frame, split_at).await;
    }

    /// The first recorded or future message matching `predicate`, upstream's
    /// `next`.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the predicate joins the waiter set, upstream's by-value closure"
    )]
    pub fn next(
        &self,
        predicate: Rc<dyn Fn(&ServerMessage) -> bool>,
    ) -> LocalBoxFuture<WireResult> {
        self.next_from(0, predicate)
    }

    /// The same wait starting after `index` recorded messages, upstream's
    /// `nextFrom`.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the predicate joins the waiter set, upstream's by-value closure"
    )]
    pub fn next_from(
        &self,
        index: usize,
        predicate: Rc<dyn Fn(&ServerMessage) -> bool>,
    ) -> LocalBoxFuture<WireResult> {
        let existing = self
            .messages
            .borrow()
            .iter()
            .skip(index)
            .find(|message| predicate(message))
            .cloned();
        if let Some(existing) = existing {
            return ready(Ok(existing));
        }
        if self.closed_value.get() {
            return ready(Err("Wire client is closed".to_string()));
        }
        let done = Rc::new(Latch::new());
        self.waiters.borrow_mut().push(TestWaiter {
            predicate,
            done: Rc::clone(&done),
        });
        boxed(async move { done.wait().await })
    }

    /// Resolves once the wire closed, upstream's `waitForClose`.
    pub fn wait_for_close(&self) -> LocalBoxFuture<()> {
        if self.closed_value.get() {
            return ready(());
        }
        self.closed_deferred.promise()
    }

    /// Closes the channel, upstream's `close`.
    pub fn close(&self) -> LocalBoxFuture<()> {
        let channel = Rc::clone(&self.channel);
        boxed(async move {
            channel.close().await;
        })
    }

    /// The recorded server messages, upstream's `messages` field.
    pub fn messages(&self) -> Vec<ServerMessage> {
        self.messages.borrow().clone()
    }

    /// Feeds one server frame, upstream's `receive`.
    pub fn receive(&self, chunk: &[u8]) {
        let messages = match self.decoder.borrow_mut().push(chunk) {
            Ok(messages) => messages,
            Err(error) => {
                self.fail(&error.to_string());
                return;
            }
        };
        for message in messages {
            if let ServerMessage::Attachment(AttachmentEnvelope { attachment }) = &message {
                *self.attachment.borrow_mut() = attachment
                    .as_ref()
                    .map(|target| (target.session_id.clone(), target.attachment_id.clone()));
            }
            self.messages.borrow_mut().push(message.clone());
            let matched: Vec<Rc<Latch<WireResult>>> = self
                .waiters
                .borrow()
                .iter()
                .filter(|waiter| (waiter.predicate)(&message))
                .map(|waiter| Rc::clone(&waiter.done))
                .collect();
            if !matched.is_empty() {
                self.waiters
                    .borrow_mut()
                    .retain(|waiter| !matched.iter().any(|done| Rc::ptr_eq(done, &waiter.done)));
                for done in matched {
                    done.settle(Ok(message.clone()));
                }
            }
        }
    }

    /// Marks the wire closed and fails every waiter, upstream's
    /// `markClosed`.
    pub fn mark_closed(&self) {
        if self.closed_value.get() {
            return;
        }
        self.closed_value.set(true);
        self.closed_deferred.resolve(());
        self.fail("Wire connection closed");
    }

    /// Fails every waiter, upstream's `fail`.
    pub fn fail(&self, error: &str) {
        let waiters: Vec<Rc<Latch<WireResult>>> = self
            .waiters
            .borrow()
            .iter()
            .map(|waiter| Rc::clone(&waiter.done))
            .collect();
        self.waiters.borrow_mut().clear();
        for done in waiters {
            done.settle(Err(error.to_string()));
        }
    }
}

/// Boxes a predicate, the coercion every `next` closure needs.
fn predicate_fn<F: Fn(&ServerMessage) -> bool + 'static>(
    function: F,
) -> Rc<dyn Fn(&ServerMessage) -> bool> {
    Rc::new(function)
}

/// The server-addressed target one attach call routes to.
fn server_target(server_id: &str) -> RpcTarget {
    let server = ServerId::new(server_id).expect("the driving case passes a canonical id");
    RpcTarget::Server(ServerTarget { server_id: server })
}

/// The Unix-socket channel one real-socket test client drives.
struct SocketChannel {
    commands: tokio::sync::mpsc::UnboundedSender<SocketCommand>,
    shutdown: Rc<Latch<()>>,
}

enum SocketCommand {
    Send { chunk: Vec<u8> },
    Close,
}

impl WireChannel for SocketChannel {
    fn send(&self, chunk: Vec<u8>) -> LocalBoxFuture<()> {
        let _ = self.commands.send(SocketCommand::Send { chunk });
        boxed(async {})
    }

    fn send_fragmented(&self, chunk: Vec<u8>, split_at: usize) -> LocalBoxFuture<()> {
        let (head, tail) = chunk.split_at(split_at);
        let _ = self.commands.send(SocketCommand::Send {
            chunk: head.to_vec(),
        });
        let _ = self.commands.send(SocketCommand::Send {
            chunk: tail.to_vec(),
        });
        boxed(async {})
    }

    fn close(&self) -> LocalBoxFuture<()> {
        let commands = self.commands.clone();
        let shutdown = Rc::clone(&self.shutdown);
        boxed(async move {
            let _ = commands.send(SocketCommand::Close);
            shutdown.wait().await;
        })
    }
}

/// Connects one real Unix-socket wire client, upstream's
/// `connectUnixTestClient`.
///
/// The reader task feeds `receive`, the writer task drains ordered sends,
/// and a close shuts the write half down and stops the reader, upstream's
/// `socket.destroy()`.
///
/// # Errors
/// Whatever the socket connect raises.
#[cfg(unix)]
pub async fn connect_unix_test_client(path: &str) -> std::io::Result<Rc<ProtocolTestClient>> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let stream = tokio::net::UnixStream::connect(path).await?;
    let (commands, mut receiver) = tokio::sync::mpsc::unbounded_channel::<SocketCommand>();
    let shutdown = Rc::new(Latch::new());
    let client = Rc::new(ProtocolTestClient::new(Rc::new(SocketChannel {
        commands,
        shutdown: Rc::clone(&shutdown),
    })));
    let (mut read_half, mut write_half) = stream.into_split();
    let writer_shutdown = Rc::clone(&shutdown);
    tokio::task::spawn_local(async move {
        while let Some(command) = receiver.recv().await {
            match command {
                SocketCommand::Send { chunk } => {
                    let _ = write_half.write_all(&chunk).await;
                }
                SocketCommand::Close => {
                    let _ = write_half.shutdown().await;
                    writer_shutdown.settle(());
                    break;
                }
            }
        }
    });
    {
        let client = Rc::clone(&client);
        let reader_shutdown = Rc::clone(&shutdown);
        tokio::task::spawn_local(async move {
            let mut buffer = vec![0u8; 64 * 1024];
            loop {
                tokio::select! {
                    read = read_half.read(&mut buffer) => match read {
                        Ok(0) => {
                            client.mark_closed();
                            break;
                        }
                        Ok(read) => client.receive(&buffer[..read]),
                        Err(error) => {
                            client.fail(&error.to_string());
                            client.mark_closed();
                            break;
                        }
                    },
                    () = reader_shutdown.wait() => {
                        client.mark_closed();
                        break;
                    }
                }
            }
        });
    }
    Ok(client)
}
