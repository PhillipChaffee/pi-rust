//! The server-core suite, ported from upstream `test/server.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. The Unix-listener cases ride
//! the `unix` cfg like the transports they exercise.

#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test assertions panic at the failing case only; the restriction lints target production code"
)]

mod support;

use std::cell::Cell;
use std::rc::Rc;

use pi_chord::future::{LocalBoxFuture, boxed};
use pi_protocol::{ServerMessage, ServerMessageDecoder};
use pi_server::testing::TestServerHost;
use pi_server::{ByteConnection, Failure, Server, ServerListener, ServerOptions};

use support::{SERVER_ID, Servers, run_local, same_failure};

fn host() -> Rc<TestServerHost> {
    Rc::new(TestServerHost::new())
}

#[test]
fn requires_a_canonical_uuidv4_server_identity() {
    run_local(async {
        let build = |server_id: &str| {
            Server::new(
                host(),
                ServerOptions {
                    listeners: Vec::new(),
                    server_id: server_id.to_string(),
                    max_frame_length: None,
                    handshake_timeout_ms: None,
                    on_connection_count_changed: None,
                    on_error: None,
                },
            )
        };
        let empty = build("").expect_err("an empty identity fails");
        assert!(empty.to_string().contains("serverId"));
        let invalid = build("invalid-server").expect_err("a non-canonical identity fails");
        assert!(invalid.to_string().contains("serverId"));
    });
}

#[cfg(unix)]
mod unix_cases {
    use super::*;
    use pi_server::unix::{UnixServerOptions, create_unix_server};
    use support::temp_socket_path;

    #[test]
    fn rejects_concurrent_start_calls_without_leaking_the_unix_listener() {
        run_local(async {
            let servers = Servers::default();
            let path = temp_socket_path("pss-start");
            let server = create_unix_server(host(), support::unix_server_options(&path)).unwrap();
            servers.track(&server);
            // `start` arms its in-flight latch synchronously, upstream's
            // eager `startPromise`.
            let starting = server.start();
            let second = server.start().await.expect_err("the second start fails");
            assert!(second.to_string().contains("starting"));
            starting.await.unwrap();
            server.close().await.unwrap();
            servers.forget(&server);
            assert!(
                std::fs::symlink_metadata(&path).is_err(),
                "the socket does not leak"
            );
            servers.close_all().await;
        });
    }

    #[test]
    fn rejects_timeout_values_above_the_maximum_timer_delay() {
        run_local(async {
            let path = "/tmp/pi-server-timeout-test.sock";
            let over = UnixServerOptions {
                handshake_timeout_ms: Some(2_147_483_648),
                ..support::unix_server_options(path)
            };
            let error = create_unix_server(host(), over).expect_err("the handshake bound fails");
            assert!(error.to_string().contains("handshakeTimeoutMs"));
            let graceful = UnixServerOptions {
                graceful_close_timeout_ms: Some(2_147_483_648),
                ..support::unix_server_options(path)
            };
            let error = create_unix_server(host(), graceful).expect_err("the graceful bound fails");
            assert!(error.to_string().contains("gracefulCloseTimeoutMs"));
        });
    }

    #[test]
    fn rejects_pending_byte_limits_smaller_than_one_maximum_frame() {
        run_local(async {
            let path = temp_socket_path("pss-pending");
            let options = UnixServerOptions {
                max_pending_bytes: Some(131),
                max_frame_length: Some(128),
                ..support::unix_server_options(&path)
            };
            let error = create_unix_server(host(), options).expect_err("the cap fails");
            assert!(error.to_string().contains("maxPendingBytes"));
        });
    }
}

#[test]
fn handshake_timeout_closes_with_a_final_hello_error_frame() {
    run_local(async {
        let final_chunk: Rc<std::cell::RefCell<Option<Vec<u8>>>> =
            Rc::new(std::cell::RefCell::new(None));
        let closed = Rc::new(Cell::new(false));
        let core = Server::new(
            host(),
            ServerOptions {
                listeners: Vec::new(),
                server_id: SERVER_ID.to_string(),
                max_frame_length: Some(1024),
                handshake_timeout_ms: Some(10),
                on_connection_count_changed: None,
                on_error: None,
            },
        )
        .unwrap();
        let connection: Rc<dyn ByteConnection> = Rc::new(TimedOutConnection {
            closed: Rc::clone(&closed),
            final_chunk: Rc::clone(&final_chunk),
        });
        core.accept(connection);

        // The timeout arms a real tokio sleep; the local set drives it.
        while !closed.get() {
            tokio::task::yield_now().await;
        }
        let final_chunk = final_chunk.borrow().clone();
        let Some(final_chunk) = final_chunk else {
            panic!("the terminal close carries the final frame");
        };
        let messages = ServerMessageDecoder::new(pi_protocol::FrameDecoderOptions {
            max_frame_length: 1024,
        })
        .unwrap()
        .push(&final_chunk)
        .unwrap();
        assert!(matches!(
            messages.first(),
            Some(ServerMessage::HelloError(envelope)) if envelope.error.code == "invalid_request"
        ));
        let servers = Servers::default();
        servers.track(&core);
        servers.close_all().await;
    });
}

/// The connection the handshake-timeout case drives, upstream's
/// `TimedOutConnection`: sends fail (the terminal frame must ride `close`).
struct TimedOutConnection {
    closed: Rc<Cell<bool>>,
    final_chunk: Rc<std::cell::RefCell<Option<Vec<u8>>>>,
}

impl ByteConnection for TimedOutConnection {
    fn closed(&self) -> bool {
        self.closed.get()
    }

    fn send(&self, _chunk: Vec<u8>) -> LocalBoxFuture<Result<(), Failure>> {
        boxed(async {
            Err(Failure::message(
                "handshake timeout must use the terminal close frame",
            ))
        })
    }

    fn close(&self, final_chunk: Option<Vec<u8>>) -> LocalBoxFuture<Result<(), Failure>> {
        *self.final_chunk.borrow_mut() = final_chunk;
        self.closed.set(true);
        boxed(async { Ok(()) })
    }
}

#[test]
fn rejects_close_and_closed_when_listener_shutdown_fails() {
    run_local(async {
        let failure = Failure::message("listener close failed");
        let listener: Rc<dyn ServerListener> =
            support::TestListener::new_with_close_error(None, Some(failure.clone()));
        let core = Server::new(
            host(),
            ServerOptions {
                listeners: vec![Rc::clone(&listener)],
                server_id: SERVER_ID.to_string(),
                max_frame_length: None,
                handshake_timeout_ms: None,
                on_connection_count_changed: None,
                on_error: None,
            },
        )
        .unwrap();
        core.start().await.unwrap();

        let close = core.close().await.expect_err("the close fails");
        assert!(
            same_failure(&close, &failure),
            "the close rethrows the listener failure"
        );
        let closed = core.closed().await.expect_err("closed rejects");
        assert!(
            same_failure(&closed, &failure),
            "closed rethrows the listener failure"
        );
    });
}
