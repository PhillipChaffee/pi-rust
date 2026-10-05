//! The conformance double factory, ported from upstream
//! `src/testing/server.ts`: an unstarted [`Server`] with deterministic
//! defaults for the transport conformance cases.

use std::rc::Rc;

use crate::listener::ServerListener;
use crate::server::Server;
use crate::types::ServerOptions;

use super::host::TestServerHost;

/// The options `createTestServer` accepts, upstream's `TestServerOptions`
/// (`ServerOptions` minus `serverId`, plus `host` and `serverId`).
pub struct TestServerOptions {
    /// The transport listeners, upstream's `listeners`; defaults to none.
    pub listeners: Option<Vec<Rc<dyn ServerListener>>>,
    /// The host; defaults to a fresh [`TestServerHost`], upstream's `host`.
    pub host: Option<Rc<TestServerHost>>,
    /// The logical server identity; defaults to the fixed conformance id,
    /// upstream's `serverId`.
    pub server_id: Option<String>,
    /// The framed-byte ceiling, upstream's `maxFrameLength`.
    pub max_frame_length: Option<usize>,
    /// The handshake budget, upstream's `handshakeTimeoutMs`.
    pub handshake_timeout_ms: Option<u64>,
    /// Reports routed errors, upstream's `onError`.
    pub on_error: Option<crate::types::ErrorObserver>,
}

impl std::fmt::Debug for TestServerOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("TestServerOptions").finish()
    }
}

/// The factory's return pair, upstream's `TestServer`.
pub struct TestServer {
    /// The unstarted server.
    pub server: Server<TestServerHost>,
    /// The host the server routes through.
    pub host: Rc<TestServerHost>,
}

impl std::fmt::Debug for TestServer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("TestServer").finish()
    }
}

/// Creates an unstarted server with deterministic defaults for the transport
/// conformance cases, upstream's `createTestServer`.
///
/// # Panics
/// When the defaulted options fail validation, which the fixed defaults
/// cannot.
#[must_use]
pub fn create_test_server(options: TestServerOptions) -> TestServer {
    let host = options
        .host
        .unwrap_or_else(|| Rc::new(TestServerHost::new()));
    let server = Server::new(
        Rc::clone(&host),
        ServerOptions {
            listeners: options.listeners.unwrap_or_default(),
            server_id: options
                .server_id
                .unwrap_or_else(|| "00000000-0000-4000-8000-000000000001".to_string()),
            max_frame_length: options.max_frame_length,
            handshake_timeout_ms: options.handshake_timeout_ms,
            on_connection_count_changed: None,
            on_error: options.on_error,
        },
    )
    .expect("the defaulted test server options are valid");
    TestServer { server, host }
}
