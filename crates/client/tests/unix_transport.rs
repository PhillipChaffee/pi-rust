//! The Unix transport suite, ported from upstream
//! `test/unix-transport.test.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

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
    clippy::unwrap_used,
    reason = "test fixtures unwrap invariants the case's assertions cover"
)]

mod support;

use support::noop_handlers;
use support::run_local;
use support::unix::{raw_unix_client, unix_transport_factory};

use pi_client::ClientError;
use pi_client::unix::{UnixTransportOptions, create_unix_transport_factory};

const SERVER_ID: &str = "00000000-0000-4000-8000-000000000001";

/// The request both transport cases drive, against the server target over
/// the real socket.
fn server_list_request(
    client: &pi_client::Client,
) -> pi_chord::future::LocalBoxFuture<Result<Option<pi_chord::types::JsonValue>, ClientError>> {
    client.request(
        &pi_protocol::RpcTarget::Server(pi_protocol::ServerTarget {
            server_id: pi_protocol::ServerId::new(SERVER_ID).unwrap(),
        }),
        &pi_chord::types::ServiceCall {
            service_id: "test.server".to_string(),
            instance: None,
            member: "list".to_string(),
            args: vec![],
        },
        None,
    )
}

#[test]
fn rejects_invalid_unix_transport_options() {
    run_local(async {
        let Err(error) = create_unix_transport_factory(UnixTransportOptions {
            path: String::new(),
            max_pending_bytes: None,
        }) else {
            panic!("an empty path fails");
        };
        assert!(error.to_string().contains("must not be empty"));
        let Err(error) = create_unix_transport_factory(UnixTransportOptions {
            path: "/tmp/pi.sock".to_string(),
            max_pending_bytes: Some(0),
        }) else {
            panic!("a zero pending cap fails");
        };
        assert!(error.to_string().contains("positive"));
    });
}

#[test]
fn carries_a_complete_client_handshake_and_request_over_a_real_unix_socket() {
    run_local(async {
        let (client, _temp, received_members) = raw_unix_client(SERVER_ID, false);
        let hello = client.connect().await.expect("the handshake answers");
        assert_eq!(hello.server_id.as_str(), SERVER_ID);
        let result = server_list_request(&client)
            .await
            .expect("the request answers");
        assert_eq!(result, Some(pi_chord::types::JsonValue::Array(vec![])));
        assert_eq!(
            received_members.borrow().clone(),
            vec!["test.server.list".to_string()]
        );
        client.dispose();
    });
}

#[test]
fn reports_truncated_final_frames_through_client() {
    run_local(async {
        let (client, _temp, _received_members) = raw_unix_client(SERVER_ID, true);
        client.connect().await.expect("the handshake answers");
        let error = server_list_request(&client)
            .await
            .expect_err("truncation fails the request");
        assert!(matches!(error, ClientError::Protocol(_)));
        assert!(
            error.to_string().to_lowercase().contains("truncated"),
            "the truncation is reported: {error}"
        );
        assert_eq!(
            client.connection_state(),
            pi_client::ConnectionState::Disconnected
        );
        client.dispose();
    });
}

#[test]
fn rejects_connection_attempts_to_missing_sockets() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let path = temp.path().join("pi.sock");
        let factory = unix_transport_factory(&path, None);
        let connected = factory(noop_handlers());
        let Err(error) = connected.await else {
            panic!("a missing socket rejects");
        };
        assert!(matches!(error, ClientError::Other { code: Some(code), .. } if code == "ENOENT"));
    });
}
