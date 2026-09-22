//! Boundary tests for the Unix transports and discovery, binding the
//! branches the 1:1 ported suites leave open: pending-byte accounting,
//! closed-transport sends, reader-task failure delivery, and the probe
//! omission boundary. Upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

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
#![allow(
    clippy::missing_const_for_fn,
    reason = "test case bodies are deliberately free functions, not const-promotable code"
)]

mod support;

use std::os::unix::fs::PermissionsExt;

use tokio::net::UnixListener;
use tokio::net::UnixStream;

use support::run_local;
use support::unix::{
    HelloErrorServer, SilentSocket, connect_transport, discovery, server_id, silent_socket_factory,
    unix_transport_factory,
};

use pi_client::ClientError;
use pi_client::unix::{DiscoverUnixServersOptions, UnixTransportOptions, discover_unix_servers};

/// Accepts one connection and hands the raw peer stream back to the test,
/// the harness the reader-task failure cases drive.
async fn accepted_peer(listener: UnixListener) -> UnixStream {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    tokio::task::spawn_local(async move {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        let _ = sender.send(stream);
    });
    receiver.await.expect("the peer accepted")
}

#[test]
fn rejects_discovery_timeouts_outside_the_timer_bound() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        for timeout_ms in [0, 2_147_483_648] {
            let error = discover_unix_servers(DiscoverUnixServersOptions {
                directory: temp.path().to_string_lossy().into_owned(),
                timeout_ms: Some(timeout_ms),
            })
            .await
            .expect_err("an out-of-range timeout fails");
            assert!(
                error.to_string().contains("between 1 and 2147483647"),
                "the error names the timer bound: {error}"
            );
        }
    });
}

#[test]
fn rejects_sends_that_exceed_the_pending_byte_limit() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let path = temp.path().join("pi.sock");
        let factory = silent_socket_factory(&path, Some(8));
        let transport = connect_transport(&factory).await;
        let error = transport
            .send(vec![0u8; 100])
            .await
            .expect_err("a send past the cap rejects");
        assert!(
            error.to_string().contains("pending byte limit"),
            "the cap is named: {error}"
        );
        let error = transport
            .send(vec![0u8; 100])
            .await
            .expect_err("the cap stays enforced after a rejection");
        assert!(error.to_string().contains("pending byte limit"));
    });
}

#[test]
fn rejects_sends_on_a_closed_transport_and_keeps_close_idempotent() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let path = temp.path().join("pi.sock");
        let factory = silent_socket_factory(&path, None);
        let transport = connect_transport(&factory).await;
        transport.close();
        transport.close();
        let error = transport
            .send(vec![0u8; 4])
            .await
            .expect_err("a send on a closed transport rejects");
        assert!(
            error.to_string().contains("closed"),
            "the close is named: {error}"
        );
    });
}

#[test]
fn completes_a_send_queued_before_its_transport_handle_is_dropped() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let path = temp.path().join("pi.sock");
        let factory = silent_socket_factory(&path, None);
        let transport = connect_transport(&factory).await;
        let pending = transport.send(vec![0u8; 4]);
        drop(transport);
        assert!(
            pending.await.is_ok(),
            "the reader task still delivers the queued send after the handle drops"
        );
    });
}

#[test]
fn fails_a_blocked_write_when_the_peer_closes() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let path = temp.path().join("pi.sock");
        let listener = UnixListener::bind(&path).expect("the socket binds");
        let factory = unix_transport_factory(&path, Some(512 * 1024));
        let transport = connect_transport(&factory).await;
        let peer = accepted_peer(listener).await;
        let blocked = transport.send(vec![0u8; 256 * 1024]);
        let _queued = transport.send(vec![0u8; 1024]);
        // The 256 KiB chunk parks the reader task's write against the
        // never-reading peer's buffer; closing the peer then fails the
        // parked write instead of leaving it pending forever.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        drop(peer);
        let error = tokio::time::timeout(std::time::Duration::from_millis(5_000), blocked)
            .await
            .expect("the parked write resolves once the peer closes")
            .expect_err("a write to a closed peer fails");
        assert!(
            matches!(&error, ClientError::Other { code: Some(code), .. } if code == "EPIPE"),
            "the broken pipe surfaces with its errno: {error}"
        );
    });
}

#[test]
fn omits_endpoints_that_reject_the_handshake_version() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let identity = server_id(1);
        let _server = HelloErrorServer::start(temp.path(), &identity, "version", "unsupported");
        assert_eq!(
            discover_unix_servers(DiscoverUnixServersOptions {
                directory: temp.path().to_string_lossy().into_owned(),
                timeout_ms: Some(1_000),
            })
            .await
            .expect("discovery answers"),
            vec![]
        );
    });
}

#[test]
fn surfaces_handshake_errors_other_than_version_rejections() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let identity = server_id(1);
        let _server = HelloErrorServer::start(temp.path(), &identity, "auth", "not allowed");
        let error = discover_unix_servers(DiscoverUnixServersOptions {
            directory: temp.path().to_string_lossy().into_owned(),
            timeout_ms: Some(1_000),
        })
        .await
        .expect_err("a non-version handshake error surfaces");
        assert_eq!(
            error,
            ClientError::Server(pi_protocol::ProtocolError {
                code: "auth".to_string(),
                message: "not allowed".to_string(),
            })
        );
    });
}

#[test]
fn skips_remaining_candidates_once_a_probe_fails() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let _server = HelloErrorServer::start(temp.path(), &server_id(1), "auth", "not allowed");
        let _silent =
            SilentSocket::start(&temp.path().join(format!("{}.sock", server_id(2))), None);
        let error = discovery(temp.path(), Some(100)).await;
        assert!(
            matches!(&error, Err(ClientError::Server(failure)) if failure.code == "auth"),
            "the recorded failure surfaces: {error:?}"
        );
    });
}

#[test]
fn skips_a_second_failed_probe_once_a_failure_is_recorded() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        // The fast endpoint records the failure; the slow one still fails
        // with a surfaced error afterwards, and its recording is skipped.
        let _fast = HelloErrorServer::start(temp.path(), &server_id(1), "auth", "not allowed");
        let _slow =
            HelloErrorServer::start_delayed(temp.path(), &server_id(2), "auth", "not allowed", 250);
        let error = discovery(temp.path(), Some(2_000)).await;
        assert!(
            matches!(&error, Err(ClientError::Server(failure)) if failure.code == "auth"),
            "the first recorded failure surfaces: {error:?}"
        );
    });
}

#[test]
fn surfaces_an_unsearchable_directory_as_a_filesystem_failure() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let nested = temp.path().join("sockets");
        std::fs::create_dir(&nested).expect("the directory writes");
        std::fs::write(nested.join(format!("{}.sock", server_id(1))), "socket-ish")
            .expect("the entry writes");
        std::fs::set_permissions(&nested, std::fs::Permissions::from_mode(0o444))
            .expect("the permissions tighten");
        let discovery = discovery(&nested, None);
        let error = discovery.await;
        std::fs::set_permissions(&nested, std::fs::Permissions::from_mode(0o755))
            .expect("the permissions restore");
        let error = error.expect_err("an lstat failure surfaces");
        assert!(
            matches!(&error, ClientError::Other { code: None, .. }),
            "an errno outside the table carries no code: {error}"
        );
    });
}

#[test]
fn formats_the_unix_option_debug_output() {
    let options = UnixTransportOptions {
        path: "/tmp/pi.sock".to_string(),
        max_pending_bytes: Some(8),
    };
    assert!(
        format!("{options:?}").contains("path"),
        "the transport options debug names its fields"
    );
    assert!(
        format!(
            "{:?}",
            DiscoverUnixServersOptions {
                directory: "/tmp".to_string(),
                timeout_ms: Some(1),
            }
        )
        .contains("directory"),
        "the discovery options debug names its fields"
    );
}
