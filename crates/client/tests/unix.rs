//! The discovery suite, ported from upstream `test/unix.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. Upstream drives its
//! cross-package imports (`../../server/src`) through the in-crate
//! handshake stub in `support/unix.rs`, which answers the same protocol
//! surface these cases exercise; the forked stale-socket fixture ports to
//! a bound-then-dropped listener, which leaves the same stale socket file
//! the `SIGKILL`ed child did.

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

use std::os::unix::fs::FileTypeExt;
use std::rc::Rc;

use support::run_local;
use support::unix::{ProbeCounts, SilentSocket, TestUnixServer, discovery, poll_until, server_id};

#[test]
fn returns_no_routes_when_the_server_directory_is_missing() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let directory = temp.path().join("missing");
        assert_eq!(
            discovery(&directory, None).await.expect("no routes"),
            vec![]
        );
    });
}

#[test]
fn discovers_reachable_servers_in_server_id_order() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let directory = temp.path();
        let first = server_id(1);
        let second = server_id(2);
        let _second_server = TestUnixServer::start(directory, &second, &second);
        let _first_server = TestUnixServer::start(directory, &first, &first);

        let routes = discovery(directory, None).await.expect("discovery answers");
        assert_eq!(
            routes,
            vec![
                pi_client::unix::UnixServerRoute {
                    server_id: pi_protocol::ServerId::new(&first).expect("canonical"),
                    path: directory
                        .join(format!("{first}.sock"))
                        .to_string_lossy()
                        .into_owned(),
                },
                pi_client::unix::UnixServerRoute {
                    server_id: pi_protocol::ServerId::new(&second).expect("canonical"),
                    path: directory
                        .join(format!("{second}.sock"))
                        .to_string_lossy()
                        .into_owned(),
                },
            ]
        );
    });
}

#[test]
fn ignores_malformed_entries_non_sockets_and_mismatched_servers() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let directory = temp.path();
        std::fs::write(
            directory.join(format!("{}.sock", server_id(1))),
            "not a socket",
        )
        .expect("the regular file writes");
        std::fs::write(directory.join("not-a-server.sock"), "ignored").expect("the file writes");
        std::fs::create_dir(directory.join(format!("{}.sock", server_id(2))))
            .expect("the directory writes");
        let _mismatched = TestUnixServer::start(directory, &server_id(3), &server_id(4));

        assert_eq!(
            discovery(directory, None).await.expect("discovery answers"),
            vec![]
        );
    });
}

#[test]
fn ignores_stale_sockets_without_deleting_them() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let path = temp.path().join(format!("{}.sock", server_id(1)));
        // Upstream forks a server and SIGKILLs it; the port binds and drops
        // a listener, which leaves the same stale socket file behind.
        drop(tokio::net::UnixListener::bind(&path).expect("the stale socket binds"));

        assert_eq!(
            discovery(temp.path(), None)
                .await
                .expect("discovery answers"),
            vec![]
        );
        let metadata = std::fs::symlink_metadata(&path).expect("the stale socket stays");
        assert!(metadata.file_type().is_socket());
    });
}

#[test]
fn times_out_an_unresponsive_socket_without_deleting_it() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let path = temp.path().join(format!("{}.sock", server_id(1)));
        let _silent = SilentSocket::start(&path, None);

        assert_eq!(
            discovery(temp.path(), Some(20))
                .await
                .expect("discovery answers"),
            vec![]
        );
        let metadata = std::fs::symlink_metadata(&path).expect("the socket stays");
        assert!(metadata.file_type().is_socket());
    });
}

#[test]
fn limits_concurrent_probes_to_16() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let counts = Rc::new(ProbeCounts::default());
        let mut sockets = Vec::new();
        for index in 1..=20u64 {
            sockets.push(SilentSocket::start(
                &temp.path().join(format!("{}.sock", server_id(index))),
                Some(Rc::clone(&counts)),
            ));
        }

        let running = tokio::task::spawn_local(discovery(temp.path(), Some(100)));
        let reached = poll_until(|| counts.active.get() == 16, 5_000).await;
        assert!(reached, "the probe pool fills to the 16-connection cap");
        assert_eq!(counts.maximum.get(), 16);
        assert_eq!(
            running
                .await
                .expect("discovery answers")
                .expect("every silent socket times out and is omitted"),
            vec![]
        );
        assert_eq!(counts.total.get(), 20);
        drop(sockets);
    });
}

#[test]
fn ignores_an_endpoint_that_closes_before_its_handshake() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let path = temp.path().join(format!("{}.sock", server_id(1)));
        let listener = tokio::net::UnixListener::bind(&path).expect("the socket binds");
        tokio::task::spawn_local(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                drop(stream);
            }
        });

        assert_eq!(
            discovery(temp.path(), None).await.expect("discovery"),
            vec![]
        );
    });
}

#[test]
fn propagates_unexpected_filesystem_errors() {
    run_local(async {
        let temp = tempfile::tempdir().expect("temp directory");
        let file = temp.path().join("not-a-directory");
        std::fs::write(&file, "content").expect("the file writes");

        let error = discovery(&file, None)
            .await
            .expect_err("reading a file path as a directory fails");
        assert_eq!(
            error,
            pi_client::ClientError::other_with_code(
                std::fs::read_dir(&file).expect_err("ENOTDIR").to_string(),
                Some("ENOTDIR".to_string()),
            )
        );
    });
}
