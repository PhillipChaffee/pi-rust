//! The Unix-listener filesystem-lifecycle suite, ported from upstream
//! `test/unix.test.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//! The stale-socket fixture upstream forks (`stale-socket-server.mjs`, a
//! child that binds and dies) restates as a bound-then-dropped listener: the
//! socket file persists with no live endpoint, the same filesystem state the
//! forked-and-killed child leaves.

#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test assertions panic at the failing case only; the restriction lints target production code"
)]

mod support;

use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};

use pi_server::testing::TestServerHost;
use pi_server::testing::connect_unix_test_client;
use pi_server::unix::get_unix_socket_path;

use support::{SERVER_ID, Servers, run_local};

fn make_server(path: &str) -> pi_server::Server<TestServerHost> {
    support::create_unix_test_server(path)
}

#[test]
fn creates_an_in_memory_server_id_and_derives_its_explicit_unix_socket_path() {
    run_local(async {
        let servers = Servers::default();
        let directory = tempfile::tempdir().unwrap();
        let path = get_unix_socket_path(SERVER_ID, directory.path().to_str().unwrap()).unwrap();
        assert!(
            path.ends_with(&format!("{SERVER_ID}.sock")),
            "the path derives from the identity"
        );

        let first = support::create_unix_test_server(&path);
        servers.track(&first);
        first.start().await.unwrap();
        let first_client = connect_unix_test_client(&path).await.unwrap();
        let hello = first_client.hello(support::version(8.0)).await.unwrap();
        assert!(
            matches!(hello, pi_protocol::ServerMessage::Hello(answer) if answer.server_id.as_str() == SERVER_ID)
        );
        first_client.close().await;
        servers.forget(&first);
        first.close().await.unwrap();

        let replacement = support::create_unix_test_server(&path);
        servers.track(&replacement);
        replacement.start().await.unwrap();
        let replacement_client = connect_unix_test_client(&path).await.unwrap();
        let hello = replacement_client
            .hello(support::version(8.0))
            .await
            .unwrap();
        assert!(
            matches!(hello, pi_protocol::ServerMessage::Hello(answer) if answer.server_id.as_str() == SERVER_ID)
        );
        replacement_client.close().await;
        servers.close_all().await;
    });
}

#[test]
fn rejects_a_live_listener_without_unlinking_it() {
    run_local(async {
        let servers = Servers::default();
        let path = support::temp_socket_path("unix");
        let first = make_server(&path);
        servers.track(&first);
        first.start().await.unwrap();
        let first_identity = std::fs::symlink_metadata(&path).unwrap();

        let second = make_server(&path);
        servers.track(&second);
        let error = second.start().await.expect_err("the live listener fails");
        assert!(error.to_string().contains("already running"));
        let current = std::fs::symlink_metadata(&path).unwrap();
        assert!(current.file_type().is_socket());
        assert_eq!(
            (current.dev(), current.ino()),
            (first_identity.dev(), first_identity.ino())
        );

        let client = connect_unix_test_client(&path).await.unwrap();
        let hello = client.hello(support::version(8.0)).await.unwrap();
        assert!(matches!(hello, pi_protocol::ServerMessage::Hello(_)));
        client.close().await;
        servers.close_all().await;
    });
}

#[test]
fn never_unlinks_a_regular_file_at_the_configured_path() {
    run_local(async {
        let servers = Servers::default();
        let path = support::temp_socket_path("unix");
        std::fs::write(&path, "do not remove").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        let server = make_server(&path);
        servers.track(&server);
        let error = server.start().await.expect_err("the regular file fails");
        assert!(error.to_string().contains("non-socket"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "do not remove");
        servers.close_all().await;
    });
}

#[test]
fn creates_nested_temp_parents_restricts_permissions_and_removes_its_own_socket() {
    run_local(async {
        let servers = Servers::default();
        let directory = tempfile::tempdir().unwrap();
        let nested = directory.path().join("p").join("n");
        let path = nested.join("server.sock").to_string_lossy().into_owned();
        let server = make_server(&path);
        servers.track(&server);
        server.start().await.unwrap();
        let stats = std::fs::symlink_metadata(&path).unwrap();
        assert!(stats.file_type().is_socket());
        assert_eq!(stats.permissions().mode() & 0o777, 0o600);
        let entries: Vec<String> = std::fs::read_dir(nested)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, vec!["server.sock".to_string()]);

        servers.forget(&server);
        server.close().await.unwrap();
        assert!(
            std::fs::symlink_metadata(&path).is_err(),
            "the socket is removed"
        );
        servers.close_all().await;
    });
}

#[test]
fn does_not_remove_a_replacement_inode_during_shutdown() {
    run_local(async {
        let servers = Servers::default();
        let path = support::temp_socket_path("unix");
        let server = make_server(&path);
        servers.track(&server);
        server.start().await.unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, "replacement").unwrap();

        servers.forget(&server);
        server.close().await.unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "replacement");
        servers.close_all().await;
    });
}

#[test]
fn removes_a_genuinely_stale_socket_before_binding() {
    run_local(async {
        let servers = Servers::default();
        let path = support::temp_socket_path("unix");
        // The stale-socket fixture restated: a listener binds, proving the
        // path live, and then drops — the socket file persists with no live
        // endpoint, the state the forked-and-killed child leaves.
        {
            let stale = std::os::unix::net::UnixListener::bind(&path).unwrap();
            assert!(
                std::fs::symlink_metadata(&path)
                    .unwrap()
                    .file_type()
                    .is_socket()
            );
            drop(stale);
        }
        assert!(
            std::fs::symlink_metadata(&path)
                .unwrap()
                .file_type()
                .is_socket()
        );

        let server = make_server(&path);
        servers.track(&server);
        server.start().await.unwrap();
        let live = std::fs::symlink_metadata(&path).unwrap();
        assert!(live.file_type().is_socket());
        let client = connect_unix_test_client(&path).await.unwrap();
        let hello = client.hello(support::version(8.0)).await.unwrap();
        assert!(matches!(hello, pi_protocol::ServerMessage::Hello(_)));
        client.close().await;
        servers.close_all().await;
    });
}
