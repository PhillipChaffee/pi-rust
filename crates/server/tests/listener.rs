//! The listener-composition suite, ported from upstream
//! `test/listener.test.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test assertions panic at the failing case only; the restriction lints target production code"
)]

mod support;

use std::rc::Rc;

use pi_server::Failure;
use pi_server::testing::create_test_server;

use support::run_local;

#[test]
fn starts_and_closes_every_configured_listener() {
    run_local(async {
        let first = support::TestListener::new(None);
        let second = support::TestListener::new(None);
        let test = create_test_server(pi_server::testing::TestServerOptions {
            listeners: Some(vec![first.clone(), second.clone()]),
            host: None,
            server_id: None,
            max_frame_length: None,
            handshake_timeout_ms: None,
            on_error: None,
        });

        test.server.start().await.unwrap();
        // The acceptor each listener received is callable, upstream's
        // `toBeTypeOf("function")`.
        test.server.close().await.unwrap();
        assert_eq!(first.close_count.get(), 1);
        assert_eq!(second.close_count.get(), 1);
    });
}

#[test]
fn closes_previously_started_listeners_when_startup_fails() {
    run_local(async {
        let first = support::TestListener::new(None);
        let failure = Failure::message("listener failed");
        let second = support::TestListener::new(Some(failure.clone()));
        let test = create_test_server(pi_server::testing::TestServerOptions {
            listeners: Some(vec![first.clone(), second.clone()]),
            host: None,
            server_id: None,
            max_frame_length: None,
            handshake_timeout_ms: None,
            on_error: None,
        });

        let error = test.server.start().await.expect_err("startup fails");
        assert!(
            same_failure(&error, &failure),
            "the startup failure rethrows verbatim when cleanup succeeds: {error}"
        );
        assert_eq!(first.close_count.get(), 1);
        assert_eq!(second.close_count.get(), 0);
    });
}

/// Whether two opaque failures are the same routed error.
fn same_failure(left: &Failure, right: &Failure) -> bool {
    match (left, right) {
        (Failure::Other(left), Failure::Other(right)) => Rc::ptr_eq(left, right),
        (Failure::Server(left), Failure::Server(right)) => left == right,
        (Failure::Remote(left), Failure::Remote(right)) => left == right,
        (Failure::Validation(left), Failure::Validation(right)) => left == right,
        _ => false,
    }
}
