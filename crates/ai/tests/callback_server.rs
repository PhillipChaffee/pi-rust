//! Rust-native coverage of the loopback callback server
//! (`crates/ai/src/auth/oauth/callback.rs`) — upstream builds the same
//! mechanics inline per flow with `node:http` in
//! `packages/ai/src/auth/oauth/{anthropic,openai-codex,openrouter,radius}.ts`
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, with no standalone
//! suite to port. The cases drive a real request over a loopback TCP
//! connection and pin the `WaitCell` settle semantics the login races ride.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use common::oauth_fixtures::send_loopback;
use pi_ai::auth::oauth::callback::{
    CallbackHandler, CallbackRequest, CallbackResponse, OAuthCallbackServer, WaitCell,
};
use pi_ai::auth::oauth::oauth_page::{oauth_error_html, oauth_success_html};

/// The success page a callback answers with.
const SUCCESS_PAGE: &str = "OAuth callback accepted";

/// A handler that records every request and answers the success page, the
/// recorder the parse and route cases drive.
fn recording_handler() -> (CallbackHandler, Arc<Mutex<Vec<CallbackRequest>>>) {
    let requests: Arc<Mutex<Vec<CallbackRequest>>> = Arc::default();
    let handler: CallbackHandler = {
        let requests = Arc::clone(&requests);
        Arc::new(move |request| {
            requests
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(request);
            Box::pin(std::future::ready(CallbackResponse {
                status: 200,
                html: SUCCESS_PAGE.to_owned(),
            }))
        })
    };
    (handler, requests)
}

/// A handler that answers every request with the success page, the stub the
/// error-page cases ride.
fn static_handler() -> CallbackHandler {
    Arc::new(|_| {
        Box::pin(std::future::ready(CallbackResponse {
            status: 200,
            html: SUCCESS_PAGE.to_owned(),
        }))
    })
}

/// Bind the callback server on an ephemeral loopback port; the case drives
/// requests against the returned port.
async fn bind_ephemeral(handler: CallbackHandler) -> (OAuthCallbackServer, u16) {
    let server = OAuthCallbackServer::bind("127.0.0.1", 0, handler)
        .await
        .expect("the ephemeral bind succeeds");
    let port = server.local_addr().expect("the bound address").port();
    (server, port)
}

#[tokio::test]
async fn serves_one_callback_with_parsed_path_and_query() {
    let (handler, requests) = recording_handler();
    let (_server, port) = bind_ephemeral(handler).await;
    assert!(port > 0, "port 0 binds an ephemeral port");

    let response = send_loopback(
        "127.0.0.1",
        port,
        "GET /path?code=x&state=y HTTP/1.1\r\nHost: localhost\r\n\r\n",
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 200 OK\r\n"),
        "the status line names the status: {response:?}"
    );
    assert!(
        response.contains(&format!("content-length: {}\r\n", SUCCESS_PAGE.len())),
        "the content-length carries the body size: {response:?}"
    );
    assert!(
        response.ends_with(SUCCESS_PAGE),
        "the handler's page is the body"
    );
    assert!(
        response.contains("content-type: text/html"),
        "the page renders as html"
    );
    assert!(
        response.contains("connection: close"),
        "one request per connection"
    );

    let recorded = requests
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(recorded.len(), 1, "one request reached the handler");
    assert_eq!(recorded[0].path, "/path");
    assert_eq!(recorded[0].query_value("code"), Some("x"));
    assert_eq!(recorded[0].query_value("state"), Some("y"));
}

#[tokio::test]
async fn a_wrong_path_surfaces_the_handler_status() {
    // The handler routes on the callback path, upstream's per-flow
    // `req.url !== path` check.
    let handler: CallbackHandler = Arc::new(|request| {
        let (status, html) = if request.path == "/callback" {
            (200, oauth_success_html("Authentication completed."))
        } else {
            (404, oauth_error_html("Callback route not found.", None))
        };
        Box::pin(std::future::ready(CallbackResponse { status, html }))
    });
    let (_server, port) = bind_ephemeral(handler).await;

    let response = send_loopback(
        "127.0.0.1",
        port,
        "GET /nope HTTP/1.1\r\nHost: localhost\r\n\r\n",
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 404 Not Found\r\n"),
        "the handler's 404 surfaces: {response:?}"
    );
    assert!(
        response.contains("Authentication failed"),
        "the error page is the body: {response:?}"
    );
}

#[tokio::test]
async fn the_wait_cell_settles_once_and_ignores_later_settles() {
    let (cell, receiver) = WaitCell::<String>::new();
    cell.settle(Some(String::from("first"))).await;
    // Later settles — including a cancellation settle — are ignored.
    cell.settle(Some(String::from("second"))).await;
    cell.settle(None).await;
    let settled = receiver.await.expect("the receiver resolves once");
    assert_eq!(settled.as_deref(), Some("first"), "the first settle wins");
}

#[tokio::test]
async fn the_wait_cell_settles_none_before_any_value_and_defaults_are_harmless() {
    // A `None` settle is upstream's `cancelWait`: the receiver resolves with
    // no value, the shape the manual-prompt handover rides.
    let (cell, receiver) = WaitCell::<String>::new();
    cell.settle(None).await;
    assert_eq!(
        receiver.await.expect("the receiver resolves once"),
        None,
        "the cancellation settle carries no value"
    );

    // A fresh cell settles a value after the cancellation settle.
    let (fresh, fresh_receiver) = WaitCell::<String>::new();
    fresh.settle(Some(String::from("value"))).await;
    assert_eq!(
        fresh_receiver.await.expect("the fresh receiver resolves"),
        Some(String::from("value"))
    );

    // The `Default` cell has no waiter, so a settle is a no-op.
    let unshared: WaitCell<String> = WaitCell::default();
    unshared.settle(Some(String::from("dropped"))).await;
}

#[test]
fn the_wait_cell_debug_names_the_cell() {
    let (cell, _receiver) = WaitCell::<u8>::new();
    assert_eq!(format!("{cell:?}"), "WaitCell(..)");
}

#[tokio::test]
async fn the_bound_server_debug_names_the_type_and_drop_stops_accepting() {
    let (server, port) = bind_ephemeral(static_handler()).await;
    let server_debug = format!("{server:?}");
    assert!(
        server_debug.contains("OAuthCallbackServer"),
        "the bound server debug names the type: {server_debug:?}"
    );

    // Explicit close frees the accept loop; the port rebinds once the old
    // listener drops, the `server.close()` contract.
    server.close();
    let rebound = {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
                Ok(listener) => break Some(listener),
                Err(_) if std::time::Instant::now() < deadline => {
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                }
                Err(_) => break None,
            }
        }
    };
    assert!(
        rebound.is_some(),
        "the closed server's port rebinds: {server_debug:?}"
    );
}

#[tokio::test]
async fn an_immediately_closed_connection_is_ignored() {
    let (handler, requests) = recording_handler();
    let (_server, port) = bind_ephemeral(handler).await;

    let mut stream = TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("the loopback server accepts");
    stream
        .shutdown()
        .await
        .expect("the client closes its write side");
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .await
        .expect("the connection ends cleanly");
    assert!(
        raw.is_empty(),
        "an EOF-only connection gets no response: {raw:?}"
    );
    assert!(
        requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty(),
        "an empty connection never reaches the handler"
    );
}

#[tokio::test]
async fn a_partial_request_flushed_at_eof_is_still_answered() {
    let (handler, requests) = recording_handler();
    let (_server, port) = bind_ephemeral(handler).await;

    // No header terminator, then EOF: the buffered request still parses,
    // the lenient-EOF read upstream's servers share.
    let mut stream = TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("the loopback server accepts");
    stream
        .write_all(b"GET /path?code=x HTTP/1.1\r\nHost: localhost\r\n")
        .await
        .expect("the partial request writes");
    stream
        .shutdown()
        .await
        .expect("the client closes its write side");
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .await
        .expect("the response reads to EOF");

    let recorded = requests
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(recorded.len(), 1, "the EOF-flushed request was parsed");
    assert_eq!(recorded[0].path, "/path");
    assert_eq!(recorded[0].query_value("code"), Some("x"));
}

#[tokio::test]
async fn an_oversized_request_yields_the_internal_error_page_without_a_panic() {
    let (_server, port) = bind_ephemeral(static_handler()).await;

    // The server answers once its buffer crosses 16 KiB; a total of exactly
    // 16 KiB + 1 keeps every written byte consumed by the read that trips
    // the limit, so the close is a clean FIN and the page survives.
    let mut request = String::from("GET / HTTP/1.1\r\nHost: localhost\r\nX-Pad: ");
    request.push_str(&"x".repeat(16 * 1024 + 1 - request.len()));
    let response = send_loopback("127.0.0.1", port, &request).await;
    assert!(
        response.starts_with("HTTP/1.1 500 Internal Server Error\r\n"),
        "the oversized request fails the parse: {response:?}"
    );
    assert!(
        response.ends_with("Internal error"),
        "the internal-error page is the body: {response:?}"
    );
}

#[tokio::test]
async fn a_non_utf8_request_yields_the_internal_error_page() {
    let (_server, port) = bind_ephemeral(static_handler()).await;

    let mut stream = TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("the loopback server accepts");
    stream
        .write_all(b"\xff\xfe GET / HTTP/1.1\r\n\r\n")
        .await
        .expect("the request writes");
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .await
        .expect("the response reads to EOF");
    let response = String::from_utf8_lossy(&raw).into_owned();
    assert!(
        response.starts_with("HTTP/1.1 500 Internal Server Error\r\n"),
        "the non-utf-8 request fails the parse: {response:?}"
    );
}

#[tokio::test]
async fn malformed_request_lines_yield_the_internal_error_page() {
    let (_server, port) = bind_ephemeral(static_handler()).await;

    // No target after the method, then an unparsable target.
    for request in ["GARBAGE\r\n\r\n", "GET %zz HTTP/1.1\r\n\r\n"] {
        let response = send_loopback("127.0.0.1", port, request).await;
        assert!(
            response.starts_with("HTTP/1.1 500 Internal Server Error\r\n"),
            "the malformed request line fails the parse: {response:?}"
        );
    }
}

#[tokio::test]
async fn each_error_status_renders_its_reason_phrase() {
    let handler: CallbackHandler = Arc::new(|request| {
        let status = match request.path.as_str() {
            "/bad" => 400,
            "/missing" => 404,
            "/conflict" => 409,
            "/gateway" => 502,
            _ => 200,
        };
        Box::pin(std::future::ready(CallbackResponse {
            status,
            html: "page".to_owned(),
        }))
    });
    let (_server, port) = bind_ephemeral(handler).await;

    for (path, status_line) in [
        ("/bad", "HTTP/1.1 400 Bad Request\r\n"),
        ("/missing", "HTTP/1.1 404 Not Found\r\n"),
        ("/conflict", "HTTP/1.1 409 Conflict\r\n"),
        ("/gateway", "HTTP/1.1 502 Bad Gateway\r\n"),
    ] {
        let response =
            send_loopback("127.0.0.1", port, &format!("GET {path} HTTP/1.1\r\n\r\n")).await;
        assert!(
            response.starts_with(status_line),
            "{path} renders {status_line:?}: {response:?}"
        );
    }
}

#[tokio::test]
async fn a_bound_port_rejects_a_second_bind_with_the_bind_error() {
    let handler: CallbackHandler = Arc::new(|_| {
        Box::pin(std::future::ready(CallbackResponse {
            status: 200,
            html: SUCCESS_PAGE.to_owned(),
        }))
    });
    let holder = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("the ephemeral holder binds");
    let port = holder.local_addr().expect("the holder address").port();

    let error = OAuthCallbackServer::bind("127.0.0.1", port, handler)
        .await
        .expect_err("the occupied port fails the bind");
    assert!(
        error
            .to_string()
            .starts_with("could not bind the OAuth callback server on 127.0.0.1"),
        "the bind error names the host and port: {error:?}"
    );
}

#[test]
fn the_error_page_escapes_every_html_special_character() {
    let page = oauth_error_html("&<>\"' marker", Some(r#"details <&>"'"#));
    assert!(page.contains("&amp;"), "ampersand escapes: {page:?}");
    assert!(page.contains("&lt;"), "less-than escapes: {page:?}");
    assert!(page.contains("&gt;"), "greater-than escapes: {page:?}");
    assert!(page.contains("&quot;"), "double quote escapes: {page:?}");
    assert!(page.contains("&#39;"), "apostrophe escapes: {page:?}");
    // The details block escapes with the message.
    assert!(
        page.contains(r#"<div class="details">details &lt;&amp;&gt;&quot;&#39;</div>"#),
        "the details escape too: {page:?}"
    );
    assert!(
        oauth_success_html("plain").contains("Authentication successful"),
        "the success page renders its heading"
    );
}

#[tokio::test]
async fn garbage_flushed_at_eof_yields_the_internal_error_page() {
    let (_server, port) = bind_ephemeral(static_handler()).await;

    // Unparsable bytes with no header terminator, then EOF: the buffered
    // request fails the parse, upstream's lenient-EOF read.
    let mut stream = TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("the loopback server accepts");
    stream
        .write_all(b"\r\nnot a request")
        .await
        .expect("the garbage writes");
    stream
        .shutdown()
        .await
        .expect("the client closes its write side");
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .await
        .expect("the response reads to EOF");
    let response = String::from_utf8_lossy(&raw).into_owned();
    assert!(
        response.starts_with("HTTP/1.1 500 Internal Server Error\r\n"),
        "the unparsable EOF buffer fails the parse: {response:?}"
    );
}

#[tokio::test]
async fn a_parked_connection_is_dropped_when_the_server_closes() {
    let (handler, requests) = recording_handler();
    let (server, port) = bind_ephemeral(handler).await;

    // Park a connection mid-request: no terminator, no EOF, so the read
    // never completes.
    let mut stream = TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("the loopback server accepts");
    stream
        .write_all(b"GET /path HTTP/1.1\r\nHost: localhost\r\n")
        .await
        .expect("the partial request writes");
    // Give the connection task a turn to park in the read.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    // The close drops the parked connection without answering it, upstream's
    // `server.close()` semantics for in-flight requests.
    server.close();
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    assert!(
        requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty(),
        "the parked connection never reached the handler"
    );
}
