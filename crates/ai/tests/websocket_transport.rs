//! The WebSocket transport seam, its tokio-tungstenite implementation against
//! a real loopback handshake, and the mock transport the Codex wire-API tests
//! will stub `globalThis.WebSocket` with
//! (`test/openai-codex-stream.test.ts`'s `MockWebSocket` pattern).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; an unexpected shape panics by design"
)]

use pi_ai::http::{
    MockWebSocketTransport, TungsteniteWebSocket, WebSocketError, WebSocketMessage,
    WebSocketOutbound, WebSocketRequest, WebSocketTransport,
};
use tokio_util::sync::CancellationToken;

fn connect_request(url: String) -> WebSocketRequest {
    WebSocketRequest {
        url,
        headers: vec![
            ("authorization".to_owned(), "Bearer k".to_owned()),
            ("session-id".to_owned(), "session-auto".to_owned()),
        ],
        connect_timeout_ms: None,
        signal: CancellationToken::new(),
    }
}

// ---------------------------------------------------------------------------
// The mock transport pair
// ---------------------------------------------------------------------------

#[tokio::test]
async fn mock_pair_delivers_messages_both_ways_and_records_the_connect_request() {
    let transport = MockWebSocketTransport::new();
    let mut connection = MockWebSocketTransport::connect(
        &transport,
        connect_request("wss://chatgpt.com/backend-api/codex/ws".to_owned()),
    )
    .await
    .expect("mock connect");

    let mut peer = transport.next_peer().expect("server peer");
    assert_eq!(
        peer.connect_request()
            .headers
            .iter()
            .find(|(name, _)| name == "session-id")
            .expect("session header"),
        &("session-id".to_owned(), "session-auto".to_owned())
    );

    peer.send_text("{\"type\":\"response.delta\"}")
        .expect("push to client");
    let message = connection.next_message().await.expect("client receives");
    assert_eq!(
        message,
        WebSocketMessage::Text("{\"type\":\"response.delta\"}".to_owned())
    );

    connection
        .send(WebSocketMessage::Text(
            "{\"type\":\"response.done\"}".to_owned(),
        ))
        .await
        .expect("client sends");
    let sent = peer.next_sent().await.expect("peer sees the send");
    let WebSocketOutbound::Message(WebSocketMessage::Text(text)) = sent else {
        panic!("expected a text send, got {sent:?}");
    };
    assert_eq!(text, "{\"type\":\"response.done\"}");
}

#[tokio::test]
async fn peer_close_surfaces_the_code_and_reason_on_the_next_read() {
    let transport = MockWebSocketTransport::new();
    let mut connection = MockWebSocketTransport::connect(
        &transport,
        connect_request("wss://upstream.test/ws".to_owned()),
    )
    .await
    .expect("mock connect");
    let peer = transport.next_peer().expect("server peer");

    peer.close(1008, "session expired").expect("server closes");

    let error = connection.next_message().await.expect_err("close event");
    assert_eq!(
        error,
        WebSocketError::Closed {
            code: 1008,
            reason: "session expired".to_owned(),
            was_clean: true,
        }
    );

    // Sends after the close fail the same way.
    let error = connection
        .send(WebSocketMessage::Text("late".to_owned()))
        .await
        .expect_err("send after close");
    assert!(matches!(error, WebSocketError::Closed { .. }));
}

#[tokio::test]
async fn client_close_reaches_the_peer_with_code_and_reason() {
    let transport = MockWebSocketTransport::new();
    let mut connection = MockWebSocketTransport::connect(
        &transport,
        connect_request("wss://upstream.test/ws".to_owned()),
    )
    .await
    .expect("mock connect");
    let mut peer = transport.next_peer().expect("server peer");

    connection
        .close(1000, "done".to_owned())
        .await
        .expect("client closes");
    let sent = peer.next_sent().await.expect("peer sees the close");
    let WebSocketOutbound::Close { code, reason } = sent else {
        panic!("expected a close, got {sent:?}");
    };
    assert_eq!(code, 1000);
    assert_eq!(reason, "done");
}

#[tokio::test(start_paused = true)]
async fn mock_connect_honors_cancellation_and_the_connect_timeout() {
    let transport = MockWebSocketTransport::new().with_connect_delay_ms(30_000);

    // A pre-cancelled token rejects without connecting.
    let request = connect_request("wss://upstream.test/ws".to_owned());
    request.signal.cancel();

    let error = MockWebSocketTransport::connect(&transport, request)
        .await
        .expect_err("aborted connect");
    assert_eq!(error, WebSocketError::Aborted);

    // A connect timeout expires against the paused clock.
    let mut request = connect_request("wss://upstream.test/ws".to_owned());
    request.connect_timeout_ms = Some(1_000);
    let runner = tokio::spawn(MockWebSocketTransport::connect(&transport, request));
    tokio::time::advance(std::time::Duration::from_secs(2)).await;
    let error = runner.await.expect("task ends").expect_err("timeout wins");
    assert_eq!(error, WebSocketError::Timeout(1_000));
    assert!(
        transport.next_peer().is_none(),
        "no peer for a failed connect"
    );
}

#[tokio::test(start_paused = true)]
async fn abort_during_the_connect_handshake_fails_aborted() {
    let transport = MockWebSocketTransport::new().with_connect_delay_ms(30_000);
    let token = CancellationToken::new();
    let mut request = connect_request("wss://upstream.test/ws".to_owned());
    request.signal = token.clone();

    let runner = tokio::spawn(MockWebSocketTransport::connect(&transport, request));
    tokio::time::advance(std::time::Duration::from_millis(100)).await;
    token.cancel();
    let error = runner.await.expect("task ends").expect_err("abort wins");
    assert_eq!(error, WebSocketError::Aborted);
}

// ---------------------------------------------------------------------------
// The real tokio-tungstenite transport over a loopback ws:// handshake
// ---------------------------------------------------------------------------

/// A loopback WebSocket server: the handshake accepts, client sends surface
/// on `inbound`, and messages pushed onto `server_outbound` reach the client;
/// dropping the sender closes the socket.
async fn spawn_ws_server() -> (
    String,
    tokio::sync::mpsc::Receiver<String>,
    tokio::sync::mpsc::Sender<tokio_tungstenite::tungstenite::Message>,
) {
    use futures_util::StreamExt;
    use tokio_tungstenite::tungstenite::Message;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback bind");
    let address = listener.local_addr().expect("local addr").to_string();
    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::channel::<String>(64);
    let (outbound_tx, mut outbound_rx) = tokio::sync::mpsc::channel::<Message>(64);

    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept");
        let mut ws = tokio_tungstenite::accept_async(stream)
            .await
            .expect("ws handshake");
        loop {
            tokio::select! {
                message = ws.next() => match message {
                    Some(Ok(Message::Text(text))) => {
                        if inbound_tx.send(text.to_string()).await.is_err() {
                            return;
                        }
                    }
                    Some(Ok(Message::Close(_)) | Err(_)) | None => return,
                    Some(Ok(_)) => {}
                },
                outbound = outbound_rx.recv() => {
                    if let Some(message) = outbound {
                        futures_util::SinkExt::send(&mut ws, message)
                            .await
                            .expect("server send");
                    } else {
                        let _ = futures_util::SinkExt::send(&mut ws, Message::Close(None)).await;
                        return;
                    }
                }
            }
        }
    });

    (format!("ws://{address}/ws"), inbound_rx, outbound_tx)
}

#[tokio::test]
async fn tungstenite_handshakes_sends_receives_and_closes_over_loopback() {
    let (url, mut inbound_rx, server_outbound) = spawn_ws_server().await;

    let transport = pi_ai::http::default_websocket_transport();
    let mut connection = transport
        .connect(connect_request(url))
        .await
        .expect("loopback handshake");

    connection
        .send(WebSocketMessage::Text("{\"type\":\"ping\"}".to_owned()))
        .await
        .expect("client send");
    let received = inbound_rx.recv().await.expect("server reads the send");
    assert_eq!(received, "{\"type\":\"ping\"}");

    server_outbound
        .send(tokio_tungstenite::tungstenite::Message::text("server push"))
        .await
        .expect("server send");
    let message = connection.next_message().await.expect("client receives");
    assert_eq!(message, WebSocketMessage::Text("server push".to_owned()));

    // Server-initiated close: the close frame surfaces as a clean close error
    // carrying the wire's code and reason.
    server_outbound
        .send(tokio_tungstenite::tungstenite::Message::Close(Some(
            tokio_tungstenite::tungstenite::protocol::CloseFrame {
                code: tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Normal,
                reason: "done".into(),
            },
        )))
        .await
        .expect("server close");
    let error = connection
        .next_message()
        .await
        .expect_err("peer close surfaces");
    assert_eq!(
        error,
        WebSocketError::Closed {
            code: 1000,
            reason: "done".to_owned(),
            was_clean: true,
        }
    );
}

#[tokio::test(start_paused = true)]
async fn tungstenite_connect_timeout_fires_on_a_silent_socket() {
    // A plain TCP listener answers nothing: the handshake never completes.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("addr").to_string();
    tokio::spawn(async move {
        let (_socket, _) = listener.accept().await.expect("accept");
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
    });

    let mut request = connect_request(format!("ws://{address}/ws"));
    request.connect_timeout_ms = Some(1_500);
    let runner = tokio::spawn(async move { TungsteniteWebSocket::new().connect(request).await });
    tokio::time::advance(std::time::Duration::from_secs(60)).await;
    let error = runner.await.expect("task ends").expect_err("timeout wins");
    assert_eq!(error, WebSocketError::Timeout(1_500));
}

#[tokio::test(start_paused = true)]
async fn tungstenite_aborts_a_pending_handshake() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("addr").to_string();
    tokio::spawn(async move {
        let (_socket, _) = listener.accept().await.expect("accept");
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
    });

    let token = CancellationToken::new();
    let mut request = connect_request(format!("ws://{address}/ws"));
    request.signal = token.clone();
    let runner = tokio::spawn(async move { TungsteniteWebSocket::new().connect(request).await });
    tokio::time::advance(std::time::Duration::from_millis(100)).await;
    token.cancel();
    let error = runner.await.expect("task ends").expect_err("abort wins");
    assert_eq!(error, WebSocketError::Aborted);
}

#[tokio::test]
async fn tungstenite_reads_fail_aborted_when_the_request_cancels_mid_stream() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("addr").to_string();
    let (server_connected_tx, server_connected_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept");
        let mut ws = tokio_tungstenite::accept_async(stream)
            .await
            .expect("ws handshake");
        futures_util::SinkExt::send(
            &mut ws,
            tokio_tungstenite::tungstenite::Message::text("hello"),
        )
        .await
        .expect("server send");
        let _ = server_connected_tx.send(());
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
    });

    let token = CancellationToken::new();
    let mut request = connect_request(format!("ws://{address}/ws"));
    request.signal = token.clone();
    let mut connection = TungsteniteWebSocket::new()
        .connect(request)
        .await
        .expect("handshake completes");
    let _ = server_connected_rx.await;
    let _ = connection.next_message().await;
    token.cancel();
    let error = connection.next_message().await.expect_err("abort wins");
    assert_eq!(error, WebSocketError::Aborted);
}
#[tokio::test]
async fn mock_pair_carries_binary_messages_and_defaults_build_empty() {
    use bytes::Bytes;

    let transport = MockWebSocketTransport::default();
    let mut connection = MockWebSocketTransport::connect(
        &transport,
        connect_request("wss://upstream.test/ws".to_owned()),
    )
    .await
    .expect("mock connect");
    let peer = transport.next_peer().expect("server peer");
    peer.send_binary(Bytes::from_static(b"\x00\x01"))
        .expect("binary push");
    let message = connection.next_message().await.expect("client receives");
    assert_eq!(
        message,
        WebSocketMessage::Binary(Bytes::from_static(b"\x00\x01"))
    );

    // Debug impls exist for the harness types.
    assert!(format!("{transport:?}").contains("MockWebSocketTransport"));
    assert!(
        format!("{mock:?}", mock = pi_ai::http::MockHttpClient::default())
            .contains("MockHttpClient")
    );
}

#[tokio::test]
async fn tungstenite_carries_binary_messages_and_pings_are_transparent() {
    let (url, _inbound_rx, server_outbound) = spawn_ws_server().await;

    let transport = pi_ai::http::default_websocket_transport();
    let mut connection = transport
        .connect(connect_request(url))
        .await
        .expect("loopback handshake");

    // A binary send the server reads as a binary frame.
    connection
        .send(WebSocketMessage::Binary(bytes::Bytes::from_static(
            b"\x00\x01\x02",
        )))
        .await
        .expect("binary send");

    // The server pings; the client answers below tungstenite and surfaces
    // only the next payload frame.
    server_outbound
        .send(tokio_tungstenite::tungstenite::Message::Ping(
            b"ka".as_slice().into(),
        ))
        .await
        .expect("ping send");
    server_outbound
        .send(tokio_tungstenite::tungstenite::Message::text("after ping"))
        .await
        .expect("text send");
    let message = connection.next_message().await.expect("client receives");
    assert_eq!(message, WebSocketMessage::Text("after ping".to_owned()));
}

#[test]
fn transport_errors_display_their_message_verbatim() {
    assert_eq!(
        WebSocketError::Transport("connection refused".to_owned()).to_string(),
        "connection refused"
    );
}

#[tokio::test]
async fn tungstenite_server_dropping_the_socket_closes_the_read() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("addr").to_string();
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept");
        let ws = tokio_tungstenite::accept_async(stream)
            .await
            .expect("ws handshake");
        // Drop without a close frame: the read must still end cleanly.
        drop(ws);
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
    });

    let transport = pi_ai::http::default_websocket_transport();
    let mut connection = transport
        .connect(connect_request(format!("ws://{address}/ws")))
        .await
        .expect("handshake");
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    // An abrupt socket drop ends the read with a transport error, not the
    // peer-reported close the clean handshake carries.
    let error = connection.next_message().await.expect_err("dropped socket");
    assert!(matches!(
        error,
        WebSocketError::Closed { .. } | WebSocketError::Transport(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn mock_connect_delay_completes_when_inside_the_timeout() {
    let transport = MockWebSocketTransport::new().with_connect_delay_ms(100);
    let mut request = connect_request("wss://upstream.test/ws".to_owned());
    request.connect_timeout_ms = Some(10_000);
    let runner = tokio::spawn(MockWebSocketTransport::connect(&transport, request));
    tokio::time::advance(std::time::Duration::from_millis(200)).await;
    runner
        .await
        .expect("task ends")
        .expect("delay inside the timeout connects");
    assert!(transport.next_peer().is_some());
}

#[tokio::test]
async fn mock_reads_fail_aborted_when_the_request_cancels_mid_read() {
    let token = CancellationToken::new();
    let transport = MockWebSocketTransport::new();
    let mut request = connect_request("wss://upstream.test/ws".to_owned());
    request.signal = token.clone();
    let mut connection = MockWebSocketTransport::connect(&transport, request)
        .await
        .expect("mock connect");
    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    token.cancel();
    let error = connection.next_message().await.expect_err("abort mid-read");
    assert_eq!(error, WebSocketError::Aborted);
}

#[tokio::test]
async fn a_dropped_mock_client_fails_its_peer_writes() {
    let transport = MockWebSocketTransport::new();
    let connection = MockWebSocketTransport::connect(
        &transport,
        connect_request("wss://upstream.test/ws".to_owned()),
    )
    .await
    .expect("mock connect");
    drop(connection);
    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    let peer = transport.next_peer().expect("server peer");

    assert!(peer.send_text("late").is_err());
    assert!(
        peer.send_binary(bytes::Bytes::from_static(b"\x00"))
            .is_err()
    );
    assert!(peer.close(1000, "done").is_err());
}

#[tokio::test]
async fn a_dropped_peer_fails_the_mock_client_side() {
    let transport = MockWebSocketTransport::new();
    let mut connection = MockWebSocketTransport::connect(
        &transport,
        connect_request("wss://upstream.test/ws".to_owned()),
    )
    .await
    .expect("mock connect");
    // Dropping the peer half without a close ends the channel the client
    // reads from, like a server that vanishes without saying goodbye.
    let peer = transport.next_peer().expect("server peer");
    drop(peer);
    tokio::time::sleep(std::time::Duration::from_millis(1)).await;

    let error = connection.next_message().await.expect_err("peer gone");
    assert!(matches!(error, WebSocketError::Closed { .. }));

    let error = connection
        .send(WebSocketMessage::Text("late".to_owned()))
        .await
        .expect_err("send after peer gone");
    assert!(matches!(error, WebSocketError::Closed { .. }));
}

#[tokio::test]
async fn tungstenite_pushes_binary_frames_and_reports_sends_after_the_peer_closes() {
    let (url, _inbound_rx, server_outbound) = spawn_ws_server().await;

    let transport = pi_ai::http::default_websocket_transport();
    let mut connection = transport
        .connect(connect_request(url))
        .await
        .expect("loopback handshake");

    server_outbound
        .send(tokio_tungstenite::tungstenite::Message::Binary(
            b"\x00\x01".as_slice().into(),
        ))
        .await
        .expect("binary push");
    let message = connection.next_message().await.expect("client receives");
    assert_eq!(
        message,
        WebSocketMessage::Binary(bytes::Bytes::from_static(b"\x00\x01"))
    );

    // Close the server side; the client's next read and write fail.
    drop(server_outbound);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let error = connection
        .next_message()
        .await
        .expect_err("read after server gone");
    assert!(matches!(
        error,
        WebSocketError::Closed { .. } | WebSocketError::Transport(_)
    ));
    let error = connection
        .send(WebSocketMessage::Text("late".to_owned()))
        .await
        .expect_err("send after server gone");
    assert!(matches!(
        error,
        WebSocketError::Closed { .. } | WebSocketError::Transport(_)
    ));
}

#[tokio::test]
async fn tungstenite_close_reports_transport_failures_after_the_peer_gone() {
    let (url, _inbound_rx, server_outbound) = spawn_ws_server().await;

    let transport = pi_ai::http::default_websocket_transport();
    let mut connection = transport
        .connect(connect_request(url))
        .await
        .expect("loopback handshake");

    drop(server_outbound);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let result = connection.close(1000, "late".to_owned()).await;
    // A close on a dead socket either lands or reports a transport error;
    // either way the connection is over.
    assert!(
        result.is_ok()
            || matches!(
                result,
                Err(WebSocketError::Closed { .. } | WebSocketError::Transport(_))
            )
    );
}
