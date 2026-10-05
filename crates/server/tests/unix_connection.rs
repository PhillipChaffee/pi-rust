//! The Unix-connection suite, ported from upstream `test/
//! unix-connection.test.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//! Upstream drives a `ControlledSocket` whose write callbacks settle by
//! hand; the port restates the contract on a real socket pair: the final
//! chunk arrives behind every pending write, in order, and the close
//! resolves once the socket is closed.

#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test assertions panic at the failing case only; the restriction lints target production code"
)]

mod support;

use std::rc::Rc;

use pi_protocol::{
    ProtocolError, ServerHelloError, ServerMessage, ServerMessageDecoder, encode_server_message,
};
use pi_server::ByteConnection;
use pi_server::unix::UnixByteConnection;

use support::run_local;

#[test]
fn queues_a_final_protocol_error_behind_pending_output_before_closing() {
    run_local(async {
        // A socket pair: the local end drives the connection, the peer end
        // records the byte order.
        let (local, peer) = tokio::net::UnixStream::pair().unwrap();
        let connection = Rc::new(UnixByteConnection::new(local, 1_000, 64 * 1024));
        // The driver task the listener would spawn; the handler is the
        // no-op the transport verification uses.
        connection.drive(pi_server::ByteConnectionHandler {
            on_data: Rc::new(|_| {}),
            on_close: Rc::new(|| {}),
            on_error: Rc::new(|_| {}),
        });

        let pending = connection.send(vec![1, 2, 3]);
        // The pending write is in flight; the close queues behind it.
        let final_message = ServerMessage::HelloError(ServerHelloError {
            error: ProtocolError {
                code: "invalid_request".to_string(),
                message: "Protocol violation".to_string(),
            },
        });
        let final_frame = encode_server_message(
            &final_message,
            pi_protocol::FrameDecoderOptions {
                max_frame_length: 1024,
            },
        )
        .unwrap();
        let closing = connection.close(Some(final_frame.clone()));

        // The peer reads the pending chunk first, then the final frame, then
        // the end-of-stream the close's shutdown produces.
        let mut peer = peer;
        let mut buffer = vec![0u8; 64];
        let mut received = Vec::new();
        loop {
            let read = tokio::io::AsyncReadExt::read(&mut peer, &mut buffer)
                .await
                .unwrap();
            if read == 0 {
                break;
            }
            received.extend_from_slice(&buffer[..read]);
        }
        pending.await.expect("the pending write settles");
        closing.await.expect("the close settles");

        assert_eq!(&received[..3], &[1, 2, 3], "the pending write lands first");
        let decoded = ServerMessageDecoder::new(pi_protocol::FrameDecoderOptions {
            max_frame_length: 1024,
        })
        .unwrap()
        .push(&received[3..])
        .unwrap();
        assert_eq!(decoded, vec![final_message]);
        assert!(connection.closed());
    });
}
