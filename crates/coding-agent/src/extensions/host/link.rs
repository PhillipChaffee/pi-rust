//! The session transport: one duplex byte link carrying [`Frame`]s under
//! pi-protocol's framing ([ADR 0010]).
//!
//! The host never reaches a process here. [`Link`] wraps whatever duplex the
//! caller injects — a `tokio::io::duplex` pair in tests, the child's piped
//! stdio in the spawn slice — and owns the frame decoder that turns bytes
//! into [`Frame`]s. A read before a complete frame is an exit, not a
//! malformed frame: the caller decides what a closed link means.
//!
//! [ADR 0010]: ../../../../../docs/adr/0010-extension-host-wire-protocol.md

use std::collections::VecDeque;

use pi_protocol::{FrameDecoder, FrameDecoderOptions};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use super::WireError;
use super::decode::decode_frame;
use super::encode_frame;
use super::frame::Frame;

/// The read window: a frame's bytes arrive in whatever chunks the transport
/// delivers, so the decoder takes the largest chunk this reader hands it.
const READ_CHUNK: usize = 64 * 1024;

/// A framed duplex link over the injected byte stream.
#[derive(Debug)]
pub struct Link<T> {
    io: T,
    decoder: FrameDecoder,
    pending: VecDeque<Vec<u8>>,
}

impl<T> Link<T>
where
    T: AsyncRead + AsyncWrite + Unpin,
{
    /// Wrap one duplex byte stream.
    ///
    /// # Errors
    ///
    /// Returns [`WireError`] when pi-protocol's decoder configuration is
    /// rejected; the 16 MiB default is always accepted.
    pub fn new(io: T) -> Result<Self, WireError> {
        Ok(Self {
            io,
            decoder: FrameDecoder::new(FrameDecoderOptions::default())
                .map_err(|error| WireError::new(error.message()))?,
            pending: VecDeque::new(),
        })
    }

    /// Read the next frame, `None` on a clean end of stream.
    ///
    /// # Errors
    ///
    /// Returns [`WireError`] when the transport fails or when the bytes are
    /// not one of the vocabulary's frames.
    pub async fn read_frame(&mut self) -> Result<Option<Frame>, WireError> {
        loop {
            if let Some(payload) = self.pending.pop_front() {
                return decode_frame(&payload).map(Some);
            }
            let mut buffer = vec![0_u8; READ_CHUNK];
            let read =
                self.io.read(&mut buffer).await.map_err(|error| {
                    WireError::new(format!("extension link read failed: {error}"))
                })?;
            if read == 0 {
                return Ok(None);
            }
            buffer.truncate(read);
            let frames = self
                .decoder
                .push(&buffer)
                .map_err(|error| WireError::new(error.message()))?;
            self.pending.extend(frames);
        }
    }

    /// Write one frame.
    ///
    /// # Errors
    ///
    /// Returns [`WireError`] when the frame does not encode or the transport
    /// fails.
    pub async fn write_frame(&mut self, frame: &Frame) -> Result<(), WireError> {
        let bytes = encode_frame(frame)?;
        self.io
            .write_all(&bytes)
            .await
            .map_err(|error| WireError::new(format!("extension link write failed: {error}")))?;
        self.io
            .flush()
            .await
            .map_err(|error| WireError::new(format!("extension link flush failed: {error}")))
    }

    /// Close the write half; the peer sees end of stream after this.
    ///
    /// # Errors
    ///
    /// Returns [`WireError`] when the transport fails to close.
    pub async fn close(&mut self) -> Result<(), WireError> {
        self.io
            .shutdown()
            .await
            .map_err(|error| WireError::new(format!("extension link close failed: {error}")))
    }
}
