//! 4-byte big-endian length-prefix framing, ported from upstream
//! `src/framing.ts` 1:1.

use std::fmt;

const FRAME_HEADER_LENGTH: usize = 4;
const MAX_UINT32: u64 = 0xffff_ffff;

/// Default upper bound for one framed CBOR payload.
pub const DEFAULT_MAX_FRAME_LENGTH: usize = 16 * 1024 * 1024;

/// Options for the frame decoder, ported from upstream's
/// `FrameDecoderOptions`.
///
/// Upstream spells the limit as an optional number defaulting to the safe
/// default; the plain field with [`Default`] carries the same contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameDecoderOptions {
    /// Maximum payload length one decoded frame may carry.
    pub max_frame_length: usize,
}

impl Default for FrameDecoderOptions {
    fn default() -> Self {
        Self {
            max_frame_length: DEFAULT_MAX_FRAME_LENGTH,
        }
    }
}

/// The error the framing layer raises, ported from upstream's `FrameError`.
///
/// Upstream extends `Error` with `name = "FrameError"`; the distinct type is
/// the same discrimination. Configuration-range rejections that upstream
/// spells as `RangeError` carry the same message through this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameError {
    message: String,
}

impl FrameError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The error message, the upstream `Error.message` surface.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for FrameError {}

/// Prefixes a payload with its unsigned 32-bit big-endian byte length.
///
/// # Errors
///
/// Returns [`FrameError`] when the payload exceeds the unsigned 32-bit
/// length limit; upstream throws a `RangeError` with the same message.
pub fn encode_frame(payload: &[u8]) -> Result<Vec<u8>, FrameError> {
    if payload.len() as u64 > MAX_UINT32 {
        return Err(FrameError::new(
            "Frame payload exceeds the unsigned 32-bit length limit",
        ));
    }
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the guard above bounds the length inside u32"
    )]
    let length = payload.len() as u32;
    let mut frame = Vec::with_capacity(FRAME_HEADER_LENGTH + payload.len());
    frame.extend_from_slice(&length.to_be_bytes());
    frame.extend_from_slice(payload);
    Ok(frame)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecoderState {
    Open,
    Ended,
    Failed,
}

/// Incrementally splits arbitrary byte chunks into length-prefixed payloads.
///
/// The decoder accepts chunks at any byte boundary, copies payload bytes
/// rather than aliasing input, and latches the failed state: after any
/// framing error every later [`push`](Self::push) and
/// [`end`](Self::end) call fails too. Upstream assembles payloads through
/// 64 KiB blocks; one growable buffer carries the same copy-not-alias and
/// chunk-boundary independence while allocation stays bounded by the bytes
/// actually received, not by the declared length.
#[derive(Debug)]
pub struct FrameDecoder {
    header: [u8; FRAME_HEADER_LENGTH],
    header_length: usize,
    max_frame_length: usize,
    payload: Vec<u8>,
    expected_payload_length: Option<usize>,
    state: DecoderState,
}

impl FrameDecoder {
    /// A decoder bounded by `options.max_frame_length`.
    ///
    /// # Errors
    ///
    /// Returns [`FrameError`] when `max_frame_length` exceeds the unsigned
    /// 32-bit range; upstream throws a `RangeError` with the same message,
    /// and the `usize` field makes non-integer inputs unrepresentable.
    pub fn new(options: FrameDecoderOptions) -> Result<Self, FrameError> {
        if options.max_frame_length as u64 > MAX_UINT32 {
            return Err(FrameError::new(format!(
                "maxFrameLength must be an integer between 0 and {MAX_UINT32}"
            )));
        }
        Ok(Self {
            header: [0; FRAME_HEADER_LENGTH],
            header_length: 0,
            max_frame_length: options.max_frame_length,
            payload: Vec::new(),
            expected_payload_length: None,
            state: DecoderState::Open,
        })
    }

    /// Feeds one chunk, returning the frames it completes in order.
    ///
    /// # Errors
    ///
    /// Returns [`FrameError`] when the decoder has ended or failed, or when
    /// a completed header declares a length past the configured limit; the
    /// latter latches the failed state.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<Vec<u8>>, FrameError> {
        match self.state {
            DecoderState::Ended => return Err(FrameError::new("Frame decoder has ended")),
            DecoderState::Failed => return Err(FrameError::new("Frame decoder has failed")),
            DecoderState::Open => {}
        }

        let mut frames = Vec::new();
        let mut chunk_offset = 0;
        while chunk_offset < chunk.len() {
            if self.expected_payload_length.is_none() {
                let header_bytes =
                    (FRAME_HEADER_LENGTH - self.header_length).min(chunk.len() - chunk_offset);
                self.header[self.header_length..self.header_length + header_bytes]
                    .copy_from_slice(&chunk[chunk_offset..chunk_offset + header_bytes]);
                self.header_length += header_bytes;
                chunk_offset += header_bytes;
                if self.header_length < FRAME_HEADER_LENGTH {
                    continue;
                }

                #[allow(
                    clippy::indexing_slicing,
                    reason = "header is exactly FRAME_HEADER_LENGTH bytes long"
                )]
                let frame_length = u32::from_be_bytes(self.header) as usize;
                self.header_length = 0;
                if frame_length > self.max_frame_length {
                    return Err(self.fail(&format!(
                        "Frame length {frame_length} exceeds configured limit of {}",
                        self.max_frame_length
                    )));
                }
                if frame_length == 0 {
                    frames.push(Vec::new());
                    continue;
                }
                self.expected_payload_length = Some(frame_length);
                self.payload.clear();
            }

            let Some(expected_payload_length) = self.expected_payload_length else {
                continue;
            };
            while chunk_offset < chunk.len() && self.payload.len() < expected_payload_length {
                let payload_bytes =
                    (expected_payload_length - self.payload.len()).min(chunk.len() - chunk_offset);
                self.payload
                    .extend_from_slice(&chunk[chunk_offset..chunk_offset + payload_bytes]);
                chunk_offset += payload_bytes;
            }
            if self.payload.len() == expected_payload_length {
                frames.push(std::mem::take(&mut self.payload));
                self.expected_payload_length = None;
            }
        }
        Ok(frames)
    }

    /// Ends the stream, rejecting a truncated frame mid-assembly.
    ///
    /// # Errors
    ///
    /// Returns [`FrameError`] when the decoder has ended or failed, or when
    /// a partial header or payload remains; those latches stay failed.
    pub fn end(&mut self) -> Result<(), FrameError> {
        match self.state {
            DecoderState::Ended => return Err(FrameError::new("Frame decoder has ended")),
            DecoderState::Failed => return Err(FrameError::new("Frame decoder has failed")),
            DecoderState::Open => {}
        }
        if self.header_length != 0 || self.expected_payload_length.is_some() {
            return Err(self.fail("Truncated frame at end of stream"));
        }
        self.state = DecoderState::Ended;
        Ok(())
    }

    fn fail(&mut self, message: &str) -> FrameError {
        self.state = DecoderState::Failed;
        self.header_length = 0;
        self.payload.clear();
        self.expected_payload_length = None;
        FrameError::new(message)
    }
}
