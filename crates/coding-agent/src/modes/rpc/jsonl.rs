//! Strict LF-only JSONL framing, upstream's `src/modes/rpc/jsonl.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Framing is LF-only. Payload strings may contain other Unicode separators
//! such as U+2028 and U+2029, so the decoder never splits on them the way
//! Node's readline does.

use tokio::io::AsyncRead;
use tokio::io::AsyncReadExt;
use tokio::sync::mpsc;

/// Serialize one strict JSONL record: compact JSON plus a trailing LF.
#[must_use]
pub fn serialize_json_line(value: &serde_json::Value) -> String {
    format!("{value}\n")
}

/// The LF-only line decoder behind `attachJsonlLineReader`.
///
/// Bytes are buffered until a `\n` arrives, so a chunk boundary inside a
/// multi-byte character is harmless; an incomplete final line decodes with
/// the replacement characters Node's `StringDecoder` produces.
#[derive(Debug, Default)]
pub struct JsonlLineDecoder {
    buffer: Vec<u8>,
}

impl JsonlLineDecoder {
    /// Feed one byte chunk, returning every line it completed.
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(chunk);
        let mut lines = Vec::new();
        while let Some(index) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=index).collect();
            line.pop();
            lines.push(emit_line(&line));
        }
        lines
    }

    /// Flush the unterminated tail, if any.
    pub fn finish(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }
        let line = std::mem::take(&mut self.buffer);
        Some(emit_line(&line))
    }
}

/// Strip one trailing `\r`, upstream's `line.endsWith("\r")` guard.
fn emit_line(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let stripped = text.strip_suffix('\r').map(str::to_owned);
    stripped.unwrap_or_else(|| text.into_owned())
}

/// Attach the decoder to a stream, sending each line to `sender` and closing
/// it at EOF, upstream's `attachJsonlLineReader`.
///
/// A read failure ends the stream the way EOF does; upstream has no error arm
/// for the reader either.
pub async fn attach_jsonl_line_reader<R>(mut reader: R, sender: mpsc::UnboundedSender<String>)
where
    R: AsyncRead + Unpin,
{
    let mut decoder = JsonlLineDecoder::default();
    let mut chunk = vec![0_u8; 8192];
    loop {
        match reader.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(read) => {
                for line in decoder.feed(&chunk[..read]) {
                    if sender.send(line).is_err() {
                        return;
                    }
                }
            }
        }
    }
    if let Some(line) = decoder.finish() {
        let _ = sender.send(line);
    }
}
