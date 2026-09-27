//! The `TextLineReader` suite, ported 1:1 from upstream
//! `test/harness/text-line-reader.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, over the nodejs execution
//! environment the foundations child ported.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use pi_agent_core::harness::context::{background_context, with_abort_signal, with_cancel};
use pi_agent_core::harness::env::nodejs::NodeExecutionEnv;
use pi_agent_core::harness::types::{FileErrorCode, FileSystem, TextLine, TextLineReader};

fn temp_root() -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("temp root");
    let path = dir.path().to_string_lossy().to_string();
    (dir, path)
}

/// The reader pool the test closes after each case, upstream's `readers`.
struct ReaderPool {
    _guard: tempfile::TempDir,
    root: String,
    env: NodeExecutionEnv,
    readers: std::sync::Mutex<Vec<Box<dyn TextLineReader>>>,
}

impl ReaderPool {
    fn new() -> Self {
        let (guard, root) = temp_root();
        Self {
            _guard: guard,
            env: NodeExecutionEnv::new(root.clone(), None, None),
            root,
            readers: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Write the fixture and open a reader over it, upstream's
    /// `openReader(content)`.
    async fn open_reader(&self, content: &str) -> Box<dyn TextLineReader> {
        let path = std::path::Path::new(&self.root).join("text.txt");
        std::fs::write(&path, content).expect("fixture write");
        let reader = self
            .env
            .open_text_line_reader("text.txt", &background_context())
            .await
            .expect("reader open");
        self.readers.lock().expect("readers").push(reader);
        self.readers
            .lock()
            .expect("readers")
            .pop()
            .expect("pushed reader")
    }

    /// Read every remaining line, upstream's `readLines(reader)`.
    async fn read_lines(&self, reader: &mut dyn TextLineReader) -> Vec<TextLine> {
        let mut lines = Vec::new();
        loop {
            match reader.read_line(&background_context()).await.expect("line") {
                Some(line) => lines.push(line),
                None => return lines,
            }
        }
    }

    /// The missing-file open error's path, upstream's `join(root, ...)`.
    fn root(&self) -> &str {
        &self.root
    }
}

#[tokio::test]
async fn decodes_unicode_blank_lines_and_a_torn_final_line() {
    let pool = ReaderPool::new();
    let mut reader = pool.open_reader("hé🙂\n\n\n終\ntorn").await;
    assert_eq!(
        pool.read_lines(&mut *reader).await,
        [
            TextLine {
                text: "hé🙂".to_owned(),
                terminated: true
            },
            TextLine {
                text: String::new(),
                terminated: true
            },
            TextLine {
                text: String::new(),
                terminated: true
            },
            TextLine {
                text: "終".to_owned(),
                terminated: true
            },
            TextLine {
                text: "torn".to_owned(),
                terminated: false
            },
        ],
    );
    assert!(
        reader
            .read_line(&background_context())
            .await
            .expect("eof")
            .is_none()
    );
}

#[tokio::test]
async fn reads_an_empty_file() {
    let pool = ReaderPool::new();
    let mut reader = pool.open_reader("").await;
    assert!(pool.read_lines(&mut *reader).await.is_empty());
}

#[tokio::test]
async fn decodes_multibyte_characters_split_across_64_kib_chunks() {
    let pool = ReaderPool::new();
    let first = format!("{}🙂{}\n", "a".repeat(64 * 1024 - 1), "é".repeat(40_000));
    let mut reader = pool.open_reader(&format!("{first}終")).await;
    assert_eq!(
        pool.read_lines(&mut *reader).await,
        [
            TextLine {
                text: first[..first.len() - 1].to_owned(),
                terminated: true,
            },
            TextLine {
                text: "終".to_owned(),
                terminated: false,
            },
        ],
    );
}

#[tokio::test]
async fn replaces_malformed_and_incomplete_utf8() {
    let pool = ReaderPool::new();
    let path = std::path::Path::new(pool.root()).join("text.txt");
    std::fs::write(&path, [0xff_u8, 0x0a, 0xe2, 0x82]).expect("fixture write");
    let mut reader = pool
        .env
        .open_text_line_reader("text.txt", &background_context())
        .await
        .expect("reader open");
    assert_eq!(
        pool.read_lines(&mut *reader).await,
        [
            TextLine {
                text: "\u{fffd}".to_owned(),
                terminated: true
            },
            TextLine {
                text: "\u{fffd}".to_owned(),
                terminated: false
            },
        ],
    );
}

#[tokio::test]
async fn rejects_an_open_with_a_pre_aborted_context() {
    let pool = ReaderPool::new();
    std::fs::write(std::path::Path::new(pool.root()).join("text.txt"), "one\n")
        .expect("fixture write");
    let (_, controller) = with_cancel(&background_context());
    controller.abort_without_reason();
    let context = with_abort_signal(controller.signal().clone(), &background_context());
    let opened = pool.env.open_text_line_reader("text.txt", &context).await;
    assert_eq!(
        opened.err().expect("aborted open").code,
        FileErrorCode::Aborted,
    );
}

#[tokio::test]
async fn does_not_consume_a_buffered_line_when_its_context_is_pre_aborted() {
    let pool = ReaderPool::new();
    let mut reader = pool.open_reader("one\ntwo\n").await;
    reader
        .read_line(&background_context())
        .await
        .expect("first")
        .expect("line");
    let (_, controller) = with_cancel(&background_context());
    controller.abort_without_reason();
    let context = with_abort_signal(controller.signal().clone(), &background_context());
    let aborted = reader.read_line(&context).await;
    assert_eq!(
        aborted.expect_err("aborted read").code,
        FileErrorCode::Aborted
    );
    assert_eq!(
        reader
            .read_line(&background_context())
            .await
            .expect("retry")
            .expect("buffered line")
            .text,
        "two",
    );
}

#[tokio::test]
async fn honors_cancellation_during_a_read_and_allows_retry() {
    let pool = ReaderPool::new();
    let mut reader = pool
        .open_reader(&format!("{}\nlast", "é".repeat(70_000)))
        .await;
    let (context, controller) = with_cancel(&background_context());
    let pending = reader.read_line(&context);
    controller.abort_without_reason();
    let pending = pending.await;
    assert_eq!(
        pending.expect_err("aborted read").code,
        FileErrorCode::Aborted
    );
    let lines = pool.read_lines(&mut *reader).await;
    assert_eq!(
        lines
            .iter()
            .map(|line| line.text.clone())
            .collect::<Vec<_>>(),
        ["é".repeat(70_000), "last".to_owned()],
    );
}

#[tokio::test]
async fn closes_idempotently_even_with_an_aborted_context_and_rejects_later_reads() {
    let pool = ReaderPool::new();
    let mut reader = pool.open_reader("one\ntwo\n").await;
    reader
        .read_line(&background_context())
        .await
        .expect("first")
        .expect("line");
    let (_, controller) = with_cancel(&background_context());
    controller.abort_without_reason();
    let context = with_abort_signal(controller.signal().clone(), &background_context());
    reader.close(&context).await;
    reader.close(&background_context()).await;
    let rejected = reader.read_line(&background_context()).await;
    assert_eq!(
        rejected.expect_err("closed reader").code,
        FileErrorCode::Invalid,
    );
}

#[tokio::test]
async fn returns_a_file_error_for_a_missing_file() {
    let pool = ReaderPool::new();
    let missing = pool
        .env
        .open_text_line_reader("missing.txt", &background_context())
        .await;
    let error = missing.err().expect("missing file");
    assert_eq!(error.code, FileErrorCode::NotFound);
    assert!(
        error
            .path
            .as_deref()
            .is_some_and(|path| path.ends_with("missing.txt"))
    );
}
