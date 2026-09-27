//! The atomic-publication and JSONL-codec suite, ported 1:1 from upstream
//! `test/harness/jsonl-io.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::Arc;

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::commit::CommittedWrite;
use pi_agent_core::harness::session::jsonl::io::{
    committed_write_wire, publish_file_atomically, publish_jsonl,
};
use pi_agent_core::harness::session::jsonl::types::JsonlStorageHeader;
use pi_agent_core::harness::session::values::{session_name, set_value};
use pi_agent_core::harness::types::FileSystem;

mod jsonl_common;
use jsonl_common::WrappedEnv;

fn missing_error(path: &str) -> Option<std::io::ErrorKind> {
    std::fs::read(path).err().map(|error| error.kind())
}

fn file_text(path: &str) -> String {
    std::fs::read_to_string(path).expect("fixture read")
}

#[tokio::test]
async fn keeps_the_destination_unchanged_until_all_content_has_been_written() {
    let root = jsonl_common::TempRoot::new();
    let path = format!("{}/session.jsonl", root.path());
    std::fs::write(&path, "original").expect("fixture write");
    let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
    let staged = path.clone();
    let destination = path.clone();

    publish_file_atomically(
        &file_system,
        &path,
        &background_context(),
        move |append| async move {
            append("first\n".to_owned()).await?;
            assert_eq!(file_text(&format!("{staged}.tmp")), "first\n");
            assert_eq!(file_text(&destination), "original");
            append("second\n".to_owned()).await?;
            assert_eq!(file_text(&destination), "original");
            Ok(())
        },
    )
    .await
    .expect("publication");

    assert_eq!(file_text(&path), "first\nsecond\n");
    assert_eq!(
        missing_error(&format!("{path}.tmp")),
        Some(std::io::ErrorKind::NotFound)
    );
}

#[tokio::test]
async fn discards_partial_content_and_preserves_the_original_error_when_the_callback_fails() {
    let root = jsonl_common::TempRoot::new();
    let path = format!("{}/session.jsonl", root.path());
    std::fs::write(&path, "original").expect("fixture write");
    let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
    let failure = pi_agent_core::harness::session::jsonl::io::JsonlError(
        "content generation failed".to_owned(),
    );

    let published = publish_file_atomically(
        &file_system,
        &path,
        &background_context(),
        |append| async move {
            append("partial".to_owned()).await?;
            Err(pi_agent_core::harness::session::jsonl::io::JsonlError(
                "content generation failed".to_owned(),
            ))
        },
    )
    .await;

    assert_eq!(published.expect_err("callback failure"), failure);
    assert_eq!(file_text(&path), "original");
    assert_eq!(
        missing_error(&format!("{path}.tmp")),
        Some(std::io::ErrorKind::NotFound)
    );
}

#[tokio::test]
async fn preserves_the_destination_and_allows_retry_after_write_file_fails() {
    retry_after_method_failure("writeFile").await;
}

#[tokio::test]
async fn preserves_the_destination_and_allows_retry_after_append_file_fails() {
    retry_after_method_failure("appendFile").await;
}

#[tokio::test]
async fn preserves_the_destination_and_allows_retry_after_rename_file_fails() {
    retry_after_method_failure("renameFile").await;
}

/// The it.each body: one injected failure per filesystem method, upstream's
/// "preserves the destination and allows retry after %s fails".
async fn retry_after_method_failure(method: &'static str) {
    let root = jsonl_common::TempRoot::new();
    let path = format!("{}/session.jsonl", root.path());
    std::fs::write(&path, "original").expect("fixture write");
    let file_system = WrappedEnv::new(root.path().to_owned());
    file_system.fail_next(method);

    let published = publish_file_atomically(
        &{
            let env: Arc<dyn FileSystem> = file_system.clone();
            env
        },
        &path,
        &background_context(),
        |append| append("replacement".to_owned()),
    )
    .await;
    assert!(
        published
            .expect_err("injected failure")
            .0
            .contains("injected I/O failure")
    );
    assert_eq!(file_text(&path), "original");
    assert_eq!(
        missing_error(&format!("{path}.tmp")),
        Some(std::io::ErrorKind::NotFound)
    );

    let env: Arc<dyn FileSystem> = file_system;
    publish_file_atomically(&env, &path, &background_context(), |append| {
        append("retry".to_owned())
    })
    .await
    .expect("retry publication");
    assert_eq!(file_text(&path), "retry");
}

#[tokio::test]
async fn writes_the_header_first_and_preserves_transaction_boundaries() {
    let root = jsonl_common::TempRoot::new();
    let path = format!("{}/session.jsonl", root.path());
    let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
    let header = JsonlStorageHeader {
        v: 4,
        kind: "header".to_owned(),
        id: "session".to_owned(),
        storage_version: 1,
        created_at: 1_700_000_000_000,
        cwd: "/workspace".to_owned(),
        ..Default::default()
    };
    let write = |seq: u64, value: &str| -> CommittedWrite {
        let mut prepared = set_value(&session_name(), value.to_owned()).expect("value write");
        CommittedWrite::ValueSet {
            seq,
            namespace: std::mem::take(&mut prepared.namespace),
            key: std::mem::take(&mut prepared.key),
            value: std::mem::take(&mut prepared.value),
        }
    };
    let first = write(1, "first");
    let second = write(2, "second");
    let third = write(3, "third");
    let expected = format!(
        "{}\n{}\n[{},{}]\n",
        serde_json::to_string(&header).expect("header wire"),
        committed_write_wire(&first),
        committed_write_wire(&second),
        committed_write_wire(&third),
    );

    publish_jsonl(
        &file_system,
        &path,
        &header,
        &background_context(),
        |append| async move {
            append(&[first]).await?;
            append(&[second, third]).await
        },
    )
    .await
    .expect("publication");

    assert_eq!(file_text(&path), expected);
}

#[tokio::test]
async fn publishes_a_header_only_file_when_the_callback_emits_no_transactions() {
    let root = jsonl_common::TempRoot::new();
    let path = format!("{}/session.jsonl", root.path());
    let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
    let header = JsonlStorageHeader {
        v: 4,
        kind: "header".to_owned(),
        id: "session".to_owned(),
        storage_version: 1,
        created_at: 1_700_000_000_000,
        cwd: "/workspace".to_owned(),
        ..Default::default()
    };

    publish_jsonl(
        &file_system,
        &path,
        &header,
        &background_context(),
        |_| async { Ok(()) },
    )
    .await
    .expect("publication");

    assert_eq!(
        file_text(&path),
        format!("{}\n", serde_json::to_string(&header).expect("header wire"))
    );
}
