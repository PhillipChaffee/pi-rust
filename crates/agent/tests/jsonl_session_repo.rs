//! The `JsonlSessionRepo` lifecycle suite, ported 1:1 from upstream
//! `test/harness/jsonl-session-repo.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::jsonl::types::{
    JSONL_STORAGE_VERSION, JsonlSessionCreateOptions, JsonlSessionListOptions,
};
use pi_agent_core::harness::session::types::{ForkOptions, SessionError, SessionMutator};
use pi_agent_core::harness::session::values::{Write, session_name, set_value};
use pi_agent_core::harness::types::FileSystem;
use serde_json::json;

mod jsonl_common;
use jsonl_common::{NOW, WrappedEnv, jsonl_repo as build_repo};

/// The create options one id+cwd pair carries.
fn create_options(id: &str, cwd: &str) -> JsonlSessionCreateOptions {
    JsonlSessionCreateOptions {
        id: Some(id.to_owned()),
        parent_session_id: None,
        cwd: cwd.to_owned(),
    }
}

#[tokio::test]
async fn persists_metadata_and_filters_discovery_by_cwd() {
    let root = jsonl_common::TempRoot::new();
    let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
    let repo = build_repo(file_system.clone());
    let (session, metadata) = repo
        .create(
            JsonlSessionCreateOptions {
                id: Some("child".to_owned()),
                parent_session_id: Some("parent".to_owned()),
                cwd: "/workspace".to_owned(),
            },
            &background_context(),
        )
        .await
        .expect("create");

    assert_eq!(metadata.id, "child");
    assert_eq!(metadata.created_at, NOW);
    assert_eq!(metadata.storage_version, JSONL_STORAGE_VERSION);
    assert_eq!(metadata.cwd, "/workspace");
    assert_eq!(metadata.parent_session_id.as_deref(), Some("parent"));
    assert!(metadata.path.contains("/sessions/--workspace--/"));
    assert!(metadata.path.ends_with("_child.jsonl"));
    assert_ne!(metadata.modified_at, 0);
    session.close(&background_context()).await.expect("close");

    let other = repo
        .list(
            Some(JsonlSessionListOptions {
                cwd: Some("/other".to_owned()),
            }),
            &background_context(),
        )
        .await
        .expect("list other");
    assert_eq!(other, []);
    let workspace = repo
        .list(
            Some(JsonlSessionListOptions {
                cwd: Some("/workspace".to_owned()),
            }),
            &background_context(),
        )
        .await
        .expect("list workspace");
    assert_eq!(workspace.len(), 1);
    assert_eq!(&workspace[0], &metadata);
    let first_line: Vec<String> = file_system
        .read_text_lines(&metadata.path, None, &background_context())
        .await
        .expect("first line read");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(first_line.first().expect("line"))
            .expect("header wire"),
        json!({
            "v": 4,
            "kind": "header",
            "id": "child",
            "storageVersion": JSONL_STORAGE_VERSION,
            "createdAt": NOW,
            "cwd": "/workspace",
            "parentSessionId": "parent",
        }),
    );
    repo.close(&background_context()).await.expect("repo close");
}

#[tokio::test]
async fn atomically_publishes_a_branchless_session_header() {
    let root = jsonl_common::TempRoot::new();
    let file_system = WrappedEnv::new(root.path().to_owned());
    let repo = build_repo(file_system.clone());
    let (session, metadata) = repo
        .create(
            create_options("session", "/workspace"),
            &background_context(),
        )
        .await
        .expect("create");

    let publication = file_system
        .publication()
        .expect("atomic session publication");
    assert_eq!(publication.destination_path, metadata.path);
    assert!(!publication.destination_existed);
    let lines: Vec<&str> = publication.staged_content.trim_end().split('\n').collect();
    assert_eq!(lines.len(), 1);
    let header: serde_json::Value = serde_json::from_str(lines[0]).expect("header line");
    assert_eq!(header["kind"], json!("header"));
    assert_eq!(header["id"], json!("session"));
    let env: Arc<dyn FileSystem> = file_system.clone();
    assert_eq!(
        env.read_text_file(&metadata.path, &background_context())
            .await
            .expect("published file"),
        publication.staged_content,
    );
    assert!(
        !env.exists(&publication.source_path, &background_context())
            .await
            .expect("staged file gone"),
    );

    session.close(&background_context()).await.expect("close");
    repo.close(&background_context()).await.expect("repo close");
}

#[tokio::test]
async fn keeps_an_explicit_session_mutation_through_commit_until_end() {
    let root = jsonl_common::TempRoot::new();
    let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
    let repo = build_repo(file_system);
    let (session, metadata) = repo
        .create(
            create_options("session", "/workspace"),
            &background_context(),
        )
        .await
        .expect("create");
    let mutation = session
        .begin_mutation(&background_context())
        .await
        .expect("begin mutation");
    let queued_started = Arc::new(AtomicBool::new(false));

    // The queued mutation owns the session handle and hands it back after
    // the mutation line releases it.
    let queued = {
        let flag = queued_started.clone();
        tokio::spawn(async move {
            let result = session
                .mutate(
                    Box::new(
                        move |_mutator: &dyn SessionMutator,
                              _context|
                              -> pi_ai::types::BoxedFuture<
                            '_,
                            Result<Box<dyn std::any::Any + Send>, SessionError>,
                        > {
                            flag.store(true, Ordering::Release);
                            let done: Box<dyn std::any::Any + Send> = Box::new(());
                            Box::pin(std::future::ready(Ok(done)))
                        },
                    ),
                    &background_context(),
                )
                .await;
            (result, session)
        })
    };
    tokio::task::yield_now().await;

    let name_write = set_value(&session_name(), "explicit".to_owned()).expect("name write");
    let result = mutation
        .commit(vec![Write::ValueSet(name_write)], &background_context())
        .await
        .expect("explicit commit");
    assert_eq!(result.seqs.len(), 1);
    assert!(!queued_started.load(Ordering::Acquire));
    let stored = mutation
        .get_value(&session_name().address, &background_context())
        .await
        .expect("stored name")
        .expect("stored value");
    assert_eq!(stored.value, serde_json::json!("explicit"));
    mutation.end(&background_context()).await.expect("end");
    let (queued_result, session) = queued.await.expect("queued task");
    queued_result.expect("queued mutate");
    assert!(queued_started.load(Ordering::Acquire));

    session.close(&background_context()).await.expect("close");
    let (reopened, _) = repo
        .open(&metadata, &background_context())
        .await
        .expect("reopen");
    assert_eq!(
        reopened
            .get_name(&background_context())
            .await
            .expect("name"),
        Some("explicit".to_owned()),
    );
    reopened.close(&background_context()).await.expect("close");
    repo.close(&background_context()).await.expect("repo close");
}

#[tokio::test]
async fn rejects_unsupported_storage_versions_without_repairing_a_torn_tail() {
    let root = jsonl_common::TempRoot::new();
    let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
    let repo = build_repo(file_system.clone());
    let (session, metadata) = repo
        .create(
            create_options("future", "/workspace"),
            &background_context(),
        )
        .await
        .expect("create");
    session.close(&background_context()).await.expect("close");

    let lines: Vec<String> = file_system
        .read_text_file(&metadata.path, &background_context())
        .await
        .expect("file read")
        .trim_end()
        .split('\n')
        .map(str::to_owned)
        .collect();
    let unsupported_version = JSONL_STORAGE_VERSION + 1;
    let mut header: serde_json::Value =
        serde_json::from_str(lines.first().expect("header line")).expect("header parse");
    header["storageVersion"] = json!(unsupported_version);
    let unsupported_content = format!("{header}\n{{\"kind\":\"entry\"");
    std::fs::write(&metadata.path, &unsupported_content).expect("file write");

    let opened = repo.open(&metadata, &background_context()).await;
    assert!(
        opened
            .err()
            .expect("open rejected")
            .to_string()
            .contains(&format!(
                "unsupported storage version {unsupported_version}"
            ))
    );
    assert_eq!(
        std::fs::read_to_string(&metadata.path).expect("file read"),
        unsupported_content,
    );

    repo.close(&background_context()).await.expect("repo close");
}

#[tokio::test]
async fn keeps_fork_destinations_claimed_until_close_and_rejects_deleting_open_sessions() {
    let root = jsonl_common::TempRoot::new();
    let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
    let repo = build_repo(file_system);
    let (source, source_metadata) = repo
        .create(
            create_options("source", "/workspace"),
            &background_context(),
        )
        .await
        .expect("source create");
    let (fork, fork_metadata) = repo
        .fork(
            &source_metadata,
            &ForkOptions::Tree {
                id: Some("fork".to_owned()),
            },
            &background_context(),
        )
        .await
        .expect("fork");

    let opened = repo.open(&fork_metadata, &background_context()).await;
    assert!(
        opened
            .err()
            .expect("open rejected")
            .to_string()
            .contains("already open")
    );
    let deleted = repo.delete(&fork_metadata, &background_context()).await;
    assert!(
        deleted
            .expect_err("delete rejected")
            .to_string()
            .contains("open")
    );
    fork.close(&background_context()).await.expect("fork close");

    let (reopened, _) = repo
        .open(&fork_metadata, &background_context())
        .await
        .expect("reopen");
    reopened.close(&background_context()).await.expect("close");
    repo.delete(&fork_metadata, &background_context())
        .await
        .expect("delete");
    let opened = repo.open(&fork_metadata, &background_context()).await;
    assert!(
        opened
            .err()
            .expect("open rejected")
            .to_string()
            .contains("does not exist")
    );

    source
        .close(&background_context())
        .await
        .expect("source close");
    repo.close(&background_context()).await.expect("repo close");
}

#[tokio::test]
async fn rejects_concurrent_creates_for_the_same_working_directory_id() {
    let root = jsonl_common::TempRoot::new();
    let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
    let repo = Arc::new(build_repo(file_system));
    let options = || create_options("session", "/workspace");

    let ctx = background_context();
    let (first, second) = tokio::join!(repo.create(options(), &ctx), repo.create(options(), &ctx),);
    let results = vec![first, second];
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(results.iter().filter(|result| result.is_err()).count(), 1);
    let listed = repo
        .list(
            Some(JsonlSessionListOptions {
                cwd: Some("/workspace".to_owned()),
            }),
            &background_context(),
        )
        .await
        .expect("list");
    assert_eq!(listed.len(), 1);
    for session in results.into_iter().flatten() {
        session.0.close(&background_context()).await.expect("close");
    }
    repo.close(&background_context()).await.expect("repo close");
}

#[tokio::test]
async fn allows_the_same_id_to_be_active_in_different_working_directories() {
    let root = jsonl_common::TempRoot::new();
    let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
    let repo = build_repo(file_system);
    let (first, first_metadata) = repo
        .create(
            create_options("shared", "/workspace-a"),
            &background_context(),
        )
        .await
        .expect("first create");
    let (second, second_metadata) = repo
        .create(
            create_options("shared", "/workspace-b"),
            &background_context(),
        )
        .await
        .expect("second create");

    assert_ne!(first_metadata.path, second_metadata.path);
    let again = repo
        .create(
            create_options("shared", "/workspace-a"),
            &background_context(),
        )
        .await;
    assert!(
        again
            .err()
            .expect("create rejected")
            .to_string()
            .contains("already exists")
    );
    let listed = repo.list(None, &background_context()).await.expect("list");
    assert_eq!(
        listed
            .iter()
            .map(|metadata| (metadata.cwd.clone(), metadata.id.clone()))
            .collect::<Vec<_>>(),
        [
            ("/workspace-a".to_owned(), "shared".to_owned()),
            ("/workspace-b".to_owned(), "shared".to_owned()),
        ],
    );

    first
        .close(&background_context())
        .await
        .expect("first close");
    second
        .close(&background_context())
        .await
        .expect("second close");
    let (first_reopened, _) = repo
        .open(&first_metadata, &background_context())
        .await
        .expect("first reopen");
    let (second_reopened, _) = repo
        .open(&second_metadata, &background_context())
        .await
        .expect("second reopen");
    first_reopened
        .close(&background_context())
        .await
        .expect("close");
    second_reopened
        .close(&background_context())
        .await
        .expect("close");
    repo.close(&background_context()).await.expect("repo close");
}
