//! The footer provider's branch/watcher characterization, upstream's
//! `test/footer-data-provider.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The git probe is injected (upstream's `spawnSync`/`execFile` mocks) and
//! the timers run in manual mode (upstream's fake timers).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use pi_coding_agent::footer_data_provider::EnvLookup;
use pi_coding_agent::footer_data_provider::FooterDataProvider;
use pi_coding_agent::footer_data_provider::FooterDataProviderOptions;
use pi_coding_agent::footer_data_provider::GitBranchRunner;
use pi_coding_agent::footer_data_provider::should_poll_git_head;
use tempfile::TempDir;

fn write(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("mkdir");
    }
    std::fs::write(path, content).expect("write");
}

fn create_plain_repo(temp: &TempDir) -> String {
    let repo = temp.path().join("repo");
    write(&repo.join(".git").join("HEAD"), "ref: refs/heads/main\n");
    repo.display().to_string()
}

fn create_plain_reftable_repo(temp: &TempDir) -> String {
    let repo = temp.path().join("repo");
    write(
        &repo.join(".git").join("HEAD"),
        "ref: refs/heads/.invalid\n",
    );
    std::fs::create_dir_all(repo.join(".git").join("reftable")).expect("reftable");
    repo.display().to_string()
}

fn create_reftable_worktree(temp: &TempDir) -> (String, String) {
    let repo = temp.path().join("repo");
    let common_git_dir = repo.join(".git");
    let git_dir = common_git_dir.join("worktrees").join("src");
    let worktree = temp.path().join("worktree");
    let reftable = common_git_dir.join("reftable");
    write(&git_dir.join("HEAD"), "ref: refs/heads/.invalid\n");
    write(&git_dir.join("commondir"), "../..\n");
    write(&reftable.join("tables.list"), "0\n");
    std::fs::create_dir_all(&worktree).expect("worktree");
    write(
        &worktree.join(".git"),
        &format!("gitdir: {}\n", git_dir.display()),
    );
    (
        worktree.display().to_string(),
        reftable.display().to_string(),
    )
}

fn counter_runner(branch: &str) -> (GitBranchRunner, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let branch = branch.to_owned();
    let runner: GitBranchRunner = {
        let calls = Arc::clone(&calls);
        Arc::new(move |_repo: &str| {
            calls.fetch_add(1, Ordering::SeqCst);
            Some(branch.clone())
        })
    };
    (runner, calls)
}

fn manual_options(runner: GitBranchRunner) -> FooterDataProviderOptions {
    FooterDataProviderOptions {
        manual_timers: true,
        resolve_branch: Some(runner),
        env: None,
    }
}

#[test]
fn uses_head_directly_in_a_regular_repo_from_a_nested_directory() {
    let temp = TempDir::new().expect("tempdir");
    let repo = create_plain_repo(&temp);
    let nested = Path::new(&repo).join("src").join("nested");
    std::fs::create_dir_all(&nested).expect("nested");

    let (runner, calls) = counter_runner("ignored");
    let provider =
        FooterDataProvider::with_options(&nested.display().to_string(), manual_options(runner));
    assert_eq!(provider.get_git_branch().as_deref(), Some("main"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    provider.dispose();
}

#[test]
fn resolves_the_branch_via_git_when_head_is_invalid() {
    let temp = TempDir::new().expect("tempdir");
    let repo = create_plain_reftable_repo(&temp);
    let (runner, calls) = counter_runner("main");
    let provider = FooterDataProvider::with_options(&repo, manual_options(runner));
    assert_eq!(provider.get_git_branch().as_deref(), Some("main"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    provider.dispose();
}

#[test]
fn resolves_the_branch_via_git_in_a_reftable_worktree() {
    let temp = TempDir::new().expect("tempdir");
    let (worktree, _) = create_reftable_worktree(&temp);
    let (runner, _) = counter_runner("main");
    let provider = FooterDataProvider::with_options(&worktree, manual_options(runner));
    assert_eq!(provider.get_git_branch().as_deref(), Some("main"));
    provider.dispose();
}

#[test]
fn treats_an_unresolved_invalid_head_as_detached() {
    let temp = TempDir::new().expect("tempdir");
    let repo = create_plain_reftable_repo(&temp);
    let runner: GitBranchRunner = Arc::new(|_repo: &str| None);
    let provider = FooterDataProvider::with_options(&repo, manual_options(runner));
    assert_eq!(provider.get_git_branch().as_deref(), Some("detached"));
    provider.dispose();
}

#[test]
fn does_not_notify_listeners_when_updates_keep_the_same_branch() {
    let temp = TempDir::new().expect("tempdir");
    let (worktree, _) = create_reftable_worktree(&temp);
    let (runner, calls) = counter_runner("main");
    let provider = FooterDataProvider::with_options(&worktree, manual_options(runner));
    assert_eq!(provider.get_git_branch().as_deref(), Some("main"));
    calls.store(0, Ordering::SeqCst);
    let notified = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&notified);
    let _unsubscribe = provider.on_branch_change(Arc::new(move || {
        counter.fetch_add(1, Ordering::SeqCst);
    }));

    provider.request_refresh();
    provider.run_pending_refresh();

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.get_git_branch().as_deref(), Some("main"));
    assert_eq!(notified.load(Ordering::SeqCst), 0);
    provider.dispose();
}

#[test]
fn debounces_rapid_updates_into_a_single_refresh() {
    let temp = TempDir::new().expect("tempdir");
    let (worktree, _) = create_reftable_worktree(&temp);
    let (runner, calls) = counter_runner("main");
    let provider = FooterDataProvider::with_options(&worktree, manual_options(runner));
    assert_eq!(provider.get_git_branch().as_deref(), Some("main"));
    calls.store(0, Ordering::SeqCst);

    provider.request_refresh();
    provider.request_refresh();
    provider.request_refresh();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    provider.run_pending_refresh();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    provider.run_pending_refresh();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    provider.dispose();
}

#[test]
fn updates_the_cached_branch_when_the_reftable_directory_changes() {
    let temp = TempDir::new().expect("tempdir");
    let (worktree, reftable) = create_reftable_worktree(&temp);
    let calls = Arc::new(AtomicUsize::new(0));
    let runner: GitBranchRunner = {
        let calls = Arc::clone(&calls);
        Arc::new(move |_repo: &str| {
            let call = calls.fetch_add(1, Ordering::SeqCst);
            Some(if call == 0 { "main" } else { "foo" }.to_owned())
        })
    };
    let provider = FooterDataProvider::with_options(&worktree, manual_options(runner));
    assert_eq!(provider.get_git_branch().as_deref(), Some("main"));
    let notified = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&notified);
    let _unsubscribe = provider.on_branch_change(Arc::new(move || {
        counter.fetch_add(1, Ordering::SeqCst);
    }));

    write(&Path::new(&reftable).join("tables.list"), "1\n");
    provider.request_refresh();
    provider.run_pending_refresh();

    assert_eq!(provider.get_git_branch().as_deref(), Some("foo"));
    assert_eq!(notified.load(Ordering::SeqCst), 1);
    provider.dispose();
}

#[test]
fn retries_the_git_watcher_after_an_error() {
    let temp = TempDir::new().expect("tempdir");
    let repo = create_plain_repo(&temp);
    let (runner, _) = counter_runner("main");
    let provider = FooterDataProvider::with_options(&repo, manual_options(runner));
    assert!(provider.has_head_watcher());

    provider.simulate_watcher_error();
    assert!(!provider.has_head_watcher());
    provider.run_pending_watcher_retry();
    assert!(provider.has_head_watcher());
    provider.dispose();
}

#[test]
fn tracks_extension_statuses_and_provider_counts() {
    let temp = TempDir::new().expect("tempdir");
    let (runner, _) = counter_runner("main");
    let provider = FooterDataProvider::with_options(
        &temp.path().display().to_string(),
        manual_options(runner),
    );
    provider.set_extension_status("a", Some("first"));
    provider.set_extension_status("b", Some("second"));
    provider.set_extension_status("a", Some("updated"));
    assert_eq!(
        provider.get_extension_statuses(),
        vec![
            ("a".to_owned(), "updated".to_owned()),
            ("b".to_owned(), "second".to_owned())
        ]
    );
    provider.set_extension_status("a", None);
    assert_eq!(provider.get_extension_statuses().len(), 1);
    provider.clear_extension_statuses();
    assert!(provider.get_extension_statuses().is_empty());
    provider.set_available_provider_count(3);
    assert_eq!(provider.get_available_provider_count(), 3);
    provider.dispose();
}

#[test]
fn set_cwd_moves_the_repo_and_notifies() {
    let temp = TempDir::new().expect("tempdir");
    let first = create_plain_repo(&temp);
    let second = temp.path().join("second");
    write(&second.join(".git").join("HEAD"), "ref: refs/heads/other\n");

    let (runner, _) = counter_runner("main");
    let provider = FooterDataProvider::with_options(&first, manual_options(runner));
    let notified = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&notified);
    let _unsubscribe = provider.on_branch_change(Arc::new(move || {
        counter.fetch_add(1, Ordering::SeqCst);
    }));
    provider.set_cwd(&second.display().to_string());
    assert_eq!(provider.get_git_branch().as_deref(), Some("other"));
    assert_eq!(notified.load(Ordering::SeqCst), 1);
    provider.set_cwd(&second.display().to_string());
    assert_eq!(notified.load(Ordering::SeqCst), 1);
    provider.dispose();
}

#[test]
fn outside_a_repo_the_branch_is_absent() {
    let temp = TempDir::new().expect("tempdir");
    let (runner, _) = counter_runner("main");
    let provider = FooterDataProvider::with_options(
        &temp.path().display().to_string(),
        manual_options(runner),
    );
    assert_eq!(provider.get_git_branch(), None);
    assert!(!provider.has_head_watcher());
    provider.dispose();
}

#[test]
fn wsl_mounted_paths_are_the_only_polled_ones() {
    let wsl: EnvLookup =
        Arc::new(|key: &str| (key == "WSL_DISTRO_NAME").then(|| "Ubuntu".to_owned()));
    let none: EnvLookup = Arc::new(|_key: &str| None);
    assert_eq!(
        should_poll_git_head("/mnt/c/repo", &wsl),
        cfg!(target_os = "linux")
    );
    assert!(!should_poll_git_head("/home/me/repo", &wsl));
    assert!(!should_poll_git_head("/mnt/c/repo", &none));
    assert!(!should_poll_git_head("/mnt/", &wsl));
}

#[test]
fn unsubscribe_stops_branch_notifications() {
    let temp = TempDir::new().expect("tempdir");
    let repo = create_plain_repo(&temp);
    let (runner, _) = counter_runner("main");
    let provider = FooterDataProvider::with_options(&repo, manual_options(runner));
    let notified = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&notified);
    let unsubscribe = provider.on_branch_change(Arc::new(move || {
        counter.fetch_add(1, Ordering::SeqCst);
    }));
    unsubscribe();
    provider.set_cwd(&temp.path().join("missing").display().to_string());
    assert_eq!(notified.load(Ordering::SeqCst), 0);
    provider.dispose();
}
