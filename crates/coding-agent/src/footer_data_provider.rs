//! The git metadata probe, upstream's `src/core/footer-data-provider.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! [`find_git_paths`] is the walk-up git-metadata reader the resource
//! loader's worktree dedup consumes; [`FooterDataProvider`] adds the branch
//! cache, branch-change watch, extension statuses, and provider count the
//! footer renders.
//!
//! Restatements: node's `execFile`/`spawnSync` git probe is one injectable
//! blocking runner called from the refresh thread; `fs.watch` is the
//! `notify` crate's directory watch (HEAD's directory, because git renames
//! HEAD atomically and a file watch would follow the old inode); the WSL
//! polling fallback keeps node's `watchFile` contract; timers run on worker
//! threads and tests drive them through the manual mode.

use std::path::Path;

/// The git metadata paths one repo walk found, upstream's `GitPaths`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GitPaths {
    /// The working tree root carrying the `.git` entry.
    pub repo_dir: String,
    /// The shared git directory — the worktree git dir's `commondir`
    /// target when linked, else the `.git` directory itself.
    pub common_git_dir: String,
    /// The git dir's `HEAD` file.
    pub head_path: String,
}

/// Find git metadata paths by walking up from the cwd, upstream's
/// `findGitPaths`.
///
/// Handles both regular repos (`.git` is a directory) and linked
/// worktrees (`.git` is a `gitdir:` file). `None` when no repo root sits
/// above the cwd, when a `gitdir:` target or its `HEAD` is missing, or
/// when the stat of a `.git` entry fails.
#[must_use]
pub fn find_git_paths(cwd: &str) -> Option<GitPaths> {
    let mut dir = cwd.to_string();
    loop {
        let git_path = Path::new(&dir).join(".git");
        if git_path.exists() {
            let Ok(fs_meta) = std::fs::metadata(&git_path) else {
                return None;
            };
            if fs_meta.is_file() {
                let Ok(content) = std::fs::read_to_string(&git_path) else {
                    return None;
                };
                let content = content.trim();
                if let Some(git_dir_raw) = content.strip_prefix("gitdir: ") {
                    let git_dir = resolve_against(&dir, git_dir_raw.trim());
                    let head_path = Path::new(&git_dir).join("HEAD");
                    if !head_path.exists() {
                        return None;
                    }
                    let common_dir_path = Path::new(&git_dir).join("commondir");
                    let common_git_dir = if common_dir_path.exists() {
                        let Ok(common_dir) = std::fs::read_to_string(&common_dir_path) else {
                            return None;
                        };
                        resolve_against(&git_dir, common_dir.trim())
                    } else {
                        git_dir
                    };
                    return Some(GitPaths {
                        repo_dir: dir,
                        common_git_dir,
                        head_path: head_path.to_string_lossy().into_owned(),
                    });
                }
            } else if fs_meta.is_dir() {
                let head_path = git_path.join("HEAD");
                if !head_path.exists() {
                    return None;
                }
                return Some(GitPaths {
                    repo_dir: dir,
                    common_git_dir: git_path.to_string_lossy().into_owned(),
                    head_path: head_path.to_string_lossy().into_owned(),
                });
            }
        }
        let parent = parent_dir(&dir)?;
        dir = parent;
    }
}

/// Resolve `target` against `base`, Node's `path.resolve` for the
/// absolute-or-relative targets git writes: the join normalizes lexically
/// (`..` pops, `.` drops), the way Node's resolver collapses the
/// `commondir` relative paths git writes.
fn resolve_against(base: &str, target: &str) -> String {
    if target.starts_with('/') {
        return normalize_segments(target);
    }
    let joined = if base.ends_with('/') {
        format!("{base}{target}")
    } else {
        format!("{base}/{target}")
    };
    normalize_segments(&joined)
}

/// Collapse `.` and `..` segments lexically, Node's `path.normalize`'
/// segment walk.
fn normalize_segments(joined: &str) -> String {
    let mut segments: Vec<&str> = Vec::new();
    for segment in joined.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    let mut normalized = segments.join("/");
    if joined.starts_with('/') {
        normalized = format!("/{normalized}");
    }
    if normalized.is_empty() {
        normalized = "/".to_string();
    }
    normalized
}

/// The parent of a POSIX path, `None` at the root.
fn parent_dir(dir: &str) -> Option<String> {
    let trimmed = dir.trim_end_matches('/');
    let parent = match trimmed.rfind('/') {
        Some(0) => "/",
        Some(at) => &trimmed[..at],
        None => return None,
    };
    if parent == trimmed {
        return None;
    }
    Some(parent.to_string())
}

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::SystemTime;

use crate::utils::fs_watch::FS_WATCH_RETRY_DELAY_MS;
use crate::utils::fs_watch::Watcher;
use crate::utils::fs_watch::close_watcher;
use crate::utils::fs_watch::watch_with_error_handler;
use crate::utils::paths::basename_posix;
use crate::utils::paths::dirname_posix;

/// How long a burst of watch events settles before the refresh runs,
/// upstream's `WATCH_DEBOUNCE_MS`.
const WATCH_DEBOUNCE_MS: Duration = Duration::from_millis(500);

/// The branch resolver, node's `git --no-optional-locks symbolic-ref
/// --quiet --short HEAD`: `Some(branch)` when git answers, `None` on a
/// detached HEAD or a missing git binary.
pub type GitBranchRunner = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// The environment lookup the WSL probe reads.
pub type EnvLookup = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// The footer provider's construction options.
#[derive(Clone, Default)]
pub struct FooterDataProviderOptions {
    /// Drive the debounce and retry timers manually, for deterministic tests.
    pub manual_timers: bool,
    /// The git branch resolver. Default: the real `git` process.
    pub resolve_branch: Option<GitBranchRunner>,
    /// The environment lookup. Default: the process environment.
    pub env: Option<EnvLookup>,
}

impl std::fmt::Debug for FooterDataProviderOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FooterDataProviderOptions")
            .field("manual_timers", &self.manual_timers)
            .finish_non_exhaustive()
    }
}

/// The live watchers the provider re-arms after a failure.
#[derive(Default)]
struct GitWatchers {
    head: Option<Watcher>,
    reftable: Option<Watcher>,
    poll_stop: Option<Arc<AtomicBool>>,
}

/// The three-state branch cache, upstream's `string | null | undefined`.
#[derive(Clone, Default, PartialEq, Eq)]
enum CachedBranch {
    /// Never resolved.
    #[default]
    Unknown,
    /// Resolved, possibly to "no branch".
    Resolved(Option<String>),
}

/// The provider's shared state.
#[expect(
    clippy::struct_excessive_bools,
    reason = "the three timer flags and the disposed latch mirror upstream's own fields"
)]
struct FooterInner {
    cwd: String,
    options: FooterDataProviderOptions,
    branch_runner: GitBranchRunner,
    env: EnvLookup,
    extension_statuses: Vec<(String, String)>,
    cached_branch: CachedBranch,
    git_paths: Option<GitPaths>,
    watchers: GitWatchers,
    branch_callbacks: Vec<(u64, Arc<dyn Fn() + Send + Sync>)>,
    next_callback_id: u64,
    available_provider_count: usize,
    refresh_pending: bool,
    refresh_in_flight: bool,
    retry_pending: bool,
    disposed: bool,
}

/// Provides git branch and extension statuses, upstream's
/// `FooterDataProvider`.
pub struct FooterDataProvider {
    inner: Arc<Mutex<FooterInner>>,
}

impl std::fmt::Debug for FooterDataProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FooterDataProvider")
            .field("cwd", &self.cwd())
            .finish_non_exhaustive()
    }
}

fn lock(inner: &Arc<Mutex<FooterInner>>) -> MutexGuard<'_, FooterInner> {
    inner.lock().unwrap_or_else(PoisonError::into_inner)
}

impl FooterDataProvider {
    /// Build the provider over `cwd`, starting its git watch.
    #[must_use]
    pub fn new(cwd: &str) -> Self {
        Self::with_options(cwd, FooterDataProviderOptions::default())
    }

    /// Build the provider over `cwd` with `options`.
    #[must_use]
    pub fn with_options(cwd: &str, options: FooterDataProviderOptions) -> Self {
        let branch_runner = options.resolve_branch.clone().unwrap_or_else(|| {
            Arc::new(|repo_dir: &str| {
                let output = std::process::Command::new("git")
                    .args([
                        "--no-optional-locks",
                        "symbolic-ref",
                        "--quiet",
                        "--short",
                        "HEAD",
                    ])
                    .current_dir(repo_dir)
                    .stderr(std::process::Stdio::null())
                    .output()
                    .ok()?;
                if !output.status.success() {
                    return None;
                }
                let branch = String::from_utf8_lossy(&output.stdout).trim().to_owned();
                (!branch.is_empty()).then_some(branch)
            })
        });
        let env = options
            .env
            .clone()
            .unwrap_or_else(|| Arc::new(|key: &str| std::env::var(key).ok()));
        let inner = Arc::new(Mutex::new(FooterInner {
            cwd: cwd.to_owned(),
            options,
            branch_runner,
            env,
            extension_statuses: Vec::new(),
            cached_branch: CachedBranch::Unknown,
            git_paths: find_git_paths(cwd),
            watchers: GitWatchers::default(),
            branch_callbacks: Vec::new(),
            next_callback_id: 1,
            available_provider_count: 0,
            refresh_pending: false,
            refresh_in_flight: false,
            retry_pending: false,
            disposed: false,
        }));
        setup_git_watcher(&inner);
        Self { inner }
    }

    /// The provider's cwd.
    #[must_use]
    pub fn cwd(&self) -> String {
        lock(&self.inner).cwd.clone()
    }

    /// The current git branch, `None` outside a repo, `"detached"` on a
    /// detached HEAD.
    #[must_use]
    pub fn get_git_branch(&self) -> Option<String> {
        {
            let state = lock(&self.inner);
            if let CachedBranch::Resolved(cached) = &state.cached_branch {
                return cached.clone();
            }
        }
        let branch = resolve_git_branch(&self.inner);
        lock(&self.inner).cached_branch = CachedBranch::Resolved(branch.clone());
        branch
    }

    /// The extension status texts, in insertion order.
    #[must_use]
    pub fn get_extension_statuses(&self) -> Vec<(String, String)> {
        lock(&self.inner).extension_statuses.clone()
    }

    /// Subscribe to git branch changes; the returned closure unsubscribes.
    pub fn on_branch_change(
        &self,
        callback: Arc<dyn Fn() + Send + Sync>,
    ) -> Box<dyn Fn() + Send + Sync> {
        let id = {
            let mut state = lock(&self.inner);
            let id = state.next_callback_id;
            state.next_callback_id += 1;
            state.branch_callbacks.push((id, callback));
            id
        };
        let inner = Arc::clone(&self.inner);
        Box::new(move || {
            lock(&inner)
                .branch_callbacks
                .retain(|(entry, _)| *entry != id);
        })
    }

    /// Set one extension status, `None` clearing it.
    pub fn set_extension_status(&self, key: &str, text: Option<&str>) {
        let mut state = lock(&self.inner);
        match text {
            Some(text) => {
                if let Some(entry) = state
                    .extension_statuses
                    .iter_mut()
                    .find(|(name, _)| name == key)
                {
                    text.clone_into(&mut entry.1);
                } else {
                    state
                        .extension_statuses
                        .push((key.to_owned(), text.to_owned()));
                }
            }
            None => state.extension_statuses.retain(|(name, _)| name != key),
        }
    }

    /// Clear every extension status.
    pub fn clear_extension_statuses(&self) {
        lock(&self.inner).extension_statuses.clear();
    }

    /// The number of providers with available models.
    #[must_use]
    pub fn get_available_provider_count(&self) -> usize {
        lock(&self.inner).available_provider_count
    }

    /// Set the available-provider count.
    pub fn set_available_provider_count(&self, count: usize) {
        lock(&self.inner).available_provider_count = count;
    }
}

impl FooterDataProvider {
    /// Move to `cwd`, re-reading the repo and re-arming the watch.
    pub fn set_cwd(&self, cwd: &str) {
        {
            let mut state = lock(&self.inner);
            if state.cwd == cwd {
                return;
            }
            cwd.clone_into(&mut state.cwd);
            state.refresh_pending = false;
            state.cached_branch = CachedBranch::Unknown;
            state.git_paths = find_git_paths(cwd);
            clear_git_watchers(&mut state);
            drop(state);
        }
        setup_git_watcher(&self.inner);
        notify_branch_change(&self.inner);
    }

    /// Stop the timers and watchers and drop every branch listener.
    pub fn dispose(&self) {
        let mut state = lock(&self.inner);
        state.disposed = true;
        state.refresh_pending = false;
        state.retry_pending = false;
        clear_git_watchers(&mut state);
        state.branch_callbacks.clear();
    }

    /// Run the pending debounced refresh inline, the manual-timer stand-in
    /// for upstream's fake-timer advance.
    pub fn run_pending_refresh(&self) {
        let pending = {
            let mut state = lock(&self.inner);
            if !state.refresh_pending {
                return;
            }
            state.refresh_pending = false;
            true
        };
        if pending {
            refresh_git_branch(&self.inner);
        }
    }

    /// Run the pending watcher-retry inline, the manual-timer stand-in.
    pub fn run_pending_watcher_retry(&self) {
        let pending = {
            let mut state = lock(&self.inner);
            if !state.retry_pending {
                return;
            }
            state.retry_pending = false;
            true
        };
        if pending {
            setup_git_watcher(&self.inner);
        }
    }

    /// Whether a head watcher is armed; the boundary tests' read.
    #[doc(hidden)]
    #[must_use]
    pub fn has_head_watcher(&self) -> bool {
        lock(&self.inner).watchers.head.is_some()
    }

    /// Request a debounced refresh, the boundary tests' stand-in for an
    /// fs-watch event.
    #[doc(hidden)]
    pub fn request_refresh(&self) {
        schedule_refresh(&self.inner);
    }

    /// Drive a watcher failure, the boundary tests' stand-in for notify's
    /// error channel.
    #[doc(hidden)]
    pub fn simulate_watcher_error(&self) {
        handle_git_watcher_error(&self.inner);
    }
}

/// Read the branch from HEAD, `None` on a read failure, upstream's
/// `resolveGitBranchSync`/`resolveGitBranchAsync` collapsed.
fn resolve_git_branch(inner: &Arc<Mutex<FooterInner>>) -> Option<String> {
    let (paths, runner) = {
        let state = lock(inner);
        (state.git_paths.clone(), Arc::clone(&state.branch_runner))
    };
    let paths = paths?;
    let content = std::fs::read_to_string(&paths.head_path).ok()?;
    let content = content.trim();
    let Some(branch) = content.strip_prefix("ref: refs/heads/") else {
        return Some("detached".to_owned());
    };
    if branch == ".invalid" {
        Some(runner(&paths.repo_dir).unwrap_or_else(|| "detached".to_owned()))
    } else {
        Some(branch.to_owned())
    }
}

/// Call every branch listener.
fn notify_branch_change(inner: &Arc<Mutex<FooterInner>>) {
    let callbacks: Vec<Arc<dyn Fn() + Send + Sync>> = lock(inner)
        .branch_callbacks
        .iter()
        .map(|(_, callback)| Arc::clone(callback))
        .collect();
    for callback in callbacks {
        callback();
    }
}

/// Schedule a debounced branch refresh, upstream's `scheduleRefresh`.
fn schedule_refresh(inner: &Arc<Mutex<FooterInner>>) {
    let manual = {
        let mut state = lock(inner);
        if state.disposed {
            return;
        }
        if state.refresh_in_flight {
            state.refresh_pending = true;
            return;
        }
        if state.refresh_pending {
            return;
        }
        state.refresh_pending = true;
        state.options.manual_timers
    };
    if manual {
        return;
    }
    let inner = Arc::clone(inner);
    std::thread::spawn(move || {
        std::thread::sleep(WATCH_DEBOUNCE_MS);
        let pending = {
            let mut state = lock(&inner);
            if state.disposed || !state.refresh_pending {
                false
            } else {
                state.refresh_pending = false;
                true
            }
        };
        if pending {
            refresh_git_branch(&inner);
        }
    });
}

/// Re-resolve the branch and notify when it changed, upstream's
/// `refreshGitBranchAsync`.
fn refresh_git_branch(inner: &Arc<Mutex<FooterInner>>) {
    {
        let mut state = lock(inner);
        if state.disposed {
            return;
        }
        if state.refresh_in_flight {
            state.refresh_pending = true;
            return;
        }
        state.refresh_in_flight = true;
    }
    let next = resolve_git_branch(inner);
    let (changed, again) = {
        let mut state = lock(inner);
        let changed =
            matches!(&state.cached_branch, CachedBranch::Resolved(previous) if previous != &next);
        state.cached_branch = CachedBranch::Resolved(next);
        state.refresh_in_flight = false;
        let again = state.refresh_pending && !state.disposed;
        if again {
            state.refresh_pending = false;
        }
        drop(state);
        (changed, again)
    };
    if changed {
        notify_branch_change(inner);
    }
    if again {
        schedule_refresh(inner);
    }
}

/// Schedule the 5-second watcher re-arm, upstream's
/// `scheduleGitWatcherRetry`.
fn schedule_git_watcher_retry(inner: &Arc<Mutex<FooterInner>>) {
    let manual = {
        let mut state = lock(inner);
        if state.disposed || state.retry_pending {
            return;
        }
        state.retry_pending = true;
        state.options.manual_timers
    };
    if manual {
        return;
    }
    let inner = Arc::clone(inner);
    std::thread::spawn(move || {
        std::thread::sleep(FS_WATCH_RETRY_DELAY_MS);
        let pending = {
            let mut state = lock(&inner);
            if state.disposed || !state.retry_pending {
                false
            } else {
                state.retry_pending = false;
                true
            }
        };
        if pending {
            setup_git_watcher(&inner);
        }
    });
}

/// A watcher failure: close the watches and re-arm after the retry delay,
/// upstream's `handleGitWatcherError`.
fn handle_git_watcher_error(inner: &Arc<Mutex<FooterInner>>) {
    clear_git_watchers(&mut lock(inner));
    schedule_git_watcher_retry(inner);
}

/// Stop the polling thread and drop both watches.
fn clear_git_watchers(state: &mut FooterInner) {
    if let Some(stop) = state.watchers.poll_stop.take() {
        stop.store(true, Ordering::Relaxed);
    }
    close_watcher(state.watchers.head.take());
    close_watcher(state.watchers.reftable.take());
    state.retry_pending = false;
}

/// Arm the HEAD-directory watch, the reftable watch, and the WSL polling
/// fallback, upstream's `setupGitWatcher`.
fn setup_git_watcher(inner: &Arc<Mutex<FooterInner>>) {
    {
        let mut state = lock(inner);
        clear_git_watchers(&mut state);
        if state.disposed {
            return;
        }
    }
    let (paths, poll_env) = {
        let state = lock(inner);
        (state.git_paths.clone(), Arc::clone(&state.env))
    };
    let Some(paths) = paths else {
        return;
    };
    let poll = should_poll_git_head(&paths.repo_dir, &poll_env);

    let head_dir = dirname_posix(&paths.head_path);
    let head_inner = Arc::clone(inner);
    let error_inner = Arc::clone(inner);
    let head = watch_with_error_handler(
        &head_dir,
        move |event| {
            let is_head = event
                .filename
                .as_deref()
                .is_none_or(|name| basename_posix(name) == "HEAD");
            if is_head {
                schedule_refresh(&head_inner);
            }
        },
        move || handle_git_watcher_error(&error_inner),
    );
    lock(inner).watchers.head = head;

    if poll {
        let head_path = paths.head_path.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let poll_inner = Arc::clone(inner);
        let poll_stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut previous = file_signature(&head_path);
            while !poll_stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_secs(1));
                let current = file_signature(&head_path);
                if current != previous {
                    previous = current;
                    schedule_refresh(&poll_inner);
                }
            }
        });
        lock(inner).watchers.poll_stop = Some(stop);
    }
    if lock(inner).watchers.head.is_none() && !poll {
        return;
    }

    // Reftable repos switch branches by rewriting files under `reftable/`,
    // not HEAD, so that directory gets its own watch.
    let reftable_dir = format!("{}/reftable", paths.common_git_dir);
    if Path::new(&reftable_dir).exists() {
        let reftable_inner = Arc::clone(inner);
        let error_inner = Arc::clone(inner);
        let reftable = watch_with_error_handler(
            &reftable_dir,
            move |_event| schedule_refresh(&reftable_inner),
            move || handle_git_watcher_error(&error_inner),
        );
        lock(inner).watchers.reftable = reftable;
    }
}

/// The mtime/ctime/size triple the polling fallback compares, node's
/// `Stats` comparison.
fn file_signature(path: &str) -> Option<(SystemTime, SystemTime, u64)> {
    let metadata = std::fs::metadata(path).ok()?;
    Some((
        metadata.modified().ok()?,
        metadata.created().ok()?,
        metadata.len(),
    ))
}

/// Whether the repo sits on a Windows mount under WSL, upstream's
/// `isWslEnvironment() && isWindowsMountedRepoPath(...)`.
#[must_use]
pub fn should_poll_git_head(repo_dir: &str, env: &EnvLookup) -> bool {
    if !cfg!(target_os = "linux") {
        return false;
    }
    let wsl = env("WSL_DISTRO_NAME").is_some() || env("WSL_INTEROP").is_some();
    if !wsl {
        return false;
    }
    let Some(rest) = repo_dir.strip_prefix("/mnt/") else {
        return false;
    };
    let mut chars = rest.chars();
    match chars.next() {
        Some(drive) if drive.is_ascii_alphabetic() => chars.next().is_none_or(|next| next == '/'),
        _ => false,
    }
}
