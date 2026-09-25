//! The nodejs execution environment, ported from upstream
//! `src/harness/env/nodejs.ts`.
//!
//! `tokio::process` replaces `node:child_process` and tokio-fs replaces
//! `node:fs/promises`; the spill machinery awaits writes inline where
//! upstream paused streams under high-water-mark backpressure, so the
//! backpressure dance and the post-exit grace timer collapse into the
//! reader-drain loop. Windows is out of scope per the map: the win32-only
//! branches ride their own ticket, and win32-only tests skip.

use std::collections::{BTreeMap, HashSet};
use std::future::Future;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::ExitStatus;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use pi_ai::types::BoxedFuture;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

use crate::harness::context::{AbortReason, AbortSignal, Context};
use crate::harness::types::{
    CreateDirOptions, ExecutionEnv, ExecutionError, ExecutionErrorCode, FileContent, FileError,
    FileErrorCode, FileInfo, FileKind, FileSystem, ReadTextLinesOptions, RemoveOptions, Shell,
    ShellExecOptions, ShellExecResult, TempFileOptions, TextLine, TextLineReader,
};
use crate::harness::utils::output_capture::{Chunk, OutputCapture, OutputCaptureHandlers};

/// Unix epoch milliseconds when the millisecond cap was recorded, upstream's
/// `MAX_TIMEOUT_MS`.
const MAX_TIMEOUT_MS: f64 = 2_147_483_647.0;

/// The seconds view of [`MAX_TIMEOUT_MS`], upstream's `MAX_TIMEOUT_SECONDS`.
const MAX_TIMEOUT_SECONDS: f64 = MAX_TIMEOUT_MS / 1000.0;

/// How long the `which bash` probe waits before killing the probe, upstream's
/// `findBashOnPath` timeout.
const BASH_PROBE_TIMEOUT_MS: u64 = 5_000;

/// The name prefix the spill file carries, upstream's `createTempFile`
/// argument.
const SPILL_FILE_PREFIX: &str = "pi-output-";

/// The name suffix the spill file carries, upstream's `spill.log`.
const SPILL_FILE_SUFFIX: &str = ".log";

/// The temp-name uniqueness counter; the process id plus this counter gives
/// `mkdtemp`-grade uniqueness without a random-uuid dependency.
static TEMP_NAME_COUNTER: AtomicU64 = AtomicU64::new(0);

/// An aborted file operation's fixed error, upstream's
/// `new FileError("aborted", "aborted", path)`.
fn aborted_file_error(path: Option<String>) -> FileError {
    FileError::new(FileErrorCode::Aborted, "aborted", path, None)
}

/// The context's abort signal when it already fired, upstream's
/// `signal?.aborted` checks.
fn fired_abort_signal(context: &Context) -> Option<AbortSignal> {
    context.abort_signal().filter(AbortSignal::aborted)
}

/// Resolves the shell timeout into milliseconds, upstream's
/// `resolveTimeoutMs`.
///
/// # Errors
/// A timeout `ExecutionError` when the timeout is not a positive finite
/// number of seconds or exceeds the millisecond cap.
fn resolve_timeout_ms(timeout: Option<f64>) -> Result<Option<u64>, ExecutionError> {
    let Some(timeout) = timeout else {
        return Ok(None);
    };
    if !timeout.is_finite() || timeout <= 0.0 {
        return Err(ExecutionError::new(
            ExecutionErrorCode::Timeout,
            "Invalid timeout: must be a finite number of seconds",
            None,
        ));
    }
    let timeout_ms = timeout * 1000.0;
    if timeout_ms > MAX_TIMEOUT_MS {
        return Err(ExecutionError::new(
            ExecutionErrorCode::Timeout,
            format!("Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds"),
            None,
        ));
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the timeout is validated positive, finite, and under the millisecond cap above; sub-millisecond precision is not observable"
    )]
    Ok(Some(timeout_ms as u64))
}

/// Percent-decodes the path portion of a `file://` URL, upstream's
/// `fileURLToPath`. Malformed input keeps its original text so filesystem
/// methods preserve the non-throwing contract.
fn file_url_to_path(url: &str) -> String {
    let Some(rest) = url.strip_prefix("file://") else {
        return url.to_owned();
    };
    // The authority half is empty for local files; the loopback markers are
    // the only hosts a local path carries.
    let path = rest
        .strip_prefix("localhost/")
        .or_else(|| rest.strip_prefix("127.0.0.1/"))
        .unwrap_or(rest);
    let bytes = path.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' if index + 2 < bytes.len() => {
                if let Ok(byte) = u8::from_str_radix(&path[index + 1..index + 3], 16) {
                    decoded.push(byte);
                    index += 3;
                } else {
                    decoded.push(bytes[index]);
                    index += 1;
                }
            }
            byte => {
                decoded.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

/// Resolves a caller path against the environment cwd, upstream's
/// `resolvePath`.
fn resolve_path(cwd: &str, path: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    let normalized = if path == "~" && !home.is_empty() {
        home
    } else if let Some(rest) = path.strip_prefix("~/") {
        if home.is_empty() {
            path.to_owned()
        } else {
            Path::new(&home).join(rest).to_string_lossy().into_owned()
        }
    } else if path.starts_with("file://") {
        file_url_to_path(path)
    } else {
        path.to_owned()
    };
    let path = Path::new(&normalized);
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        Path::new(cwd).join(path)
    };
    normalize_path(&resolved).to_string_lossy().into_owned()
}

/// Normalizes `.`/`..` segments and duplicate separators, upstream's
/// `resolve`'s lexical pass.
fn normalize_path(path: &Path) -> PathBuf {
    let mut components: Vec<std::ffi::OsString> = Vec::new();
    let mut prefix: Option<std::ffi::OsString> = None;
    let mut absolute = false;
    for component in path.components() {
        match component {
            std::path::Component::Prefix(prefix_part) => {
                prefix = Some(prefix_part.as_os_str().to_owned());
            }
            std::path::Component::RootDir => absolute = true,
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                components.pop();
            }
            std::path::Component::Normal(part) => components.push(part.to_owned()),
        }
    }
    let mut joined = if absolute {
        PathBuf::from("/")
    } else {
        prefix.map_or_else(PathBuf::new, PathBuf::from)
    };
    for component in components {
        joined.push(component);
    }
    if joined.as_os_str().is_empty() {
        joined.push(".");
    }
    joined
}

/// Builds one [`FileInfo`] from lstat data, upstream's
/// `fileInfoFromStats`.
fn file_info_from_metadata(
    path: &str,
    metadata: &std::fs::Metadata,
) -> Result<FileInfo, FileError> {
    let file_type = metadata.file_type();
    let kind = if file_type.is_file() {
        FileKind::File
    } else if file_type.is_dir() {
        FileKind::Directory
    } else if file_type.is_symlink() {
        FileKind::Symlink
    } else {
        return Err(FileError::new(
            FileErrorCode::Invalid,
            "Unsupported file type",
            Some(path.to_owned()),
            None,
        ));
    };
    Ok(FileInfo {
        name: Path::new(path).file_name().map_or_else(
            || path.to_owned(),
            |name| name.to_string_lossy().into_owned(),
        ),
        path: path.to_owned(),
        kind,
        size: metadata.len(),
        mtime_ms: mtime_ms_of(metadata),
    })
}

/// The file's modification time, upstream's `mtimeMs`; `0` when the
/// platform cannot resolve it, so callers never see a fabricated date.
fn mtime_ms_of(metadata: &std::fs::Metadata) -> i64 {
    metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX)
        })
}

/// Maps an OS error onto the backend-independent file error, upstream's
/// `toFileError`.
fn to_file_error(error: &std::io::Error, path: Option<String>) -> FileError {
    let code = match error.kind() {
        ErrorKind::NotFound => FileErrorCode::NotFound,
        ErrorKind::PermissionDenied => FileErrorCode::PermissionDenied,
        ErrorKind::NotADirectory => FileErrorCode::NotDirectory,
        ErrorKind::IsADirectory => FileErrorCode::IsDirectory,
        ErrorKind::InvalidInput | ErrorKind::InvalidData => FileErrorCode::Invalid,
        _ => FileErrorCode::Unknown,
    };
    FileError::new(
        code,
        error.to_string(),
        path,
        Some(Box::new(std::io::Error::from(error.kind()))),
    )
}

/// Whether the addressed path exists, upstream's `pathExists`.
async fn path_exists(path: &str) -> bool {
    tokio::fs::metadata(path).await.is_ok()
}

/// One shell invocation's transport, upstream's `ShellConfig`.
#[derive(Clone, Debug)]
struct ShellConfig {
    /// The shell executable.
    shell: String,
    /// The shell's fixed arguments.
    args: Vec<String>,
    /// The command transport, upstream's `"argv" | "stdin"`.
    command_transport: CommandTransport,
}

/// The command transport, upstream's `"argv" | "stdin"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommandTransport {
    /// The command rides the shell's argv; the default.
    Argv,
    /// The command rides the shell's stdin; legacy WSL bash only.
    Stdin,
}

/// Whether the path is a legacy WSL bash, upstream's
/// `isLegacyWslBashPath`.
fn is_legacy_wsl_bash_path(path: &str) -> bool {
    let normalized = path.replace('/', "\\").to_ascii_lowercase();
    normalized.contains(":\\windows\\system32\\bash.exe")
        || normalized.contains(":\\windows\\sysnative\\bash.exe")
}

/// Builds the shell config for one resolved bash, upstream's
/// `getBashShellConfig`.
fn get_bash_shell_config(shell: String) -> ShellConfig {
    if is_legacy_wsl_bash_path(&shell) {
        ShellConfig {
            shell,
            args: vec!["-s".to_owned()],
            command_transport: CommandTransport::Stdin,
        }
    } else {
        ShellConfig {
            shell,
            args: vec!["-c".to_owned()],
            command_transport: CommandTransport::Argv,
        }
    }
}

/// Resolves the shell the environment executes through, upstream's
/// `getShellConfig`. The win32 Git-for-Windows discovery rides the map's
/// win32 ticket; the unix branch checks `/bin/bash`, then `which bash`,
/// then falls back to `sh -c`.
///
/// # Errors
/// A `shell_unavailable` `ExecutionError` when a configured shell path
/// does not exist.
async fn get_shell_config(custom_shell_path: Option<&str>) -> Result<ShellConfig, ExecutionError> {
    if let Some(custom_shell_path) = custom_shell_path {
        if path_exists(custom_shell_path).await {
            return Ok(get_bash_shell_config(custom_shell_path.to_owned()));
        }
        return Err(ExecutionError::new(
            ExecutionErrorCode::ShellUnavailable,
            format!("Custom shell path not found: {custom_shell_path}"),
            None,
        ));
    }
    if path_exists("/bin/bash").await {
        return Ok(get_bash_shell_config("/bin/bash".to_owned()));
    }
    if let Some(bash_on_path) = find_bash_on_path().await {
        return Ok(get_bash_shell_config(bash_on_path));
    }
    Ok(ShellConfig {
        shell: "sh".to_owned(),
        args: vec!["-c".to_owned()],
        command_transport: CommandTransport::Argv,
    })
}

/// Builds the child's environment, upstream's `getShellEnv`.
fn get_shell_env(
    base_env: Option<&BTreeMap<String, String>>,
    extra_env: Option<&BTreeMap<String, String>>,
    inherit_env: bool,
) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    if inherit_env {
        for (key, value) in std::env::vars() {
            env.insert(key, value);
        }
        if let Some(base_env) = base_env {
            for (key, value) in base_env {
                env.insert(key.clone(), value.clone());
            }
        }
    }
    if let Some(extra_env) = extra_env {
        for (key, value) in extra_env {
            env.insert(key.clone(), value.clone());
        }
    }
    env
}

/// Kills the process tree rooted at one detached child, upstream's
/// `killProcessTree`. The environment spawns detached (its own process
/// group), so a negative-pid `SIGKILL` reaches the whole tree; a failed
/// group kill falls back to the child alone. The win32 `taskkill` branch
/// rides the map's win32 ticket.
#[expect(
    clippy::cast_possible_wrap,
    reason = "the kernel's pid_t is i32; a process id beyond it cannot exist on a supported platform"
)]
fn kill_process_tree(pid: u32) {
    #[cfg(unix)]
    {
        use nix::sys::signal::{Signal, kill};
        use nix::unistd::Pid;
        let group = Pid::from_raw(-(pid as i32));
        if kill(group, Signal::SIGKILL).is_err() {
            let _ = kill(Pid::from_raw(pid as i32), Signal::SIGKILL);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
    }
}

/// Runs one command capturing stdout, upstream's `runCommand`. Spawn
/// failures and timeouts report an empty stdout with no status, so the
/// probe callers fall through to their next candidate.
async fn run_command(command: &str, args: &[&str], timeout_ms: u64) -> (String, Option<i32>) {
    let Ok(mut child) = Command::new(command)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    else {
        return (String::new(), None);
    };
    let mut pipe = child.stdout.take();
    let probe = async {
        let mut stdout = Vec::new();
        if let Some(pipe) = pipe.as_mut() {
            let _ = pipe.read_to_end(&mut stdout).await;
        }
        let status = child.wait().await.ok().and_then(|status| status.code());
        (String::from_utf8_lossy(&stdout).into_owned(), status)
    };
    if let Ok(outcome) = tokio::time::timeout(Duration::from_millis(timeout_ms), probe).await {
        outcome
    } else {
        if let Some(pid) = child.id() {
            kill_process_tree(pid);
        }
        let _ = child.wait().await;
        (String::new(), None)
    }
}

/// Finds the first bash on `PATH`, upstream's `findBashOnPath`. The win32
/// `where bash.exe` probe rides the map's win32 ticket.
async fn find_bash_on_path() -> Option<String> {
    let (stdout, status) = run_command("which", &["bash"], BASH_PROBE_TIMEOUT_MS).await;
    if status != Some(0) || stdout.is_empty() {
        return None;
    }
    let first_match = stdout
        .trim()
        .lines()
        .next()
        .map(str::to_owned)
        .unwrap_or_default();
    if first_match.is_empty() || !path_exists(&first_match).await {
        return None;
    }
    Some(first_match)
}

/// One unique temp-name segment, upstream's `mkdtemp`/`randomUUID` suffix.
fn temp_name(counter: u64) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
        });
    format!("{:x}-{:x}-{:x}", std::process::id(), nanos, counter)
}

/// Reads the next UTF-8 line from one open file with strict LF
/// termination, upstream's `NodeTextLineReader`. Node readline does not
/// report whether its final line was newline-terminated; this reader does.
pub struct NodeTextLineReader {
    file: tokio::fs::File,
    path: String,
    buffered: Vec<u8>,
    ended: bool,
    closed: bool,
}

impl std::fmt::Debug for NodeTextLineReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeTextLineReader")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

/// The armed timeout wait, upstream's `setTimeout` handle.
type TimeoutWait = Pin<Box<dyn Future<Output = ()> + Send>>;

/// The abort wait, upstream's `signal.addEventListener("abort", ...)`.
type AbortWait = Pin<Box<dyn Future<Output = AbortReason> + Send>>;

impl NodeTextLineReader {
    /// The line reader over one open file, upstream's constructor.
    const fn new(file: tokio::fs::File, path: String) -> Self {
        Self {
            file,
            path,
            buffered: Vec::new(),
            ended: false,
            closed: false,
        }
    }
}

impl TextLineReader for NodeTextLineReader {
    /// Reads the next line, upstream's `readLine`.
    ///
    /// # Errors
    /// An `invalid` `FileError` after [`close`](TextLineReader::close),
    /// the abort error when the context signal fired before or during the
    /// read, and any other backend failure.
    fn read_line<'a>(
        &'a mut self,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Option<TextLine>, FileError>> {
        Box::pin(async move {
            let path = self.path.clone();
            if fired_abort_signal(context).is_some() {
                return Err(aborted_file_error(Some(path)));
            }
            if self.closed {
                return Err(FileError::new(
                    FileErrorCode::Invalid,
                    "Text line reader is closed",
                    Some(path),
                    None,
                ));
            }
            loop {
                if let Some(newline) = self.buffered.iter().position(|byte| *byte == b'\n') {
                    let line: Vec<u8> = self.buffered.drain(..newline).collect();
                    self.buffered.remove(0);
                    return Ok(Some(TextLine {
                        text: String::from_utf8_lossy(&line).into_owned(),
                        terminated: true,
                    }));
                }
                if self.ended {
                    if self.buffered.is_empty() {
                        return Ok(None);
                    }
                    let text = String::from_utf8_lossy(&self.buffered).into_owned();
                    self.buffered.clear();
                    return Ok(Some(TextLine {
                        text,
                        terminated: false,
                    }));
                }
                if fired_abort_signal(context).is_some() {
                    return Err(aborted_file_error(Some(path)));
                }
                // Reads are sequential, so an aborted read resumes at the
                // stream position without skipping bytes, upstream's
                // positioned-read retry guarantee.
                let mut chunk = vec![0_u8; 64 * 1024];
                match self.file.read(&mut chunk).await {
                    Err(error) => return Err(to_file_error(&error, Some(path))),
                    Ok(0) => self.ended = true,
                    Ok(bytes_read) => self.buffered.extend_from_slice(&chunk[..bytes_read]),
                }
            }
        })
    }

    /// Releases the open file, upstream's `close`. Closing is best-effort,
    /// including after cancellation or an earlier I/O failure; dropping
    /// the handle discards the rest of the buffered bytes.
    fn close<'a>(&'a mut self, _context: &'a Context) -> BoxedFuture<'a, ()> {
        Box::pin(async move {
            self.closed = true;
            self.buffered.clear();
        })
    }
}

/// The nodejs execution environment, upstream's `NodeExecutionEnv`.
pub struct NodeExecutionEnv {
    cwd: String,
    shell_path: Option<String>,
    shell_env: Option<BTreeMap<String, String>>,
    active_child_pids: Mutex<HashSet<u32>>,
}

impl std::fmt::Debug for NodeExecutionEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeExecutionEnv")
            .field("cwd", &self.cwd)
            .finish_non_exhaustive()
    }
}

impl NodeExecutionEnv {
    /// Builds the environment over one working directory, upstream's
    /// constructor.
    #[must_use]
    pub fn new(
        cwd: String,
        shell_path: Option<String>,
        shell_env: Option<BTreeMap<String, String>>,
    ) -> Self {
        Self {
            cwd,
            shell_path,
            shell_env,
            active_child_pids: Mutex::new(HashSet::new()),
        }
    }
}

impl FileSystem for NodeExecutionEnv {
    fn cwd(&self) -> &str {
        &self.cwd
    }

    fn absolute_path<'a>(
        &'a self,
        path: &'a str,
        _context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        Box::pin(async move { Ok(resolve_path(&self.cwd, path)) })
    }

    fn join_path<'a>(
        &'a self,
        parts: &'a [String],
        _context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        Box::pin(async move {
            let mut joined = PathBuf::new();
            for part in parts {
                joined.push(part);
            }
            Ok(joined.to_string_lossy().into_owned())
        })
    }

    fn read_text_file<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        Box::pin(async move {
            let resolved = resolve_path(&self.cwd, path);
            if fired_abort_signal(context).is_some() {
                return Err(aborted_file_error(Some(resolved)));
            }
            tokio::fs::read_to_string(&resolved)
                .await
                .map_err(|error| to_file_error(&error, Some(resolved)))
        })
    }

    fn open_text_line_reader<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Box<dyn TextLineReader>, FileError>> {
        Box::pin(async move {
            let resolved = resolve_path(&self.cwd, path);
            if fired_abort_signal(context).is_some() {
                return Err(aborted_file_error(Some(resolved)));
            }
            match tokio::fs::File::open(&resolved).await {
                Err(error) => Err(to_file_error(&error, Some(resolved))),
                Ok(file) => {
                    if fired_abort_signal(context).is_some() {
                        return Err(aborted_file_error(Some(resolved)));
                    }
                    let reader: Box<dyn TextLineReader> =
                        Box::new(NodeTextLineReader::new(file, resolved));
                    Ok(reader)
                }
            }
        })
    }

    fn read_text_lines<'a>(
        &'a self,
        path: &'a str,
        options: Option<ReadTextLinesOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Vec<String>, FileError>> {
        Box::pin(async move {
            let max_lines = options.and_then(|options| options.max_lines);
            if max_lines == Some(0) {
                return Ok(Vec::new());
            }
            let mut reader = FileSystem::open_text_line_reader(self, path, context).await?;
            let mut lines = Vec::new();
            while max_lines
                .is_none_or(|max| usize::try_from(max).unwrap_or(usize::MAX) > lines.len())
            {
                match reader.read_line(context).await {
                    Err(error) => {
                        reader.close(context).await;
                        return Err(error);
                    }
                    Ok(None) => break,
                    Ok(Some(line)) => lines.push(line.text),
                }
            }
            reader.close(context).await;
            Ok(lines)
        })
    }

    fn read_binary_file<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Vec<u8>, FileError>> {
        Box::pin(async move {
            let resolved = resolve_path(&self.cwd, path);
            if fired_abort_signal(context).is_some() {
                return Err(aborted_file_error(Some(resolved)));
            }
            tokio::fs::read(&resolved)
                .await
                .map_err(|error| to_file_error(&error, Some(resolved)))
        })
    }

    fn write_file<'a>(
        &'a self,
        path: &'a str,
        content: FileContent,
        ctx: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>> {
        Box::pin(async move {
            let resolved = resolve_path(&self.cwd, path);
            if fired_abort_signal(ctx).is_some() {
                return Err(aborted_file_error(Some(resolved)));
            }
            if let Some(parent) = Path::new(&resolved).parent()
                && let Err(error) = tokio::fs::create_dir_all(parent).await
            {
                return Err(to_file_error(&error, Some(resolved)));
            }
            if fired_abort_signal(ctx).is_some() {
                return Err(aborted_file_error(Some(resolved)));
            }
            let write = match &content {
                FileContent::Text(text) => tokio::fs::write(&resolved, text.as_bytes()).await,
                FileContent::Bytes(bytes) => tokio::fs::write(&resolved, bytes).await,
            };
            write.map_err(|error| to_file_error(&error, Some(resolved)))
        })
    }

    fn append_file<'a>(
        &'a self,
        path: &'a str,
        content: FileContent,
        ctx: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>> {
        Box::pin(async move {
            let resolved = resolve_path(&self.cwd, path);
            if fired_abort_signal(ctx).is_some() {
                return Err(aborted_file_error(Some(resolved)));
            }
            if let Some(parent) = Path::new(&resolved).parent()
                && let Err(error) = tokio::fs::create_dir_all(parent).await
            {
                return Err(to_file_error(&error, Some(resolved)));
            }
            if fired_abort_signal(ctx).is_some() {
                return Err(aborted_file_error(Some(resolved)));
            }
            let mut file = match tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&resolved)
                .await
            {
                Err(error) => return Err(to_file_error(&error, Some(resolved))),
                Ok(file) => file,
            };
            let written = match content {
                FileContent::Text(text) => file.write_all(text.as_bytes()).await,
                FileContent::Bytes(bytes) => file.write_all(&bytes).await,
            };
            if let Err(error) = written {
                return Err(to_file_error(&error, Some(resolved)));
            }
            if fired_abort_signal(ctx).is_some() {
                return Err(aborted_file_error(Some(resolved)));
            }
            Ok(())
        })
    }

    fn rename_file<'a>(
        &'a self,
        source_path: &'a str,
        destination_path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>> {
        Box::pin(async move {
            let source = resolve_path(&self.cwd, source_path);
            let destination = resolve_path(&self.cwd, destination_path);
            if fired_abort_signal(context).is_some() {
                return Err(aborted_file_error(Some(destination)));
            }
            tokio::fs::rename(&source, &destination)
                .await
                .map_err(|error| to_file_error(&error, Some(source)))
        })
    }

    fn file_info<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<FileInfo, FileError>> {
        Box::pin(async move {
            let resolved = resolve_path(&self.cwd, path);
            if fired_abort_signal(context).is_some() {
                return Err(aborted_file_error(Some(resolved)));
            }
            match tokio::fs::symlink_metadata(&resolved).await {
                Err(error) => Err(to_file_error(&error, Some(resolved))),
                Ok(metadata) => file_info_from_metadata(&resolved, &metadata),
            }
        })
    }

    fn list_dir<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Vec<FileInfo>, FileError>> {
        Box::pin(async move {
            let resolved = resolve_path(&self.cwd, path);
            if fired_abort_signal(context).is_some() {
                return Err(aborted_file_error(Some(resolved)));
            }
            let mut entries = match tokio::fs::read_dir(&resolved).await {
                Err(error) => return Err(to_file_error(&error, Some(resolved))),
                Ok(entries) => entries,
            };
            let mut infos = Vec::new();
            while let Some(entry) = entries.next_entry().await.transpose() {
                if fired_abort_signal(context).is_some() {
                    return Err(aborted_file_error(Some(resolved)));
                }
                let entry_path = match entry {
                    Err(error) => return Err(to_file_error(&error, Some(resolved))),
                    Ok(entry) => entry.path().to_string_lossy().into_owned(),
                };
                match tokio::fs::symlink_metadata(&entry_path).await {
                    Err(error) => return Err(to_file_error(&error, Some(entry_path))),
                    Ok(metadata) => infos.push(file_info_from_metadata(&entry_path, &metadata)?),
                }
            }
            Ok(infos)
        })
    }

    fn canonical_path<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        Box::pin(async move {
            let resolved = resolve_path(&self.cwd, path);
            if fired_abort_signal(context).is_some() {
                return Err(aborted_file_error(Some(resolved)));
            }
            tokio::fs::canonicalize(&resolved)
                .await
                .map(|canonical| canonical.to_string_lossy().into_owned())
                .map_err(|error| to_file_error(&error, Some(resolved)))
        })
    }

    fn exists<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<bool, FileError>> {
        Box::pin(async move {
            match FileSystem::file_info(self, path, context).await {
                Ok(_) => Ok(true),
                Err(error) if error.code == FileErrorCode::NotFound => Ok(false),
                Err(error) => Err(error),
            }
        })
    }

    fn create_dir<'a>(
        &'a self,
        path: &'a str,
        options: Option<CreateDirOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>> {
        Box::pin(async move {
            let resolved = resolve_path(&self.cwd, path);
            if fired_abort_signal(context).is_some() {
                return Err(aborted_file_error(Some(resolved)));
            }
            let recursive = options
                .and_then(|options| options.recursive)
                .unwrap_or(true);
            let created = if recursive {
                tokio::fs::create_dir_all(&resolved).await
            } else {
                tokio::fs::create_dir(&resolved).await
            };
            created.map_err(|error| to_file_error(&error, Some(resolved)))
        })
    }

    fn remove<'a>(
        &'a self,
        path: &'a str,
        options: Option<RemoveOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>> {
        Box::pin(async move {
            let resolved = resolve_path(&self.cwd, path);
            if fired_abort_signal(context).is_some() {
                return Err(aborted_file_error(Some(resolved)));
            }
            let recursive = options
                .and_then(|options| options.recursive)
                .unwrap_or(false);
            let force = options.and_then(|options| options.force).unwrap_or(false);
            let removed = if recursive {
                match tokio::fs::symlink_metadata(&resolved).await {
                    Err(error) => Err(error),
                    Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
                        tokio::fs::remove_dir_all(&resolved).await
                    }
                    Ok(_) => tokio::fs::remove_file(&resolved).await,
                }
            } else {
                tokio::fs::remove_file(&resolved).await
            };
            match removed {
                Ok(()) => Ok(()),
                Err(error) => {
                    let file_error = to_file_error(&error, Some(resolved));
                    if force && file_error.code == FileErrorCode::NotFound {
                        Ok(())
                    } else {
                        Err(file_error)
                    }
                }
            }
        })
    }

    fn create_temp_dir<'a>(
        &'a self,
        prefix: Option<&'a str>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        Box::pin(async move {
            if fired_abort_signal(context).is_some() {
                return Err(aborted_file_error(None));
            }
            let prefix = prefix.unwrap_or("tmp-");
            let dir = std::env::temp_dir().join(format!(
                "{prefix}{}",
                temp_name(TEMP_NAME_COUNTER.fetch_add(1, Ordering::Relaxed))
            ));
            let path = dir.to_string_lossy().into_owned();
            tokio::fs::create_dir(&dir)
                .await
                .map(|()| path)
                .map_err(|error| to_file_error(&error, None))
        })
    }

    fn create_temp_file<'a>(
        &'a self,
        options: Option<TempFileOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        Box::pin(async move {
            let dir = FileSystem::create_temp_dir(self, Some("tmp-"), context).await?;
            let prefix = options
                .as_ref()
                .and_then(|options| options.prefix.as_deref())
                .unwrap_or("");
            let suffix = options
                .as_ref()
                .and_then(|options| options.suffix.as_deref())
                .unwrap_or("");
            let file_path = Path::new(&dir)
                .join(format!(
                    "{prefix}{}{suffix}",
                    temp_name(TEMP_NAME_COUNTER.fetch_add(1, Ordering::Relaxed))
                ))
                .to_string_lossy()
                .into_owned();
            match tokio::fs::File::create(&file_path).await {
                Err(error) => Err(to_file_error(&error, Some(file_path))),
                Ok(_) => Ok(file_path),
            }
        })
    }

    fn cleanup<'a>(&'a self, _context: &'a Context) -> BoxedFuture<'a, ()> {
        Box::pin(async move {})
    }
}

impl Shell for NodeExecutionEnv {
    fn exec<'a>(
        &'a self,
        command: &'a str,
        options: Option<ShellExecOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<ShellExecResult, ExecutionError>> {
        Box::pin(exec_command(
            command.to_owned(),
            options,
            context.clone(),
            self.cwd.clone(),
            self.shell_path.clone(),
            self.shell_env.clone(),
            &self.active_child_pids,
            self,
        ))
    }

    fn cleanup<'a>(&'a self, _context: &'a Context) -> BoxedFuture<'a, ()> {
        Box::pin(async move {
            let pids = self
                .active_child_pids
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .drain()
                .collect::<Vec<_>>();
            for pid in pids {
                kill_process_tree(pid);
            }
        })
    }
}

impl ExecutionEnv for NodeExecutionEnv {}

/// The shared failure slots the capture's reporter and the driver share,
/// upstream's `callbackError` variable plus the killable child pid.
struct ExecFailure {
    callback_error: Mutex<Option<ExecutionError>>,
    pid: Mutex<Option<u32>>,
}

/// The spill file a driver preserves complete output through, upstream's
/// `spillStream`/`spillPath` pair plus the held pre-truncation chunks.
struct SpillState {
    file: Option<(String, tokio::fs::File)>,
    prefixes: Vec<Vec<u8>>,
    error: Option<ExecutionError>,
}

impl SpillState {
    const fn new() -> Self {
        Self {
            file: None,
            prefixes: Vec::new(),
            error: None,
        }
    }
}

/// The shell-exec driver, upstream's `NodeExecutionEnv.exec` promise body:
/// settle ordering, spill state, and kill bookkeeping read together.
#[expect(
    clippy::too_many_lines,
    reason = "the driver mirrors upstream's single exec closure; splitting it would \
    separate the settle ordering from the state it reads"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the parameters are the resolved exec environment (command, options, context, resolved shell, kill bookkeeping, the file seam); a bundle struct would only re-wrap them for one caller"
)]
async fn exec_command(
    command: String,
    options: Option<ShellExecOptions>,
    context: Context,
    env_cwd: String,
    shell_path: Option<String>,
    shell_env: Option<BTreeMap<String, String>>,
    active_child_pids: &Mutex<HashSet<u32>>,
    files: &dyn FileSystem,
) -> Result<ShellExecResult, ExecutionError> {
    let signal = context.abort_signal();
    if signal.as_ref().is_some_and(AbortSignal::aborted) {
        return Err(ExecutionError::new(
            ExecutionErrorCode::Aborted,
            "aborted",
            None,
        ));
    }
    let timeout_seconds = options.as_ref().and_then(|options| options.timeout);
    let timeout_ms = resolve_timeout_ms(timeout_seconds)?;

    let cwd = options
        .as_ref()
        .and_then(|options| options.cwd.as_deref())
        .map_or_else(|| env_cwd.clone(), |cwd| resolve_path(&env_cwd, cwd));
    let shell_config = get_shell_config(shell_path.as_deref()).await?;
    if tokio::fs::metadata(&cwd).await.is_err() {
        return Err(ExecutionError::new(
            ExecutionErrorCode::SpawnError,
            format!("Working directory does not exist: {cwd}\nCannot execute bash commands."),
            None,
        ));
    }

    let on_update = options
        .as_ref()
        .and_then(|options| options.on_update.clone());
    let spill_requested = options
        .as_ref()
        .and_then(|options| options.capture.as_ref())
        .is_some_and(|capture| capture.spill);

    // The capture's failure reporter records the first callback failure and
    // kills the child; the pid arrives after the capture exists, so it
    // rides its own shared slot.
    let failure = Arc::new(ExecFailure {
        callback_error: Mutex::new(None),
        pid: Mutex::new(None),
    });
    let report_failure: Arc<dyn Fn(String) + Send + Sync> = {
        let failure = Arc::clone(&failure);
        Arc::new(move |message: String| {
            {
                let mut error = failure
                    .callback_error
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if error.is_none() {
                    *error = Some(ExecutionError::new(
                        ExecutionErrorCode::CallbackError,
                        message,
                        None,
                    ));
                }
            }
            let pid = failure
                .pid
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            if let Some(pid) = pid {
                kill_process_tree(pid);
            }
        })
    };

    let mut capture = match OutputCapture::new(
        options
            .as_ref()
            .and_then(|options| options.capture.as_ref()),
        context.clone(),
        OutputCaptureHandlers {
            on_update,
            on_error: Arc::clone(&report_failure),
        },
    ) {
        Err(message) => {
            return Err(ExecutionError::new(
                ExecutionErrorCode::Unknown,
                message,
                None,
            ));
        }
        Ok(capture) => capture,
    };

    let command_from_stdin = shell_config.command_transport == CommandTransport::Stdin;
    let mut shell_command = Command::new(&shell_config.shell);
    if command_from_stdin {
        shell_command.args(&shell_config.args);
        shell_command.stdin(std::process::Stdio::piped());
    } else {
        shell_command
            .args(&shell_config.args)
            .arg(&command)
            .stdin(std::process::Stdio::null());
    }
    shell_command
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .current_dir(&cwd);
    for (key, value) in get_shell_env(
        shell_env.as_ref(),
        options.as_ref().and_then(|options| options.env.as_ref()),
        options
            .as_ref()
            .and_then(|options| options.inherit_env)
            .unwrap_or(true),
    ) {
        shell_command.env(key, value);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        shell_command.as_std_mut().process_group(0);
    }
    let mut child = match shell_command.spawn() {
        Err(error) => {
            return Err(ExecutionError::new(
                ExecutionErrorCode::SpawnError,
                error.to_string(),
                Some(Box::new(error)),
            ));
        }
        Ok(child) => child,
    };
    let pid = child.id();
    if let Some(pid) = pid {
        active_child_pids
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(pid);
        *failure
            .pid
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(pid);
    }
    if command_from_stdin && let Some(mut stdin) = child.stdin.take() {
        if let Err(error) = stdin.write_all(command.as_bytes()).await {
            return Err(ExecutionError::new(
                ExecutionErrorCode::SpawnError,
                error.to_string(),
                Some(Box::new(error)),
            ));
        }
        drop(stdin);
    }

    // Readers feed the driver through one channel; the driver owns the
    // capture and the spill state, upstream's stdout/stderr `data` feeds.
    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(16);
    let stdout_task = pipe_reader_task(child.stdout.take(), chunk_tx.clone());
    let stderr_task = pipe_reader_task(child.stderr.take(), chunk_tx.clone());
    // The driver holds no sender: the channel closes at reader EOF, which
    // is what ends the drain loop after the child exits.
    drop(chunk_tx);

    let mut spill = SpillState::new();
    let mut timed_out = false;
    let mut aborted = false;
    let mut exit_status: Option<std::io::Result<ExitStatus>> = None;
    let mut timeout_sleep: Option<TimeoutWait> = timeout_ms.map(|ms| {
        let sleep: TimeoutWait = Box::pin(tokio::time::sleep(Duration::from_millis(ms)));
        sleep
    });
    let mut abort_wait: Option<AbortWait> = signal.map(|signal| {
        let wait: AbortWait = Box::pin(signal.wait());
        wait
    });

    loop {
        tokio::select! {
            chunk = chunk_rx.recv() => {
                let Some(chunk) = chunk else {
                    break;
                };
                feed_chunk(
                    &mut capture,
                    &chunk,
                    spill_requested,
                    &mut spill,
                    files,
                    &context,
                    pid,
                )
                .await;
            }
            status = child.wait(), if exit_status.is_none() => {
                exit_status = Some(status);
            }
            () = wait_timeout(&mut timeout_sleep), if timeout_sleep.is_some() && !timed_out => {
                timed_out = true;
                if let Some(pid) = pid {
                    kill_process_tree(pid);
                }
            }
            _ = wait_abort(&mut abort_wait), if abort_wait.is_some() && !aborted => {
                aborted = true;
                if let Some(pid) = pid {
                    kill_process_tree(pid);
                }
            }
        }
    }
    let _ = tokio::join!(stdout_task, stderr_task);
    if exit_status.is_none() {
        exit_status = Some(child.wait().await);
    }
    if let Some(pid) = pid {
        active_child_pids
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&pid);
    }
    if let Some((_, mut spill_file)) = spill.file.take() {
        let _ = spill_file.flush().await;
    }

    capture.finish();
    capture.flush();
    let error = failure
        .callback_error
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    if let Some(error) = error {
        return Err(error);
    }
    if timed_out {
        return Err(ExecutionError::new(
            ExecutionErrorCode::Timeout,
            format!(
                "timeout:{}",
                timeout_seconds
                    .map_or_else(|| "undefined".to_owned(), |seconds| seconds.to_string())
            ),
            None,
        ));
    }
    if aborted {
        return Err(ExecutionError::new(
            ExecutionErrorCode::Aborted,
            "aborted",
            None,
        ));
    }
    if let Some(error) = spill.error {
        return Err(error);
    }
    let output = capture.snapshot();
    capture.dispose();
    let Some(status) = exit_status else {
        return Err(ExecutionError::new(
            ExecutionErrorCode::SpawnError,
            "The shell driver settled without an exit status",
            None,
        ));
    };
    let Ok(status) = status else {
        return Err(ExecutionError::new(
            ExecutionErrorCode::SpawnError,
            "The shell process failed while running",
            None,
        ));
    };
    // A process killed by a signal (e.g. OOM killer) has no exit code; map
    // it to the conventional 128 + signal number so callers do not mistake
    // it for a successful exit.
    let exit_code = status.code().unwrap_or_else(|| {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt as _;
            status
                .signal()
                .map_or(1, |signal_number| 128 + signal_number)
        }
        #[cfg(not(unix))]
        {
            1
        }
    });
    Ok(ShellExecResult {
        exit_code,
        truncation: output.metadata.truncation,
        spill_path: output.metadata.spill_path,
        last_line_bytes: output.metadata.last_line_bytes,
    })
}

/// Awaits the armed timeout, staying pending while none is armed, so the
/// select arm can carry the option without re-arming per loop.
async fn wait_timeout(sleep: &mut Option<TimeoutWait>) {
    if let Some(sleep) = sleep.as_mut() {
        sleep.await;
    } else {
        std::future::pending::<()>().await;
    }
}

/// Awaits the abort signal, mirroring [`wait_timeout`].
async fn wait_abort(wait: &mut Option<AbortWait>) -> AbortReason {
    if let Some(wait) = wait.as_mut() {
        wait.await
    } else {
        std::future::pending().await
    }
}

/// Feeds one raw chunk into the capture and the requested spill, upstream's
/// `feed`.
async fn feed_chunk(
    capture: &mut OutputCapture,
    chunk: &[u8],
    spill_requested: bool,
    spill: &mut SpillState,
    files: &dyn FileSystem,
    context: &Context,
    pid: Option<u32>,
) {
    let was_truncated = capture.truncated();
    capture.push(Chunk::Bytes(chunk));
    if !spill_requested || chunk.is_empty() {
        return;
    }
    if spill.file.is_some() || was_truncated {
        write_spill(capture, spill, chunk, files, context, pid).await;
    } else if capture.truncated() {
        let prefixes = std::mem::take(&mut spill.prefixes);
        for prefix in &prefixes {
            write_spill(capture, spill, prefix, files, context, pid).await;
        }
        write_spill(capture, spill, chunk, files, context, pid).await;
    } else {
        spill.prefixes.push(chunk.to_vec());
    }
}

/// Appends one raw chunk to the spill file, creating it on first use,
/// upstream's `startSpill`/`writeSpill`.
async fn write_spill(
    capture: &mut OutputCapture,
    spill: &mut SpillState,
    chunk: &[u8],
    files: &dyn FileSystem,
    context: &Context,
    pid: Option<u32>,
) {
    if spill.error.is_some() {
        return;
    }
    if spill.file.is_none() {
        let created = files
            .create_temp_file(
                Some(TempFileOptions {
                    prefix: Some(SPILL_FILE_PREFIX.to_owned()),
                    suffix: Some(SPILL_FILE_SUFFIX.to_owned()),
                }),
                context,
            )
            .await;
        let path = match created {
            Ok(path) => path,
            Err(error) => {
                fail_spill(spill, &error.message, pid);
                return;
            }
        };
        match tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
        {
            Err(error) => {
                fail_spill(spill, &error.to_string(), pid);
                return;
            }
            Ok(file) => {
                capture.set_spill_path(&path);
                spill.file = Some((path, file));
            }
        }
    }
    let Some((_, file)) = spill.file.as_mut() else {
        return;
    };
    if let Err(error) = file.write_all(chunk).await {
        fail_spill(spill, &error.to_string(), pid);
    }
}

/// Fails the spill, upstream's `failSpill`: the first failure wins and the
/// child dies, so no output is silently lost.
fn fail_spill(spill: &mut SpillState, message: &str, pid: Option<u32>) {
    if spill.error.is_some() {
        return;
    }
    spill.error = Some(ExecutionError::new(
        ExecutionErrorCode::Unknown,
        format!("Failed to preserve complete shell output: {message}"),
        None,
    ));
    if let Some(pid) = pid {
        kill_process_tree(pid);
    }
}

/// Pumps one child pipe into the driver's chunk channel, upstream's
/// `child.stdout?.on("data", feed)`.
fn pipe_reader_task<R>(
    mut pipe: Option<R>,
    tx: tokio::sync::mpsc::Sender<Vec<u8>>,
) -> tokio::task::JoinHandle<()>
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let Some(pipe) = pipe.as_mut() else {
            return;
        };
        let mut buffer = vec![0_u8; 64 * 1024];
        loop {
            match pipe.read(&mut buffer).await {
                Ok(0) | Err(_) => break,
                Ok(bytes_read) => {
                    if tx.send(buffer[..bytes_read].to_vec()).await.is_err() {
                        break;
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests;
