//! File watching with error containment, upstream's `src/utils/fs-watch.ts`.
//!
//! node's `fs.watch` restate onto the `notify` crate: the event callback
//! carries the two event names node emits (`rename` for create/remove, and
//! `change`), non-recursive by default like node, and the retry delay the
//! consumers re-arm with is a plain constant.

use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use notify::{RecommendedWatcher, RecursiveMode, Watcher as _};

/// How long the consumers wait before re-arming a failed watch, upstream's
/// `FS_WATCH_RETRY_DELAY_MS`.
pub const FS_WATCH_RETRY_DELAY_MS: Duration = Duration::from_millis(5000);

/// A live directory watch, upstream's `FSWatcher`; closing is the drop,
/// which stops the notify thread it owns.
#[derive(Debug)]
pub struct Watcher(
    #[allow(
        dead_code,
        reason = "the handle exists to keep the watch alive; closing is the drop"
    )]
    RecommendedWatcher,
);

/// A watched-path event, the `(eventType, filename)` pair node's listener
/// receives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchEvent {
    /// `"rename"` for create/remove, `"change"` for content and metadata
    /// updates — node's two event names.
    pub event_type: &'static str,
    /// The basename that changed, when the platform reported one.
    pub filename: Option<String>,
}

/// Close a watcher, ignoring close errors — the drop of the notify handle,
/// which never throws, upstream's `watcher.close()` try/catch.
pub fn close_watcher(watcher: Option<Watcher>) {
    drop(watcher);
}

/// Watch `path`, delivering events to `listener` and failures to
/// `on_error`, upstream's `watchWithErrorHandler`.
///
/// Creation failures call `on_error` and answer `None`, upstream's
/// try/catch around the constructor; runtime errors ride the event channel
/// and call it the same way.
#[must_use]
pub fn watch_with_error_handler(
    path: &str,
    listener: impl Fn(WatchEvent) + Send + 'static,
    on_error: impl Fn() + Send + Sync + 'static,
) -> Option<Watcher> {
    let on_error: Arc<dyn Fn() + Send + Sync> = Arc::new(on_error);
    let (sender, receiver) = mpsc::channel();
    let forwarder = std::thread::spawn(move || {
        while let Ok(event) = receiver.recv() {
            listener(event);
        }
    });
    let handler = {
        let sender = sender.clone();
        let on_error = Arc::clone(&on_error);
        move |result: Result<notify::Event, notify::Error>| match result {
            Ok(event) => {
                if let Some(event) = to_watch_event(&event) {
                    let _queued = sender.send(event);
                }
            }
            Err(_) => on_error(),
        }
    };
    let Ok(mut watcher) = notify::recommended_watcher(handler) else {
        on_error();
        drop(sender);
        let _joined = forwarder.join();
        return None;
    };
    if watcher
        .watch(Path::new(path), RecursiveMode::NonRecursive)
        .is_err()
    {
        on_error();
        return None;
    }
    let _detached = forwarder;
    Some(Watcher(watcher))
}

/// Map one notify event onto node's two event names: create and remove are
/// `rename` (`FSEvents` folds both into renames), content and metadata
/// edits are `change`. Access and other kinds drop.
fn to_watch_event(event: &notify::Event) -> Option<WatchEvent> {
    use notify::event::{EventKind, ModifyKind};
    let event_type = match event.kind {
        EventKind::Create(_) | EventKind::Remove(_) | EventKind::Modify(ModifyKind::Name(_)) => {
            "rename"
        }
        EventKind::Modify(_) => "change",
        _ => return None,
    };
    let filename = event.paths.first().and_then(|path| {
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
    });
    Some(WatchEvent {
        event_type,
        filename,
    })
}
