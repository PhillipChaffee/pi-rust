//! Keyed-instance lifetime, ported from upstream
//! `src/services/instances.ts`.
//!
//! The directory owns keyed instances and the observers watching them:
//! inserting, replacing, or removing an entry starts or cancels each
//! observer's per-entry task, and a task's cancellation is the abort of the
//! context the handler received — upstream's `withCancel` handle, since the
//! handler body itself runs synchronously.

use std::cell::{Cell, RefCell};
use std::panic::AssertUnwindSafe;
use std::rc::Rc;

use crate::context::with_cancel;
use crate::errors::ChordError;
use crate::handle::{ErrorReporter, ServiceTarget};
use crate::types::{KeyedServiceHandler, Unsubscribe};

/// One live keyed instance: its address, the bound service target, and the
/// deactivation the directory runs on removal.
pub struct InstanceDirectoryEntry {
    /// The instance key.
    pub key: String,
    /// The generation that fences stale references.
    pub generation: u64,
    /// The service target observers resolve views against.
    pub service: ServiceTarget,
    /// Deactivates the instance, upstream's per-entry `deactivate()`.
    pub deactivate: Rc<dyn Fn()>,
}

impl std::fmt::Debug for InstanceDirectoryEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstanceDirectoryEntry")
            .field("key", &self.key)
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}

struct Observer {
    handler: KeyedServiceHandler,
    tasks: RefCell<Vec<(Rc<InstanceDirectoryEntry>, crate::context::AbortController)>>,
    closed: Cell<bool>,
}

/// Owns keyed instance lifetime and the cancellable tasks observing those
/// instances, upstream's `InstanceDirectory`.
pub struct InstanceDirectory {
    entries: RefCell<Vec<(String, Rc<InstanceDirectoryEntry>)>>,
    // Shared, not owned: the unsubscribe closure outlives the observe call
    // and must retain its observer out of the live list, upstream's
    // `this.observers.filter(...)` on the same array.
    observers: Rc<RefCell<Vec<Rc<Observer>>>>,
    report_error: ErrorReporter,
    ready: Cell<bool>,
    disposed: Cell<bool>,
}

impl std::fmt::Debug for InstanceDirectory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstanceDirectory")
            .field("ready", &self.ready.get())
            .finish_non_exhaustive()
    }
}

impl InstanceDirectory {
    /// A directory that either starts observations immediately or waits for
    /// [`ready`](Self::ready).
    #[must_use]
    pub fn new(ready: bool, report_error: ErrorReporter) -> Self {
        Self {
            entries: RefCell::new(Vec::new()),
            observers: Rc::new(RefCell::new(Vec::new())),
            report_error,
            ready: Cell::new(ready),
            disposed: Cell::new(false),
        }
    }

    /// How many observers are watching.
    #[must_use]
    pub fn observer_count(&self) -> usize {
        self.observers.borrow().len()
    }

    /// The live entry under `key`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<Rc<InstanceDirectoryEntry>> {
        self.entries
            .borrow()
            .iter()
            .find(|(stored, _)| stored == key)
            .map(|(_, entry)| entry.clone())
    }

    /// Inserts a fresh instance.
    ///
    /// # Errors
    /// [`ChordError`] when the directory is disposed or the key is live.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the entry is registered and observed; the by-value signature is the ported public surface"
    )]
    pub fn insert(&self, entry: Rc<InstanceDirectoryEntry>) -> Result<(), ChordError> {
        self.assert_active()?;
        if self
            .entries
            .borrow()
            .iter()
            .any(|(key, _)| *key == entry.key)
        {
            return Err(ChordError::Message(format!(
                "Keyed service already has a live instance with key {}",
                entry.key
            )));
        }
        self.entries
            .borrow_mut()
            .push((entry.key.clone(), entry.clone()));
        if self.ready.get() {
            self.start_all(&entry);
        }
        Ok(())
    }

    /// Replaces one instance's generation.
    ///
    /// # Errors
    /// [`ChordError`] when the directory is disposed or the live generation
    /// repeats.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the entry is registered and observed; the by-value signature is the ported public surface"
    )]
    pub fn replace(&self, entry: Rc<InstanceDirectoryEntry>) -> Result<(), ChordError> {
        self.assert_active()?;
        let previous = self.get(&entry.key);
        if let Some(previous) = previous {
            if previous.generation == entry.generation {
                return Err(ChordError::Message(
                    "Keyed service repeated a live generation".to_string(),
                ));
            }
            self.remove(&previous);
        }
        self.entries
            .borrow_mut()
            .push((entry.key.clone(), entry.clone()));
        if self.ready.get() {
            self.start_all(&entry);
        }
        Ok(())
    }

    /// Removes `entry` if it is still the live one for its key.
    pub fn remove(&self, entry: &Rc<InstanceDirectoryEntry>) {
        if !self
            .entries
            .borrow()
            .iter()
            .any(|(_, current)| Rc::ptr_eq(current, entry))
        {
            return;
        }
        self.remove_internal(entry);
    }

    /// Marks the directory ready, starting observations for every live
    /// entry.
    ///
    /// # Errors
    /// [`ChordError`] when the directory is disposed.
    pub fn ready(&self) -> Result<(), ChordError> {
        self.assert_active()?;
        if self.ready.get() {
            return Ok(());
        }
        self.ready.set(true);
        let entries: Vec<Rc<InstanceDirectoryEntry>> = self
            .entries
            .borrow()
            .iter()
            .map(|(_, e)| e.clone())
            .collect();
        for entry in entries {
            self.start_all(&entry);
        }
        Ok(())
    }

    /// Drops every entry and cancels its observations; a later
    /// [`ready`](Self::ready) restarts observations on newly inserted
    /// entries.
    pub fn reset(&self) {
        if self.disposed.get() {
            return;
        }
        self.ready.set(false);
        let entries: Vec<Rc<InstanceDirectoryEntry>> = self
            .entries
            .borrow()
            .iter()
            .map(|(_, e)| e.clone())
            .collect();
        for entry in entries {
            self.remove_internal(&entry);
        }
    }

    /// Registers an observer; already-ready directories start it for every
    /// live entry. The returned handle stops the observation.
    ///
    /// # Errors
    /// [`ChordError`] when the directory is disposed.
    pub fn observe(&self, handler: KeyedServiceHandler) -> Result<Unsubscribe, ChordError> {
        self.assert_active()?;
        let observer = Rc::new(Observer {
            handler,
            tasks: RefCell::new(Vec::new()),
            closed: Cell::new(false),
        });
        self.observers.borrow_mut().push(observer.clone());
        if self.ready.get() {
            let entries: Vec<Rc<InstanceDirectoryEntry>> = self
                .entries
                .borrow()
                .iter()
                .map(|(_, e)| e.clone())
                .collect();
            for entry in entries {
                self.start(&observer, &entry);
            }
        }
        let observers = Rc::clone(&self.observers);
        Ok(Box::new(move || {
            if observer.closed.get() {
                return;
            }
            observer.closed.set(true);
            for (_, controller) in observer.tasks.borrow_mut().drain(..) {
                controller.abort_without_reason();
            }
            observers
                .borrow_mut()
                .retain(|current| !Rc::ptr_eq(current, &observer));
        }))
    }

    /// Tears the directory down: every observation cancels, every entry
    /// deactivates.
    pub fn dispose(&self) {
        if self.disposed.get() {
            return;
        }
        self.disposed.set(true);
        for observer in self.observers.borrow().iter() {
            observer.closed.set(true);
            for (_, controller) in observer.tasks.borrow_mut().drain(..) {
                controller.abort_without_reason();
            }
        }
        self.observers.borrow_mut().clear();
        let entries: Vec<Rc<InstanceDirectoryEntry>> = self
            .entries
            .borrow()
            .iter()
            .map(|(_, e)| e.clone())
            .collect();
        for entry in entries {
            (entry.deactivate)();
        }
        self.entries.borrow_mut().clear();
    }

    fn remove_internal(&self, entry: &Rc<InstanceDirectoryEntry>) {
        self.entries
            .borrow_mut()
            .retain(|(_, current)| !Rc::ptr_eq(current, entry));
        (entry.deactivate)();
        for observer in self.observers.borrow().iter() {
            let at = observer
                .tasks
                .borrow()
                .iter()
                .position(|(task, _)| Rc::ptr_eq(task, entry));
            if let Some(at) = at {
                let (_, controller) = observer.tasks.borrow_mut().remove(at);
                controller.abort_without_reason();
            }
        }
    }

    fn start_all(&self, entry: &Rc<InstanceDirectoryEntry>) {
        for observer in self.observers.borrow().iter() {
            self.start(observer, entry);
        }
    }

    fn start(&self, observer: &Rc<Observer>, entry: &Rc<InstanceDirectoryEntry>) {
        if observer.closed.get()
            || observer
                .tasks
                .borrow()
                .iter()
                .any(|(task, _)| Rc::ptr_eq(task, entry))
        {
            return;
        }
        let (context, controller) = with_cancel(&crate::context::background_context());
        observer
            .tasks
            .borrow_mut()
            .push((entry.clone(), controller));
        let report = |error: ChordError| {
            if !context
                .abort_signal()
                .is_some_and(|signal| signal.aborted())
            {
                (self.report_error)(&error);
            }
        };
        let target = entry.service.clone();
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            (observer.handler)(target, context.clone());
        }));
        if let Err(panic) = result {
            report(ChordError::Message(crate::services::state::panic_message(
                &*panic,
            )));
        }
    }

    fn assert_active(&self) -> Result<(), ChordError> {
        if self.disposed.get() {
            return Err(ChordError::Message(
                "Keyed service directory is disposed".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::expect_used,
        clippy::redundant_clone,
        clippy::panic,
        reason = "the directory fixtures settle results and panics the case's own assertions pin"
    )]
    use super::*;
    use crate::handle::ServiceImplementation;

    fn entry(key: &str, generation: u64) -> Rc<InstanceDirectoryEntry> {
        Rc::new(InstanceDirectoryEntry {
            key: key.to_string(),
            generation,
            service: ServiceTarget::Local(Rc::new(ServiceImplementation::new())),
            deactivate: Rc::new(|| ()),
        })
    }

    fn directory() -> InstanceDirectory {
        InstanceDirectory::new(false, crate::handle::no_error_reporter())
    }

    #[test]
    fn directories_fence_duplicates_and_stale_generations() {
        let directory = directory();
        assert!(format!("{directory:?}").starts_with("InstanceDirectory { ready: false"));
        let live = entry("dialog", 1);
        assert!(format!("{live:?}").starts_with("InstanceDirectoryEntry { key: \"dialog\""));
        assert!(directory.insert(live.clone()).is_ok());
        let error = directory
            .insert(entry("dialog", 2))
            .expect_err("a live key rejects");
        assert!(error.to_string().contains("already has a live instance"));
        let error = directory
            .replace(entry("dialog", 1))
            .expect_err("a repeated generation rejects");
        assert!(error.to_string().contains("repeated a live generation"));

        // Replacing with a newer generation retires the previous entry.
        assert!(directory.replace(newer_entry()).is_ok());
        assert!(directory.get("dialog").is_some());
        // Removing an entry that is not the live one is a no-op.
        directory.remove(&live);
        assert!(directory.get("dialog").is_some());

        // Readiness is idempotent and observes only when armed.
        assert!(directory.ready().is_ok());
        assert!(directory.ready().is_ok());

        // Observations stop when the unsubscribe runs; a closed observer
        // ignores later starts.
        let handler: KeyedServiceHandler = Box::new(|_target, _context| ());
        let unsubscribe = directory.observe(handler).expect("the observation starts");
        unsubscribe();
        unsubscribe();
        // A closed observer ignores starts for later entries.
        let insertion = directory.insert(entry("late", 1));
        assert!(insertion.is_ok());
        // Reset drops entries; a later ready restarts observations.
        directory.reset();
        assert!(directory.get("dialog").is_none());
    }

    #[test]
    fn disposal_cancels_tasked_observations_and_reports_handler_panics() {
        // A handler panic surfaces through the directory's reporter while
        // the observation context is still live.
        let errors: Rc<RefCell<Vec<ChordError>>> = Rc::new(RefCell::new(Vec::new()));
        let reporter: ErrorReporter = {
            let errors = errors.clone();
            Rc::new(move |error| errors.borrow_mut().push(error.clone()))
        };
        let directory = InstanceDirectory::new(true, reporter);
        let panicking: KeyedServiceHandler = Box::new(|_target, _context| {
            panic!("handler failed");
        });
        let unsubscribe = directory
            .observe(panicking)
            .expect("the observation starts");
        let live = entry("dialog", 1);
        directory.insert(live).expect("the insertion lands");
        assert_eq!(errors.borrow().len(), 1);
        assert!(errors.borrow()[0].to_string().contains("handler failed"));

        // An aborted context swallows the report: the task is cancelled
        // before the panic would report again.
        unsubscribe();
        directory.reset();
        assert_eq!(errors.borrow().len(), 1);

        // Disposal cancels every task and deactivates every entry.
        let deactivated = Rc::new(Cell::new(0u32));
        let second = Rc::new(InstanceDirectoryEntry {
            key: "second".to_string(),
            generation: 1,
            service: ServiceTarget::Local(Rc::new(ServiceImplementation::new())),
            deactivate: Rc::new({
                let deactivated = deactivated.clone();
                move || deactivated.set(deactivated.get() + 1)
            }),
        });
        directory
            .insert(second)
            .expect("the second insertion lands");
        let watched: KeyedServiceHandler = Box::new(|_target, _context| ());
        let _watcher = directory.observe(watched).expect("the watcher starts");
        directory.dispose();
        directory.dispose();
        directory.reset();
        assert_eq!(deactivated.get(), 1);
    }

    fn newer_entry() -> Rc<InstanceDirectoryEntry> {
        entry("dialog", 2)
    }

    #[test]
    fn disposal_arms_settle_once() {
        let directory = directory();
        directory.dispose();
        directory.dispose();
        directory.reset();
        assert!(directory.insert(entry("late", 1)).is_err());
        assert!(directory.ready().is_err());
        let handler: KeyedServiceHandler = Box::new(|_target, _context| ());
        assert!(directory.observe(handler).is_err());
    }

    #[test]
    fn disposed_directories_reject_insertion() {
        let directory = directory();
        directory.dispose();
        let error = directory
            .insert(entry("late", 1))
            .expect_err("a disposed directory rejects insertion");
        assert!(error.to_string().contains("disposed"));
    }
}
