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
