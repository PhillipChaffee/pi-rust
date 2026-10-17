use super::*;
use std::sync::Arc;

/// Recorded call of one cleanup: its marker and the session id passed.
type RecordedCall = (usize, Option<String>);

fn expect_ok(outcome: &Result<(), SessionCleanupError>) {
    #[allow(
        clippy::option_if_let_else,
        reason = "the match reads clearer than a map_or_else with an identity closure"
    )]
    match outcome {
        Ok(()) => (),
        Err(_) => unreachable!("no cleanup failed"),
    }
}

/// The registry is process-wide and integration tests run in parallel, so
/// the whole lifecycle asserts sequentially inside one test.
#[test]
fn registry_lifecycle_runs_cleanups_in_order_and_aggregates_failures() {
    let calls: Arc<Mutex<Vec<RecordedCall>>> = Arc::new(Mutex::new(Vec::new()));

    let first_calls = Arc::clone(&calls);
    let first = register_session_resource_cleanup(Box::new(move |session_id| {
        lock(&first_calls).push((1, session_id.map(str::to_owned)));
        Ok(())
    }));
    let second_calls = Arc::clone(&calls);
    let second = register_session_resource_cleanup(Box::new(move |session_id| {
        lock(&second_calls).push((2, session_id.map(str::to_owned)));
        Ok(())
    }));

    // Registration order drives the invocation order; every cleanup sees
    // the session id.
    expect_ok(&cleanup_session_resources(Some("session-1")));
    assert_eq!(
        lock(&calls).clone(),
        vec![
            (1, Some("session-1".to_string())),
            (2, Some("session-1".to_string())),
        ]
    );

    // Unregistering one leaves the others; the guard is consumed.
    second.unregister();
    lock(&calls).clear();
    expect_ok(&cleanup_session_resources(None));
    assert_eq!(lock(&calls).clone(), vec![(1, None)]);

    // Failures aggregate into one error; the other cleanups still run.
    let failing_calls = Arc::clone(&calls);
    let failing = register_session_resource_cleanup(Box::new(move |session_id| {
        lock(&failing_calls).push((3, session_id.map(str::to_owned)));
        Err(Box::<dyn StdError + Send + Sync>::from("boom"))
    }));
    lock(&calls).clear();
    lock(&calls).clear();
    let outcome = cleanup_session_resources(Some("session-2"));
    let Err(error) = outcome else {
        unreachable!("cleanup must aggregate failures")
    };
    assert_eq!(error.failures, vec!["boom".to_string()]);
    assert_eq!(
        error.to_string(),
        "Failed to cleanup session resources: boom"
    );
    assert_eq!(
        lock(&calls).clone(),
        vec![
            (1, Some("session-2".to_string())),
            (3, Some("session-2".to_string())),
        ]
    );

    first.unregister();
    failing.unregister();
    lock(&calls).clear();
    expect_ok(&cleanup_session_resources(None));
    assert_eq!(lock(&calls).clone(), vec![]);
    let remaining = lock(registry()).cleanups.len();
    assert_eq!(remaining, 0);
}
