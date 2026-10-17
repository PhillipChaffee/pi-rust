use super::*;
use crate::test_support::js;
use std::cell::RefCell;

#[test]
fn spells_the_state_surfaces() {
    let source = MutableReplicatedState::new(js("initial"));
    assert!(format!("{source:?}").contains("initial"));
    let replica = ReplicatedStateReplica::new(no_error_reporter());
    assert!(format!("{replica:?}").contains("sequence"));
}

#[test]
fn replicas_hydrate_a_base_batch_then_reject_out_of_order_updates() {
    let errors: Rc<RefCell<Vec<ChordError>>> = Rc::new(RefCell::new(Vec::new()));
    let report: ErrorReporter = {
        let errors = errors.clone();
        Rc::new(move |error: &ChordError| {
            errors.borrow_mut().push(error.clone());
        })
    };
    let replica = ReplicatedStateReplica::new(report);
    let context = background_context();
    // Updates before hydration reject.
    let Err(error) = replica.update(1, &[Op::Replace(js("one"))], &context) else {
        unreachable!("an unhydrated update rejects")
    };
    assert!(error.to_string().contains("before hydration"));
    // A non-base batch rejects hydration.
    let Err(error) = replica.hydrate(
        0,
        &[Op::Set {
            path: vec![crate::delta::Seg::Key("a".to_string())],
            value: js("one"),
        }],
        &context,
    ) else {
        unreachable!("a non-base batch rejects")
    };
    assert!(error.to_string().contains("not a base operation batch"));
    // Hydration publishes to subscribers, and the panic of one listener
    // is reported rather than propagated.
    #[allow(
        clippy::panic,
        reason = "the panicked listener is the case's subject; the delivery reports it"
    )]
    let _unsubscribe = replica.subscribe(Rc::new({
        move |_value: &JsonValue, _context: &Context, _delivery: &ReplicatedStateDelivery| {
            panic!("listener boom");
        }
    }));
    assert!(
        replica
            .hydrate(0, &[Op::Replace(js("hydrated"))], &context)
            .is_ok()
    );
    let reported = errors.borrow().clone();
    assert!(
        reported
            .iter()
            .any(|error| error.to_string().contains("listener boom")),
        "the listener panic is reported: {reported:?}"
    );
}

#[test]
fn panic_payloads_spell_their_message() {
    let payload = panic_message(&"str boom");
    assert_eq!(payload, "str boom");
    let payload = panic_message(&String::from("string boom"));
    assert_eq!(payload, "string boom");
    let payload = panic_message(&7u32);
    assert_eq!(payload, String::new());
    let reporter = default_reporter();
    reporter(&ChordError::Message("dropped".to_string()));
}
