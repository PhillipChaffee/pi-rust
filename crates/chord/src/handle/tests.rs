use super::*;
use crate::context::background_context;

fn implementation_with_state_and_value() -> Rc<ServiceImplementation> {
    let state = MutableReplicatedState::new(JsonValue::Null);
    let mut implementation = ServiceImplementation::new();
    implementation.method(
        "read",
        sync_method(|_args: Vec<JsonValue>, _context: &Context| Ok(None)),
    );
    implementation.state("state", state);
    implementation.value("value", Rc::new(7u32));
    Rc::new(implementation)
}

#[test]
fn spells_member_kinds_and_slot_surfaces() {
    let implementation = implementation_with_state_and_value();
    assert_eq!(
        format!("{:?}", implementation.member("read")),
        "Some(Method(_))"
    );
    assert!(
        format!("{:?}", implementation.member("state"))
            .starts_with("Some(State(MutableReplicatedState")
    );
    assert_eq!(
        format!("{:?}", implementation.member("value")),
        "Some(Value(_))"
    );
    let slot = ServiceSlot::new("test.member-kinds");
    assert!(!slot.is_bound());
    assert!(format!("{slot:?}").contains("test.member-kinds"));
    assert!(format!("{:?}", slot.view(allow_access())).contains("test.member-kinds"));
}

#[test]
fn validates_remote_implementations_by_shape() {
    let empty = ServiceImplementation::new();
    let Err(error) = validate_remote_implementation("test.remote", &empty) else {
        unreachable!("an empty implementation rejects")
    };
    assert!(error.to_string().contains("has no members"));
}

#[test]
fn local_views_surface_state_members_and_rejections() {
    let slot = ServiceSlot::new("test.local-surface");
    slot.bind(ServiceTarget::Local(implementation_with_state_and_value()));
    let view = slot.view(allow_access());
    let Ok(state) = view.state("state") else {
        unreachable!("the state member binds")
    };
    assert!(format!("{state:?}").starts_with("StateMemberView"));
    let Ok(value) = state.value() else {
        unreachable!("the state value reads")
    };
    assert_eq!(value, Some(JsonValue::Null));
    let Err(error) = view.state("read") else {
        unreachable!("a method member is not state")
    };
    assert!(error.to_string().contains("not replicated state"));
    let Err(error) = view.state("absent") else {
        unreachable!("an absent member disconnects")
    };
    assert!(error.to_string().contains("disconnected"));
    // A value member reads through its data; a non-value member rejects.
    let answer: u32 = view
        .with_value("value", |data: &dyn Any| {
            data.downcast_ref::<u32>().copied().unwrap_or_default()
        })
        .unwrap_or_default();
    assert_eq!(answer, 7);
    let Err(read_error): Result<u32, ChordError> =
        view.with_value("read", |_never: &dyn Any| unreachable!())
    else {
        unreachable!("a method member is not a value")
    };
    assert!(read_error.to_string().contains("not a value"));
    let Err(absent_error): Result<u32, ChordError> =
        view.with_value("absent", |_never: &dyn Any| unreachable!())
    else {
        unreachable!("an absent member disconnects")
    };
    assert!(absent_error.to_string().contains("disconnected"));
    // A value member subscribes to its source state through the same
    // gate.
    let listener: crate::services::state::ValueListener = Rc::new(|_, _, _| ());
    assert!(state.subscribe(listener).is_ok());
    slot.unbind();
    let Err(error): Result<u32, ChordError> =
        view.with_value("value", |_never: &dyn Any| unreachable!())
    else {
        unreachable!("an unbound slot disconnects")
    };
    assert!(error.to_string().contains("disconnected"));
}

#[test]
fn disconnected_views_reject_invocations() {
    let slot = ServiceSlot::new("test.disconnected");
    let view = slot.view(allow_access());
    let Some(Err(error)) =
        crate::future::drive_once(view.call("read", vec![], background_context())).ok()
    else {
        unreachable!("a disconnected call settles immediately")
    };
    assert!(error.to_string().contains("disconnected"));
}

#[test]
fn non_callable_members_reject_invocations() {
    let slot = ServiceSlot::new("test.not-callable");
    slot.bind(ServiceTarget::Local(implementation_with_state_and_value()));
    let view = slot.view(allow_access());
    let Some(Err(error)) =
        crate::future::drive_once(view.call("state", vec![], background_context())).ok()
    else {
        unreachable!("a local rejection settles immediately")
    };
    assert!(error.to_string().contains("not callable"));
}
