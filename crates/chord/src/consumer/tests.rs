#![allow(
    clippy::expect_used,
    reason = "the slot fixtures settle results the case's own assertions pin"
)]

use super::*;
use crate::handle::no_error_reporter;
use crate::services::loopback::create_loopback_service_transport;

fn transport() -> Rc<dyn RemoteServiceTransport> {
    let provider = crate::services::provider::RemoteServiceProvider::new(Vec::new())
        .expect("an empty catalogue has no duplicates");
    create_loopback_service_transport(&provider)
}

fn slot() -> MemberSlot {
    MemberSlot::new(
        "test.slot",
        "member",
        Rc::new(|_args: Vec<JsonValue>, _context: Context| boxed(ready_with(Ok(None)))),
        Rc::new(|| true),
        allow_access(),
        no_error_reporter(),
    )
}

#[test]
fn spells_the_slot_and_facade_debug_surfaces() {
    assert!(format!("{:?}", slot()).starts_with("MemberSlot { service_id: \"test.slot\""));
    let options = RemoteServiceBindingOptions {
        services: Vec::new(),
        transport: transport(),
        bound: true,
        on_error: no_error_reporter(),
        assert_access: None,
    };
    assert!(format!("{options:?}").starts_with("RemoteServiceBindingOptions {"));
    let is_active: Rc<dyn Fn() -> bool> = Rc::new(|| true);
    let facade = RemoteFacade::new(
        "test.facade",
        None,
        transport(),
        is_active,
        allow_access(),
        no_error_reporter(),
    );
    assert!(format!("{facade:?}").contains("test.facade"));
    assert!(slot().state_value().is_ok());
    assert!(facade.same_facade(&facade));
}

#[test]
fn slots_pin_member_kinds() {
    // A member described twice with different kinds rejects.
    let described = slot();
    described
        .set_description(ServiceMemberKind::State)
        .expect("the first kind lands");
    let error = described
        .set_description(ServiceMemberKind::Method)
        .expect_err("a kind change rejects");
    assert!(error.to_string().contains("changed kind"));

    // An expectation the description contradicts rejects.
    let expected = slot();
    expected
        .expect(ServiceMemberKind::Method)
        .expect("the first use pins");
    let error = expected
        .set_description(ServiceMemberKind::State)
        .expect_err("the description contradicts the expectation");
    assert!(error.to_string().contains("is state, not method"));

    // A member used as two different kinds rejects.
    let used = slot();
    used.expect(ServiceMemberKind::State)
        .expect("the state use pins");
    let error = used
        .expect(ServiceMemberKind::Method)
        .expect_err("the second use rejects");
    assert!(error.to_string().contains("used as two different kinds"));

    // A described method member rejects state reads.
    let method = slot();
    method
        .set_description(ServiceMemberKind::Method)
        .expect("the kind pins");
    let error = method
        .state_value()
        .expect_err("a method member rejects state reads");
    assert!(error.to_string().contains("is method, not state"));
}
