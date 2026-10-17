use super::*;

#[test]
fn spells_the_internal_registry_and_slot_debug_surfaces() {
    let reporter = crate::handle::no_error_reporter();
    let registry = LocalKeyedServiceRegistry::new(Vec::new(), &reporter);
    assert!(format!("{registry:?}").starts_with("LocalKeyedServiceRegistry { disposed: false"));
    let slots = HostServiceSlots::new();
    assert!(format!("{slots:?}").starts_with("HostServiceSlots"));
}
