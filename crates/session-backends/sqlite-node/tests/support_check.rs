//! Throwaway compile check for the support module. Deleted with the suites.
mod support;
#[test]
fn compiles() {
    let (factory, _entered, _release) = support::GatedOpenExistingFactory::new();
    let _ = support::CloseTrackingFactory::new();
    let _ = factory;
    assert_eq!(support::NOW, 1_700_000_000_000);
}
