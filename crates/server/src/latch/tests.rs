use super::*;

#[test]
fn the_latch_debug_renders() {
    let latch: Latch<u32> = Latch::new();
    assert!(format!("{latch:?}").contains("Latch"));
    latch.settle(1);
    assert!(latch.is_settled());
}

#[tokio::test]
async fn the_latch_hands_late_waiters_the_settled_value() {
    let latch: Latch<u32> = Latch::new();
    latch.settle(7);
    assert_eq!(latch.wait().await, 7);
    // A second settle is the no-op, upstream's resolve-after-resolve.
    latch.settle(9);
    assert_eq!(latch.wait().await, 7);
}
