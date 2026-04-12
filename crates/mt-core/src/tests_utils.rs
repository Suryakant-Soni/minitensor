pub(crate) fn assert_approx_eq(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-6);
}
