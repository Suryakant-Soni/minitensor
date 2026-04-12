use crate::Tensor;
use crate::{Result, TensorError};

pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let len = shape.len();
    let mut strides = vec![0usize; len];
    if len == 0 {
        return strides;
    }
    strides[len - 1] = 1;
    for i in (0..len - 1).rev() {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    strides
}
/// Computes the number of elements in a tensor shape.
///
/// Returns an error if the multiplication overflows.
pub(crate) fn compute_numel(shape: &[usize]) -> Result<usize> {
    let mut res = 1usize;
    for &elem in shape {
        res = res.checked_mul(elem).ok_or(TensorError::NumelOverflow)?;
    }
    Ok(res)
}

/// Validates that two tensors have the same shape.
pub(crate) fn validate_same_shape(a: &Tensor, b: &Tensor) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch.into());
    }
    Ok(())
}
