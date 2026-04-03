 use crate::error::{TensorError,Result};

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
    // it computes the number of element in the tensor by multiplying elements of shape,also checks multiplication overflow in computation
    pub(crate) fn compute_numel(shape: &[usize]) -> Result<usize> {
        let mut res = 1usize;
        for &elem in shape {
            res = res.checked_mul(elem).ok_or(TensorError::NumelOverflow)?;
        }
        Ok(res)
    }