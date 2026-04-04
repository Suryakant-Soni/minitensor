use crate::Result;
use crate::Tensor;
use crate::tensor::{indexing, shape};

// add method is stateless and it is called from tensor internal
// add method adds 2 tensors element wise, it call binary_op with the add operation function directive

pub(crate) fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(a, b, |x, y| x + y)
}

// sub method is stateless and it is called from tensor internal
// sub method subtracts b from a tensors element wise, it call binary_op with the sub operation function directive

pub(crate) fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(a, b, |x, y| x - y)
}

pub(crate) fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(a, b, |x, y| x * y)
}

// binary_op is a generic method for binary/element-wise operations, the operation function is passed separately
// this is stateless because no need to hold any state for the element wise operation
fn binary_op<F>(a: &Tensor, b: &Tensor, f: F) -> Result<Tensor>
where
    F: Fn(f32, f32) -> f32,
{
    // validate if shape of both input tensors is identical
    shape::validate_same_shape(a, b)?;
    //  create a new tensor out for result storing
    let mut out = Tensor::zeros(a.shape().to_vec())?;
    let out_buf = out.as_mut_slice_contiguous_unique()?;
    for i in 0..a.numel()? {
        // get the logical format on index
        // why unravel is important - unravel_index is needed here because a.get(&idx) and b.get(&idx) expect a logical N-D index, while your loop variable i is only a flat traversal position.
        let idx = indexing::convert_flat_position_to_logical_nd(i, a.shape());
        let av = a.get(&idx)?;
        let bv = b.get(&idx)?;
        out_buf[i] = f(av, bv);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_utils::assert_approx_eq;
    use crate::{MtError, TensorError};

    #[test]
    fn binary_op_successful() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let tensor_b = Tensor::from_vec(vec![1.1, 4.5, 0.8, 2.1], vec![2, 2]).unwrap();
        let tensor_c = binary_op(&tensor_a, &tensor_b, |x, y| x + y).unwrap();
        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 3.4);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 7.7);
        assert_eq!(tensor_c.numel().unwrap(), 4);
        assert_eq!(tensor_a.shape(), tensor_c.shape());
        assert_eq!(tensor_b.strides(), tensor_c.strides());
    }
    #[test]
    fn test_binary_op_shape_validation_failed() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let tensor_b = Tensor::from_vec(vec![1.1, 4.5, 0.8, 2.1, 3.2, 5.3], vec![3, 2]).unwrap();
        let tensor_c = binary_op(&tensor_a, &tensor_b, |x, y| x + y);
        assert!(matches!(
            tensor_c,
            Err(MtError::Tensor(TensorError::ShapeMismatch))
        ));
    }
}
