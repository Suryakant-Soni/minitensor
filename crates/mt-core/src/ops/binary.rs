use crate::Result;
use crate::Tensor;
use crate::tensor::{indexing, shape};
use crate::transforms::broadcast;

/// Adds two tensors element-wise.
pub(crate) fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(a, b, |x, y| x + y)
}

/// Subtracts tensor `b` from tensor `a` element-wise.
pub(crate) fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(a, b, |x, y| x - y)
}

pub(crate) fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(a, b, |x, y| x * y)
}

/// Applies a generic element-wise binary operation without traversal state reuse.
fn binary_op_old<F>(a: &Tensor, b: &Tensor, f: F) -> Result<Tensor>
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

/// Applies an element-wise binary operation while reusing traversal state.
///
/// Tracks the current logical index and flat indexes for both operands to compute
/// the next flat positions without rebuilding the full logical index each time.
fn binary_op<F>(a: &Tensor, b: &Tensor, f: F) -> Result<Tensor>
where
    F: Fn(f32, f32) -> f32,
{
    // validate if shape of both input tensors is identical
    let binary_md = broadcast::compute_broadcast_metadata(a, b)?;
    //  create a new tensor out for result storing
    let mut out = Tensor::zeros(binary_md.result_shape().to_vec())?;
    let out_buf = out.as_mut_slice_contiguous_unique()?;
    // flat index
    let mut a_flat = binary_md.a().offset();
    let mut b_flat = binary_md.b().offset();
    // logical index
    let mut l_idx = vec![0usize; binary_md.result_shape().len()];
    let len = shape::compute_numel(binary_md.result_shape())?;
    for i in 0..len {
        // intital traversal - will always start with zero
        // logical index - [0,0]
        // initial flat index - 0
        unsafe {
            let av = a.get_with_flat_unchecked(a_flat);
            let bv = b.get_with_flat_unchecked(b_flat);
            out_buf[i] = f(av, bv);
        }
        // no need to compute next flat index for last index
        if i + 1 < len {
            indexing::compute_index_on_increment(
                &mut l_idx,
                &mut a_flat,
                &mut b_flat,
                binary_md.result_shape(),
                binary_md.a().strides(),
                binary_md.b().strides(),
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_utils::assert_approx_eq;
    use crate::{MtError, OpError};

    #[test]
    fn binary_op_successful() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let tensor_b = Tensor::from_vec(vec![1.1, 4.5, 0.8, 2.1], vec![2, 2]).unwrap();
        let tensor_c = binary_op(&tensor_a, &tensor_b, |x, y| x + y).unwrap();
        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 3.4);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 7.7);
        assert_eq!(tensor_c.numel().unwrap(), 4);
        assert_eq!(tensor_a.shape(), tensor_c.shape());
        assert_eq!(tensor_c.strides(), &[2, 1]);
    }
    #[test]
    fn binary_op_a_broadcasted_successful() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let tensor_b = Tensor::from_vec(vec![0.8, 2.1], vec![2]).unwrap();
        let tensor_c = binary_op(&tensor_a, &tensor_b, |x, y| x + y).unwrap();
        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 3.1);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 5.3);
        assert_approx_eq(tensor_c.get(&[1, 0]).unwrap(), 6.1);
        assert_approx_eq(tensor_c.get(&[1, 1]).unwrap(), 6.8);
        assert_eq!(tensor_c.numel().unwrap(), 4);
        assert_eq!(tensor_a.shape(), tensor_c.shape());
    }
    #[test]
    fn binary_op_b_broadcasted_successful() {
        let tensor_b = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let tensor_a = Tensor::from_vec(vec![0.8, 2.1], vec![2]).unwrap();
        let tensor_c = binary_op(&tensor_a, &tensor_b, |x, y| x + y).unwrap();
        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 3.1);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 5.3);
        assert_approx_eq(tensor_c.get(&[1, 0]).unwrap(), 6.1);
        assert_approx_eq(tensor_c.get(&[1, 1]).unwrap(), 6.8);
        assert_eq!(tensor_c.numel().unwrap(), 4);
        assert_eq!(tensor_b.shape(), tensor_c.shape());
    }
    #[test]
    fn broadcasting_unit_tensor_2by2() {
        let tensor_b = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let tensor_a = Tensor::from_vec(vec![0.8], vec![1]).unwrap();
        let tensor_c = binary_op(&tensor_a, &tensor_b, |x, y| x + y).unwrap();
        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 3.1);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 4.0);
        assert_approx_eq(tensor_c.get(&[1, 0]).unwrap(), 6.1);
        assert_approx_eq(tensor_c.get(&[1, 1]).unwrap(), 5.5);
        assert_eq!(tensor_c.numel().unwrap(), 4);
        assert_eq!(tensor_b.shape(), tensor_c.shape());
    }
    #[test]
    fn single_elements_both_sides_1by1() {
        let tensor_b = Tensor::from_vec(vec![2.3], vec![1]).unwrap();
        let tensor_a = Tensor::from_vec(vec![0.8], vec![1]).unwrap();
        let tensor_c = binary_op(&tensor_a, &tensor_b, |x, y| x + y).unwrap();
        assert_approx_eq(tensor_c.get(&[0]).unwrap(), 3.1);
        assert_eq!(tensor_c.numel().unwrap(), 1);
        assert_eq!(tensor_b.shape(), tensor_c.shape());
    }
    #[test]
    fn one_element_expansion_on_multiple_dimensions() {
        let tensor_b = Tensor::from_vec(
            vec![2.3, 3.2, 5.3, 4.7, 1.5, 3.2, 4.2, 1.5, 1.3, 2.6, 1.1, 1.3],
            vec![2, 3, 2],
        )
        .unwrap();
        let tensor_a = Tensor::from_vec(vec![1.1], vec![1, 1, 1]).unwrap();
        let tensor_c = binary_op(&tensor_a, &tensor_b, |x, y| x + y).unwrap();
        assert_approx_eq(tensor_c.get(&[0, 0, 0]).unwrap(), 3.4);
        assert_approx_eq(tensor_c.get(&[0, 1, 1]).unwrap(), 5.8);
        assert_approx_eq(tensor_c.get(&[1, 0, 0]).unwrap(), 5.3);
        assert_approx_eq(tensor_c.get(&[1, 2, 1]).unwrap(), 2.4);
        assert_eq!(tensor_c.numel().unwrap(), 12);
        assert_eq!(tensor_b.shape(), tensor_c.shape());
    }
    #[test]
    fn binary_op_both_tensors_broadcasted_successful() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7, 1.4, 0.9], vec![2, 3]).unwrap();
        let tensor_b = Tensor::from_vec(vec![0.8, 2.1, 4.3, 2.2, 1.4, 0.9], vec![2, 1, 3]).unwrap();
        let tensor_c = binary_op(&tensor_a, &tensor_b, |x, y| x + y).unwrap();
        assert_approx_eq(tensor_c.get(&[0, 0, 0]).unwrap(), 3.1);
        assert_approx_eq(tensor_c.get(&[0, 0, 1]).unwrap(), 5.3);
        assert_approx_eq(tensor_c.get(&[0, 0, 2]).unwrap(), 9.6);
        assert_approx_eq(tensor_c.get(&[0, 1, 0]).unwrap(), 5.5);
        assert_approx_eq(tensor_c.get(&[0, 1, 1]).unwrap(), 3.5);
        assert_approx_eq(tensor_c.get(&[0, 1, 2]).unwrap(), 5.2);
        assert_approx_eq(tensor_c.get(&[1, 0, 0]).unwrap(), 4.5);
        assert_approx_eq(tensor_c.get(&[1, 0, 1]).unwrap(), 4.6);
        assert_approx_eq(tensor_c.get(&[1, 0, 2]).unwrap(), 6.2);
        assert_approx_eq(tensor_c.get(&[1, 1, 0]).unwrap(), 6.9);
        assert_approx_eq(tensor_c.get(&[1, 1, 1]).unwrap(), 2.8);
        assert_approx_eq(tensor_c.get(&[1, 1, 2]).unwrap(), 1.8);
        assert_eq!(tensor_c.numel().unwrap(), 12);
        assert_eq!(&[2, 2, 3], tensor_c.shape());
    }
    #[test]
    fn test_binary_op_shape_broadcasted_incompatible() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let tensor_b = Tensor::from_vec(vec![1.1, 4.5, 0.8, 2.1, 3.2, 5.3], vec![3, 2]).unwrap();
        let tensor_c = binary_op(&tensor_a, &tensor_b, |x, y| x + y);
        assert!(matches!(
            tensor_c,
            Err(MtError::Operations(OpError::BroadcastIncompatible))
        ));
    }
}
