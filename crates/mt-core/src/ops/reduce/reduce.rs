use crate::ops::reduce::reducer::{MaxReducer, Reducer, SumReducer};
use crate::tensor::{indexing, validate};
use crate::{Result, Tensor};

/// invariants
/// 1. is keepdim = true, the shape dimension will be reduced to one otherwise the dimension will be removed
/// 2. accumulator acc is initialized as per reducer's initialization policy
/// 3. for obtaining single element, all the non-reduced dimensions value will be fixed and the axis dimension value will be accumulated from 0 to n - 1
/// steps  : 1. after the new derived shape, allocate a storage/ zero tensor
/// 2.loop all other dimensions in a nested way and then accumulate to get value on the input axis
/// 3. use combine for accumulation
/// 4. return tensor whose shape is reduced and the values are accumulated
///
/// assumptions: 1. the axis is already validated by the caller for tensor shape
/// 2. input tensor should be a non-zero tensor
fn reduce_impl<R>(tensor: &Tensor, axis: usize, keepdim: bool) -> Result<Tensor>
where
    R: Reducer,
{
    let mut reduced_shape = tensor.shape().to_vec();
    if keepdim {
        reduced_shape[axis] = 1;
    } else {
        reduced_shape.remove(axis);
    }
    let mut out = Tensor::zeros(reduced_shape.clone())?;
    let out_buf = out.as_mut_slice_contiguous_unique()?;
    let input_strides = tensor.strides();
    let input_rank = tensor.shape().len();
    let out_numel = reduced_shape.iter().product();
    let axis_stride = input_strides[axis];
    let axis_dim = tensor.shape()[axis];
    for i in 0..out_numel {
        // get the logical index from out shape, this will also be logical index for non-axis input elements
        // TODO : remove convert_flat_position_to_logical_nd method in every loop pass and move it to incremental index calculation
        let out_index = indexing::convert_flat_position_to_logical_nd(i, &reduced_shape);
        let mut base = 0;
        // calculate base value for calculating flat index of input tensor per non-axis increment
        for d in 0..input_rank {
            if d == axis {
                continue;
            }
            let out_d = if keepdim {
                d
            } else if d < axis {
                d
            } else {
                d - 1
            };
            base += input_strides[d] * out_index[out_d];
        }

        let (mut acc, start_j) = if R::use_first_elem_as_init() {
            unsafe { (tensor.get_with_flat_unchecked(base), 1) }
        } else {
            (R::identity(), 0)
        };
        for j in start_j..axis_dim {
            // unsafe invariants
            // 1. tensor is contiguous
            // 2. base + j * input_strides[axis] < tensor.numel()
            // 3. out_index is within reduced_shape
            // 4. input_strides are valide for tensor's shape
            unsafe {
                let sv = tensor.get_with_flat_unchecked(base + j * axis_stride);
                acc = R::combine(acc, sv);
            }
        }
        out_buf[i] = acc;
    }
    Ok(out)
}

impl Tensor {
    pub fn sum(&self, axis: usize, keepdim: bool) -> Result<Tensor> {
        validate::non_empty_shape(self.shape())?;
        validate::validate_axis(self.shape(), axis)?;

        reduce_impl::<SumReducer>(self, axis, keepdim)
    }

    pub fn max(&self, axis: usize, keepdim: bool) -> Result<Tensor> {
        validate::non_empty_shape(self.shape())?;
        validate::validate_axis(self.shape(), axis)?;

        reduce_impl::<MaxReducer>(self, axis, keepdim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_utils::assert_approx_eq;
    use crate::{MtError, OpError, TensorError};

    // test with axis = 0,1,2 in any 2,3,5
    // test for invalid axis
    // test for zero shape
    // test for keepdim true/false
    // test for max, and negative values

    // any critical invariants further left to test

    #[test]
    fn reduce_sum_three_dim_tensor_axis0() {
        let data: Vec<f32> = (0..30).map(|value| value as f32).collect();
        let tensor = Tensor::from_vec(data, vec![2, 3, 5]).unwrap();
        let res_tensor_keepdim = tensor.sum(0, true).unwrap();
        let res_tensor = tensor.sum(0, false).unwrap();

        assert_eq!(&[1, 3, 5], res_tensor_keepdim.shape());
        assert_eq!(&[3, 5], res_tensor.shape());

        let expected = [
            15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0, 35.0, 37.0, 39.0, 41.0,
            43.0,
        ];
        unsafe {
            for (idx, value) in expected.iter().enumerate() {
                assert_approx_eq(res_tensor.get_with_flat_unchecked(idx), *value);
            }
        }
    }

    #[test]
    fn reduce_sum_three_dim_tensor_axis1() {
        let data: Vec<f32> = (0..30).map(|value| value as f32).collect();
        let tensor = Tensor::from_vec(data, vec![2, 3, 5]).unwrap();
        // now reduce the tensor on axis 1
        let res_tensor_keepdim = tensor.sum(1, true).unwrap();
        let res_tensor = tensor.sum(1, false).unwrap();

        assert_eq!(&[2, 1, 5], res_tensor_keepdim.shape());
        assert_eq!(&[2, 5], res_tensor.shape());

        let expected = [15.0, 18.0, 21.0, 24.0, 27.0, 60.0, 63.0, 66.0, 69.0, 72.0];
        unsafe {
            for (idx, value) in expected.iter().enumerate() {
                assert_approx_eq(res_tensor.get_with_flat_unchecked(idx), *value);
            }
        }
    }

    #[test]
    fn reduce_sum_three_dim_tensor_axis2() {
        let data: Vec<f32> = (0..30).map(|value| value as f32).collect();
        let tensor = Tensor::from_vec(data, vec![2, 3, 5]).unwrap();
        let res_tensor_keepdim = tensor.sum(2, true).unwrap();
        let res_tensor = tensor.sum(2, false).unwrap();

        assert_eq!(&[2, 3, 1], res_tensor_keepdim.shape());
        assert_eq!(&[2, 3], res_tensor.shape());

        let expected = [10.0, 35.0, 60.0, 85.0, 110.0, 135.0];
        unsafe {
            for (idx, value) in expected.iter().enumerate() {
                assert_approx_eq(res_tensor.get_with_flat_unchecked(idx), *value);
            }
        }
    }

    #[test]
    fn reduce_invalid_axis_returns_error() {
        let data: Vec<f32> = (0..6).map(|value| value as f32).collect();
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        assert!(matches!(
            tensor.sum(2, false),
            Err(MtError::Operations(OpError::AxisIndexInvalid))
        ));
        assert!(matches!(
            tensor.max(2, true),
            Err(MtError::Operations(OpError::AxisIndexInvalid))
        ));
    }

    #[test]
    fn reduce_zero_sized_tensor_returns_error() {
        let tensor = Tensor::zeros(vec![2, 0, 5]).unwrap();

        assert!(matches!(
            tensor.sum(1, false),
            Err(MtError::Tensor(TensorError::ZeroSizedTensor))
        ));
        assert!(matches!(
            tensor.max(1, true),
            Err(MtError::Tensor(TensorError::ZeroSizedTensor))
        ));
    }

    #[test]
    fn reduce_max_three_dim_tensor_with_negative_values() {
        let data = vec![
            -1.0, -5.0, 3.0, 4.5, 0.0, 2.0, -2.0, 8.0, -1.0, 7.0, 1.0, -3.0, 6.0, 5.0, -4.0,
            -10.0, 11.0, -12.0, 13.0, -14.0, 9.0, -8.0, 7.0, -6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
            0.0,
        ];
        let tensor = Tensor::from_vec(data, vec![2, 3, 5]).unwrap();
        let res_tensor_keepdim = tensor.max(1, true).unwrap();
        let res_tensor = tensor.max(1, false).unwrap();

        assert_eq!(&[2, 1, 5], res_tensor_keepdim.shape());
        assert_eq!(&[2, 5], res_tensor.shape());

        let expected = [2.0, -2.0, 8.0, 5.0, 7.0, 9.0, 11.0, 7.0, 13.0, 5.0];
        unsafe {
            for (idx, value) in expected.iter().enumerate() {
                assert_approx_eq(res_tensor.get_with_flat_unchecked(idx), *value);
            }
        }
    }
}
