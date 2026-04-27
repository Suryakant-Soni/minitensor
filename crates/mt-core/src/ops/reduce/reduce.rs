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

    fn assert_tensor_flat_values(tensor: &Tensor, expected: &[f32]) {
        assert_eq!(tensor.numel().unwrap(), expected.len());
        unsafe {
            for (idx, value) in expected.iter().enumerate() {
                assert_approx_eq(tensor.get_with_flat_unchecked(idx), *value);
            }
        }
    }

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

    #[test]
    fn axis_size_is_one_no_diff_after_reduction() {
        // data should be equivalent of shape [2,1,5]
        // and reduce it at axis = 1
        // thei input andoutput shape should be same
        // infact input values andoutput values all should be same in both tensors
        let data: Vec<f32> = (0..10).map(|value| value as f32).collect();
        let tensor = Tensor::from_vec(data, vec![2, 1, 5]).unwrap();
        let reduced = tensor.sum(1, true).unwrap();

        assert_eq!(tensor.shape(), reduced.shape());

        unsafe {
            for idx in 0..10 {
                assert_approx_eq(
                    tensor.get_with_flat_unchecked(idx),
                    reduced.get_with_flat_unchecked(idx),
                );
            }
        }
    }

    #[test]
    fn reduction_of_unit_tensor() {
        // declare a tensor with shape something like [3]
        // reduce it at axis0
        // compare the value and assert, assert the new shape as well
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let reduced = tensor.sum(0, false).unwrap();

        assert_eq!(&[] as &[usize], reduced.shape());
        unsafe {
            assert_approx_eq(reduced.get_with_flat_unchecked(0), 6.0);
        }
    }

    #[test]
    fn max_for_single_element() {
        // declare a tensor with only one value that is negative
        // add max to it and find what is the outcome
        let tensor = Tensor::from_vec(vec![-7.5], vec![1]).unwrap();
        let reduced = tensor.max(0, false).unwrap();
        let reduced_keepdim = tensor.max(0, true).unwrap();

        assert_eq!(&[] as &[usize], reduced.shape());
        assert_eq!(&[1], reduced_keepdim.shape());
        assert_tensor_flat_values(&reduced, &[-7.5]);
        assert_tensor_flat_values(&reduced_keepdim, &[-7.5]);
    }

    #[test]
    fn large_axis_size_unit_outer_dimension_size() {
        // test for tensor with shape 1,1,1000 and reduce it on axis = 2
        // use the inputs which are floating point numbers
        let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.25 + 0.5).collect();
        let expected_sum: f32 = data.iter().sum();
        let tensor = Tensor::from_vec(data, vec![1, 1, 1000]).unwrap();
        let reduced = tensor.sum(2, false).unwrap();
        let reduced_keepdim = tensor.sum(2, true).unwrap();

        assert_eq!(&[1, 1], reduced.shape());
        assert_eq!(&[1, 1, 1], reduced_keepdim.shape());
        assert_tensor_flat_values(&reduced, &[expected_sum]);
        assert_tensor_flat_values(&reduced_keepdim, &[expected_sum]);
    }

    #[test]
    fn wide_shape_and_axis_1_reduction() {
        // input tensor shape is [4,2,3,4,5,6]
        // input values should be decimal
        // and reduce it on axis = 1
        // assert if all elements or output tensor are correctly reduced
        let shape = vec![4, 2, 3, 4, 5, 6];
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = (0..numel).map(|i| i as f32 + 0.125).collect();
        let tensor = Tensor::from_vec(data, shape).unwrap();
        let reduced = tensor.sum(1, false).unwrap();
        let reduced_keepdim = tensor.sum(1, true).unwrap();

        assert_eq!(&[4, 3, 4, 5, 6], reduced.shape());
        assert_eq!(&[4, 1, 3, 4, 5, 6], reduced_keepdim.shape());

        let trailing_block = 3 * 4 * 5 * 6;
        let expected: Vec<f32> = (0..(4 * 3 * 4 * 5 * 6))
            .map(|i| {
                let outer = i / trailing_block;
                let inner = i % trailing_block;
                let base = outer * 2 * trailing_block + inner;
                (base as f32 + 0.125) + ((base + trailing_block) as f32 + 0.125)
            })
            .collect();
        assert_tensor_flat_values(&reduced, &expected);
        assert_tensor_flat_values(&reduced_keepdim, &expected);
    }

    #[test]
    fn multiple_cascaded_reductions() {
        // reduce tensor a to becomes tensor b on one axis ,
        // reduce tensor b to become tensor c on another axis,
        // verify results and shape of intermediate as well as final tensors
        let data: Vec<f32> = (0..24).map(|i| i as f32 + 1.0).collect();
        let tensor_a = Tensor::from_vec(data, vec![2, 3, 4]).unwrap();
        let tensor_b = tensor_a.sum(2, false).unwrap();
        let tensor_c = tensor_b.sum(1, false).unwrap();

        assert_eq!(&[2, 3], tensor_b.shape());
        assert_eq!(&[2], tensor_c.shape());
        assert_tensor_flat_values(&tensor_b, &[10.0, 26.0, 42.0, 58.0, 74.0, 90.0]);
        assert_tensor_flat_values(&tensor_c, &[78.0, 222.0]);
    }

    #[test]
    fn reduce_tensor_with_unit_dimesions() {
        // take 2 separate tensors with shapes 1,3 and 3,1
        // reduce both of them at axis 0 and axis 1,
        // for these 4 cases, verify resulting shape and resulting reduced values
        let row_tensor = Tensor::from_vec(vec![1.5, 2.5, 3.5], vec![1, 3]).unwrap();
        let col_tensor = Tensor::from_vec(vec![1.5, 2.5, 3.5], vec![3, 1]).unwrap();

        let row_axis0 = row_tensor.sum(0, false).unwrap();
        let row_axis1 = row_tensor.sum(1, false).unwrap();
        let col_axis0 = col_tensor.sum(0, false).unwrap();
        let col_axis1 = col_tensor.sum(1, false).unwrap();

        assert_eq!(&[3], row_axis0.shape());
        assert_eq!(&[1], row_axis1.shape());
        assert_eq!(&[1], col_axis0.shape());
        assert_eq!(&[3], col_axis1.shape());

        assert_tensor_flat_values(&row_axis0, &[1.5, 2.5, 3.5]);
        assert_tensor_flat_values(&row_axis1, &[7.5]);
        assert_tensor_flat_values(&col_axis0, &[7.5]);
        assert_tensor_flat_values(&col_axis1, &[1.5, 2.5, 3.5]);
    }
}
