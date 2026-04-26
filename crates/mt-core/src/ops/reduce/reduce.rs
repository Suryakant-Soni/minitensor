use crate::ops::reduce::reducer::{MaxReducer, Reducer, SumReducer};
use crate::tensor::{indexing, validate};
use crate::{ Result, Tensor};

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
/// 2. should be a non-zero tensors
fn reduce_impl<R>(tensor: &Tensor, axis: usize, keepdim: bool) -> Result<Tensor>
where
    R: Reducer,
{
    let mut reduced_shape= tensor.shape().to_vec();
    if keepdim {
        reduced_shape[axis] = 1;
    }else{
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
        let mut acc = R::identity();
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

        for j in 0..axis_dim {
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

impl Tensor{
    pub fn sum(&self,axis: usize,keepdim: bool) -> Result<Tensor>{
        validate::non_empty_shape(self.shape())?;
        validate::validate_axis(self.shape(), axis)?;

        reduce_impl::<SumReducer>(self,axis,keepdim)
    }

    pub fn max(&self,axis: usize,keepdim: bool) -> Result<Tensor>{
        validate::non_empty_shape(self.shape())?;
        validate::validate_axis(self.shape(), axis)?;

        reduce_impl::<MaxReducer>(self,axis,keepdim)
    }
}