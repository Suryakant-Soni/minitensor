use crate::{Tensor,Result};
use crate::ops::reduce::reducer::Reducer;

/// invariants
/// pre processing 1. is keepdim = true, the shape dimension will be reduced to one otherwise the dimension will be removed
/// 2. accumulator acc is initialized as per reducer's initializstion policy
/// 3. for obtaining single element, all the non-reduced dimensions value will be fixed and the axis dimension value will be accumulated from 0 to n - 1
/// steps  : 1. after the new derived shape, allocate a storage/ zero tensor 
/// 2.loop all other dimensions in a nested way and then accumulate to get value on the input axis 
/// 3. use combine for accumulation
/// 4. return tensor whose shape is reduced and the values are accumulated
fn reduce_impl<R,T>(
    tensor : &Tensor,
    axis : usize,
    keepdim : bool,
) -> Result<Tensor<T>>
where 
    R : Reducer<T>,
    {
        let id = R::identity();
        let reduced_shape = reduce_shape(tensor.shape(),axis,keepdim);
        let mut out = Tensor::zeros(reduced_shape)?;
        let mut out_buf = out.as_mut_slice_contiguous_unique();
        

    };

    fn reduce_shape(shape: &[usize],axis : usize,keepdim:bool) -> Vec<usize>{
        let mut res = shape.to_vec(); 
        if keepdim{
           res[axis] = 1;
        }else{
            res.remove(axis);
        }
        res
    }