use crate::Tensor;
use crate::error::*;

// forms the logical N-D coordinate for a row-major traversal position, based only on shape
fn unravel_index(mut index: usize, shape: &[usize]) -> Vec<usize> {
    // initiate a vector of the same length as shape
    let mut idx = vec![0; shape.len()];
    let numel = Tensor::compute_numel(shape).expect("input shape is invalid");
    assert!(index < numel);
    // now loop the shape to find the logical indices at every dimenstion with the help of shape
    // we will start from the reverse of loop from the units place of the lowest dimension(fastest changing dimension)
    for i in (0..shape.len()).rev() {
        // dimension index will be the remaining after the shape dimension length's multiple
        idx[i] = index % shape[i];
        // updated index because it has to view from one dimesion up
        index /= shape[i];
    }
    idx
}

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
    validate_same_shape(a, b)?;
    //  create a new vector out for result storing
    let mut out = Tensor::zeros(a.shape().to_vec())?;
    let out_buf = out.as_mut_slice_contiguous_unique()?;
    for i in 0..a.numel()? {
        // get the logical format on index
        let idx = unravel_index(i, a.shape());
        let av = a.get(&idx)?;
        let bv = b.get(&idx)?;
        out_buf[i] = f(av, bv);
    }
    Ok(out)
}

// validate if the shape is same for the 2 tensors
fn validate_same_shape(a: &Tensor, b: &Tensor) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_op_successful() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let tensor_b = Tensor::from_vec(vec![1.1, 4.5, 0.8, 2.1], vec![2, 2]).unwrap();
        let tensor_c = binary_op(&tensor_a, &tensor_b, |x, y| x + y).unwrap();
        assert_eq!(tensor_c.get(&[0, 0]).unwrap(), 3.4);
        assert_eq!(tensor_c.get(&[0, 1]).unwrap(), 7.7);
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

    #[test]
    fn unravel_index_working_as_expected(){
        assert_eq!(unravel_index(0,&[2,2]),vec![0,0]);        
        assert_eq!(unravel_index(2,&[2,2]),vec![1,0]);        
    }

    #[test]
    #[should_panic]
    fn unravel_index_panics_with_index_greater_than_numel(){
        assert_eq!(unravel_index(5,&[2,2]),vec![1,1]);
    }
}
