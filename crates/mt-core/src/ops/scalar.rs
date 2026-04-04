use crate::Result;
use crate::Tensor;
use crate::tensor::indexing;

// element wise scalar operation
fn scalar_op<F>(a: &Tensor, b: f32, f: F) -> Result<Tensor>
where
    F: Fn(f32, f32) -> f32,
{
    // create a new tensor out for storing result
    let mut out = Tensor::zeros(a.shape().to_vec())?;
    let out_buf = out.as_mut_slice_contiguous_unique()?;
    for i in 0..a.numel()? {
        let idx = indexing::convert_flat_position_to_logical_nd(i, a.shape());
        let av = a.get(&idx)?;
        out_buf[i] = f(av, b);
    }
    Ok(out)
}

// add method for scalar operation , second top api , after user facing tensor add api is called
pub(crate) fn add(a: &Tensor, b: f32) -> Result<Tensor> {
    scalar_op(a, b, |x, y| x + y)
}

pub(crate) fn sub(a: &Tensor, b: f32) -> Result<Tensor> {
    scalar_op(a, b, |x, y| x - y)
}

pub(crate) fn mul(a: &Tensor, b: f32) -> Result<Tensor> {
    scalar_op(a, b, |x, y| x * y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_utils::assert_approx_eq;
    #[test]
    fn scalar_op_successful() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let rhs: f32 = 1.1;
        let tensor_c = scalar_op(&tensor_a, rhs, |x, y| x + y).unwrap();
        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 3.4);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 4.3);
        assert_eq!(tensor_c.numel().unwrap(), 4);
        assert_eq!(tensor_a.shape(), tensor_c.shape());
    }
    #[test]
    fn scalar_op_sub_successful() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let rhs: f32 = 1.1;

        let tensor_c = scalar_op(&tensor_a, rhs, |x, y| x - y).unwrap();

        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 1.2);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 2.1);
        assert_approx_eq(tensor_c.get(&[1, 0]).unwrap(), 4.2);
        assert_approx_eq(tensor_c.get(&[1, 1]).unwrap(), 3.6);

        assert_eq!(tensor_c.numel().unwrap(), 4);
        assert_eq!(tensor_a.shape(), tensor_c.shape());
    }

    #[test]
    fn scalar_op_mul_successful() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let rhs: f32 = 2.0;

        let tensor_c = scalar_op(&tensor_a, rhs, |x, y| x * y).unwrap();

        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 4.6);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 6.4);
        assert_approx_eq(tensor_c.get(&[1, 0]).unwrap(), 10.6);
        assert_approx_eq(tensor_c.get(&[1, 1]).unwrap(), 9.4);

        assert_eq!(tensor_c.numel().unwrap(), 4);
        assert_eq!(tensor_a.shape(), tensor_c.shape());
    }
}
