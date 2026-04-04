use crate::{Result, Tensor, tensor::indexing};

pub(crate) fn unary_op<F>(a: &Tensor, f: F) -> Result<Tensor>
where
    F: Fn(f32) -> f32,
{
    let mut out = Tensor::zeros(a.shape().to_vec())?;
    let out_buf = out.as_mut_slice_contiguous_unique()?;
    for i in 0..a.numel()? {
        let idx = indexing::convert_flat_position_to_logical_nd(i, a.shape());
        let av = a.get(&idx)?;
        out_buf[i] = f(av);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_utils::assert_approx_eq;
    #[test]
    fn unary_op_working_correct() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let fn_unary = |x: f32| x * 2.0;
        let tensor_c = unary_op(&tensor_a, fn_unary).unwrap();
        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 4.6);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 6.4);
    }
}
