use crate::{Result, Tensor, ops::unary};

impl Tensor {
    pub(crate) fn map<F>(&self, f: F) -> Result<Tensor>
    where
        F: Fn(f32) -> f32,
    {
        unary::unary_op(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_utils::assert_approx_eq;
    #[test]
    fn unary_op_working_correct() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let fn_unary = |x: f32| x * 2.0;
        let tensor_c = tensor_a.map(fn_unary).unwrap();
        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 4.6);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 6.4);
    }
}
