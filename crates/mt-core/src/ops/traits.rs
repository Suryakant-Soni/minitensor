use crate::Result;
use crate::Tensor;
use crate::ops::scalar;
use std::ops::{Add, Mul, Sub};

// ===== Binary Compute Operations =====
impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: &'b Tensor) -> Self::Output {
        crate::ops::binary::add(self, rhs)
    }
}

impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Result<Tensor>;

    fn sub(self, rhs: &'b Tensor) -> Self::Output {
        crate::ops::binary::sub(self, rhs)
    }
}

impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Result<Tensor>;

    fn mul(self, rhs: &'b Tensor) -> Self::Output {
        crate::ops::binary::mul(self, rhs)
    }
}

// ===== Scalar Compute Operations =====

impl<'a> Add<f32> for &'a Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: f32) -> Self::Output {
        scalar::add(self, rhs)
    }
}

impl<'a> Sub<f32> for &'a Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: f32) -> Self::Output {
        scalar::sub(self, rhs)
    }
}

impl<'a> Mul<f32> for &'a Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: f32) -> Self::Output {
        scalar::mul(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_utils::assert_approx_eq;
    #[test]
    fn test_tensor_binary_addition_success() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let tensor_b = Tensor::from_vec(vec![1.1, 4.5, 0.8, 2.1], vec![2, 2]).unwrap();
        let tensor_c = (&tensor_a + &tensor_b).unwrap();
        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 3.4);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 7.7);
        assert_eq!(tensor_c.numel().unwrap(), 4);
        assert_eq!(tensor_a.shape(), tensor_c.shape());
        assert_eq!(tensor_b.strides(), tensor_c.strides());
    }
    #[test]
    fn test_tensor_binary_subtraction_success() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let tensor_b = Tensor::from_vec(vec![1.1, 4.5, 0.8, 2.1], vec![2, 2]).unwrap();
        let tensor_c = (&tensor_a - &tensor_b).unwrap();

        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 1.2);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), -1.3);
        assert_eq!(tensor_c.numel().unwrap(), 4);
        assert_eq!(tensor_a.shape(), tensor_c.shape());
        assert_eq!(tensor_b.strides(), tensor_c.strides());
    }

    #[test]
    fn test_tensor_binary_multiplication_success() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let tensor_b = Tensor::from_vec(vec![1.1, 4.5, 0.8, 2.1], vec![2, 2]).unwrap();
        let tensor_c = (&tensor_a * &tensor_b).unwrap();

        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 2.53);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 14.4);
        assert_eq!(tensor_c.numel().unwrap(), 4);
        assert_eq!(tensor_a.shape(), tensor_c.shape());
        assert_eq!(tensor_b.strides(), tensor_c.strides());
    }

    #[test]
    fn test_tensor_scalar_addition_success() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let rhs: f32 = 1.1;
        let tensor_c = (&tensor_a + rhs).unwrap();
        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 3.4);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 4.3);
        assert_eq!(tensor_c.numel().unwrap(), 4);
        assert_eq!(tensor_a.shape(), tensor_c.shape());
    }
    #[test]
    fn test_tensor_scalar_subtraction_success() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let rhs: f32 = 1.1;

        let tensor_c = (&tensor_a - rhs).unwrap();

        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 1.2);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 2.1);

        assert_eq!(tensor_c.numel().unwrap(), 4);
        assert_eq!(tensor_a.shape(), tensor_c.shape());
    }

    #[test]
    fn test_tensor_scalar_multiplication_success() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let rhs: f32 = 2.0;

        let tensor_c = (&tensor_a * rhs).unwrap();

        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 4.6);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 6.4);

        assert_eq!(tensor_c.numel().unwrap(), 4);
        assert_eq!(tensor_a.shape(), tensor_c.shape());
    }
}
