// mental model - tensor.rs file will not update or change the warehouse blocks but it will always tell you how to interpret and what are the viewing rules of the raw memory of storage.rs
// tensor will never own the layout, it defines view mappping using tools like strides and logical indexing
// tensor will only describe how to read the line in N-D grid
// Storage = warehouse row
// Shape = grid overlay placed on top of warehouse
// Strides = walking rule inside warehouse
// Offset = starting block
//Tensor = (Storage + interpretation rules)

use crate::error::*;
use crate::storage::Storage;
use std::ops::{Add, Mul, Sub};
pub struct Tensor {
    storage: Storage,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

// ===== Constructors =====
impl Tensor {
    pub(crate) fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        // validate product(shape) == data.len() // compute strides
        // validating that data length should be equal to numel
        let numel = Self::compute_numel(&shape)?;
        if data.len() != numel {
            return Err(TensorError::ShapeDataLenMismatch {
                expected: numel,
                got: data.len(),
            }
            .into());
        }
        let obj = Self {
            storage: Storage::from_vec(data),
            offset: 0,
            strides: Self::contiguous_strides(&shape),
            shape: shape,
        };
        Ok(obj)
    }

    pub(crate) fn zeros(shape: Vec<usize>) -> Result<Self> {
        // allocate storage
        let obj = Self {
            storage: Storage::zeros(Self::compute_numel(&shape)?),
            offset: 0,
            strides: Self::contiguous_strides(&shape),
            shape,
        };
        Ok(obj)
    }
}

// ===== Metadata =====
impl Tensor {
    #[inline]
    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub(crate) fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }
    // this method checks if the tensor underlying storage is contiguous or not
    pub(crate) fn is_contiguous(&mut self) -> bool {
        self.strides == Self::contiguous_strides(&self.shape)
    }

    // get the numel by on the fly computation, we will not save this in state as this is derived quantity not fundamental one
    pub(crate) fn numel(&self) -> Result<usize> {
        Self::compute_numel(&self.shape)
    }
}

// ===== Indexing =====
impl Tensor {
    // it will take the logical index slice and convert into corresponding flat backing index
    fn get_flat_index(&self, idx: &[usize]) -> Result<usize> {
        // validate if the length of given indices is correct
        if self.shape.len() != idx.len() {
            return Err(TensorError::IndexRankMismatch {
                expected: self.shape.len(),
                got: idx.len(),
            }
            .into());
        }
        // validate if all the input indices are bound
        for i in 0..idx.len() {
            if idx[i] >= self.shape[i] {
                return Err(TensorError::IndexOutOfBounds {
                    dimension_length: self.shape[i],
                    requested_index: idx[i],
                }
                .into());
            }
        }
        // compute flat index using strides
        let index: usize = idx
            .iter()
            .zip(self.strides.iter())
            .map(|(&i, &j)| i * j)
            .sum();
        Ok(index + self.offset)
    }

    // this method will take indices of the high level and give out data at flat index position
    pub(crate) fn get(&self, idx: &[usize]) -> Result<f32> {
        let index = self.get_flat_index(idx)?;
        let value = self
            .storage
            .get(index)
            .expect("Tensor::get: flat index out of bounds (internal bug)");
        Ok(*value)
    }

    // this method will take indices of the high level and give out mutable handle for data at flat index position
    pub(crate) fn get_mut(&mut self, idx: &[usize]) -> Result<&mut f32> {
        let index = self.get_flat_index(idx)?;
        let slice = self.storage.as_mut_slice_unique();
        let elem = slice
            .get_mut(index)
            .expect("Tensor::get_mut: calculated flat index is out of bounds");
        Ok(elem)
    }
}

// ===== Helper functions =====
impl Tensor {
    fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let len = shape.len();
        let mut strides = vec![0usize; len];
        if len == 0 {
            return strides;
        }
        strides[len - 1] = 1;
        for i in (0..len - 1).rev() {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        strides
    }
    // it computes the number of element in the tensor by multiplying elements of shape,also checks multiplication overflow in computation
    fn compute_numel(shape: &[usize]) -> Result<usize> {
        let mut res = 1usize;
        for &elem in shape {
            res = res.checked_mul(elem).ok_or(TensorError::NumelOverflow)?;
        }
        Ok(res)
    }
}

// ===== Storage operations =====
impl Tensor {
    // this method gives us a mutable hanlde to guaranted unique contiguous buffer
    // only works for contiguous buffer
    pub(crate) fn as_mut_slice_contiguous_unique(&mut self) -> Result<&mut [f32]> {
        // check if the storage is contiguous and offset is zero
        if !self.is_contiguous() || self.offset != 0 {
            return Err(TensorError::NotContiguous.into());
        }
        // check if the length of logical elements in tensor is same as no. of elements in storage
        // imp check for fast path storage api (contiguous) to work
        if self.storage.len() != Self::compute_numel(&self.shape)? {
            return Err(TensorError::InvalidLayout.into());
        }
        Ok(self.storage.as_mut_slice_unique())
    }
}

// ===== Compute Operations =====
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_utilities::assert_approx_eq;
    #[test]
    fn from_vec_working_expected() {
        let mut tensor = Tensor::from_vec(vec![2.3, 3.2, 5.32, 43.3], vec![2, 2]).unwrap();
        // shape should be correctly propagated
        assert_eq!(tensor.shape(), vec![2, 2]);
        // backing storage length should be 4
        assert_eq!(tensor.storage.len(), 4);
        // tensor should be contiguoug by default
        assert!(tensor.is_contiguous());
        // shape and strides should be of same length
        assert_eq!(tensor.strides().len(), tensor.shape().len());
        // by default offset should be zero
        assert_eq!(tensor.offset(), 0);
        // strides should be correct
        assert_eq!(tensor.strides(), vec![2, 1]);
    }
    #[test]
    fn zeros_working_expected() {
        let mut tensor = Tensor::zeros(vec![2, 2]).unwrap();
        // shape should be correctly propagated
        assert_eq!(tensor.shape(), vec![2, 2]);
        // backing storage length should be 4
        assert_eq!(tensor.storage.len(), 4);
        // tensor should be contiguoug by default
        assert!(tensor.is_contiguous());
        // shape and strides should be of same length
        assert_eq!(tensor.strides().len(), tensor.shape().len());
        // by default offset should be zero
        assert_eq!(tensor.offset(), 0);
        // strides should be correct
        assert_eq!(tensor.strides(), vec![2, 1]);
    }

    #[test]
    fn from_vec_shape_data_len_mismatch() {
        let tensor = Tensor::from_vec(vec![2.3, 3.2, 5.32, 43.3], vec![2, 1]);
        assert!(matches!(
            tensor,
            Err(MtError::Tensor(TensorError::ShapeDataLenMismatch {
                expected: 2,
                got: 4
            }))
        ));
    }

    #[test]
    fn from_vec_numel_calculation_overflows() {
        let tensor = Tensor::from_vec(vec![2.0, 1.0], vec![usize::MAX, 2]);
        assert!(matches!(
            tensor,
            Err(MtError::Tensor(TensorError::NumelOverflow))
        ));
    }

    #[test]
    fn test_tensor_indexing_features() {
        let tensor = Tensor::from_vec(vec![2.3, 3.2, 5.32, 43.3], vec![2, 2]).unwrap();
        // positive cases
        // getting flat index from logical index is working
        assert_eq!(tensor.get_flat_index(&[1, 1]).unwrap(), 3);
        assert_eq!(tensor.get_flat_index(&[0, 1]).unwrap(), 1);
        // get value at logical index is working
        assert_eq!(tensor.get(&[1, 0]).unwrap(), 5.32);
        assert_eq!(tensor.get(&[0, 0]).unwrap(), 2.3);
        // negative cases
        // if invalid logical index length is passed
        assert!(matches!(
            tensor.get_flat_index(&[2, 2, 1]),
            Err(MtError::Tensor(TensorError::IndexRankMismatch {
                expected: 2,
                got: 3
            }))
        ));
        //if logical index' length is correct but the index element is invalid
        assert!(matches!(
            tensor.get_flat_index(&[0, 3]),
            Err(MtError::Tensor(TensorError::IndexOutOfBounds {
                dimension_length: 2,
                requested_index: 3
            }))
        ));
    }

    #[test]
    fn test_getting_mutable_handle_to_element() {
        let mut tensor = Tensor::from_vec(vec![2.3, 3.2, 5.32, 43.3], vec![2, 2]).unwrap();
        assert_eq!(tensor.get(&[1, 0]).unwrap(), 5.32);
        let handle = tensor.get_mut(&[1, 0]).unwrap();
        *handle = 5.20;
        assert_ne!(tensor.get(&[1, 0]).unwrap(), 5.32);
        assert_eq!(tensor.get(&[1, 0]).unwrap(), 5.20);
    }

    // testing compute operations
    #[test]
    fn test_tensor_addition_success() {
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
    fn test_tensor_subtraction_success() {
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
    fn test_tensor_multiplication_success() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7], vec![2, 2]).unwrap();
        let tensor_b = Tensor::from_vec(vec![1.1, 4.5, 0.8, 2.1], vec![2, 2]).unwrap();
        let tensor_c = (&tensor_a * &tensor_b).unwrap();

        assert_approx_eq(tensor_c.get(&[0, 0]).unwrap(), 2.53);
        assert_approx_eq(tensor_c.get(&[0, 1]).unwrap(), 14.4);
        assert_eq!(tensor_c.numel().unwrap(), 4);
        assert_eq!(tensor_a.shape(), tensor_c.shape());
        assert_eq!(tensor_b.strides(), tensor_c.strides());
    }
}
