// mental model - tensor.rs file will not update or change the warehouse blocks but it will always tell you how to interpret and what are the viewing rules of the raw memory of storage.rs
// tensor will never own the layout, it defines view mappping using tools like strides and logical indexing
// tensor will only describe how to read the line in N-D grid
// Storage = warehouse row
// Shape = grid overlay placed on top of warehouse
// Strides = walking rule inside warehouse
// Offset = starting block
//Tensor = (Storage + interpretation rules)

use crate::Storage;
use crate::error::*;
use crate::tensor::shape;
pub struct Tensor {
    storage: Storage,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

// ===== Constructors =====
impl Tensor {
    pub(crate) fn from_parts_unchecked(
        storage: Storage,
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
    ) -> Tensor {
        Tensor {
            storage,
            shape,
            strides,
            offset,
        }
    }
    pub(crate) fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        // validate product(shape) == data.len() // compute strides
        // validating that data length should be equal to numel
        let numel = shape::compute_numel(&shape)?;
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
            strides: shape::contiguous_strides(&shape),
            shape: shape,
        };
        Ok(obj)
    }

    pub(crate) fn zeros(shape: Vec<usize>) -> Result<Self> {
        // allocate storage
        let obj = Self {
            storage: Storage::zeros(shape::compute_numel(&shape)?),
            offset: 0,
            strides: shape::contiguous_strides(&shape),
            shape,
        };
        Ok(obj)
    }
}

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

    // get the numel by on the fly computation, we will not save this in state as this is derived quantity not fundamental one
    pub(crate) fn numel(&self) -> Result<usize> {
        shape::compute_numel(&self.shape)
    }
}
// ===== Helper functions =====
impl Tensor {
    pub(crate) fn storage_ref(&self) -> &Storage {
        &self.storage
    }
    pub(crate) fn storage_mut(&mut self) -> &mut Storage {
        &mut self.storage
    }
    pub(crate) fn storage_clone(&self) -> Storage {
        self.storage.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_utils::assert_approx_eq;
    #[test]
    fn from_vec_working_expected() {
        let tensor = Tensor::from_vec(vec![2.3, 3.2, 5.32, 43.3], vec![2, 2]).unwrap();
        // shape should be correctly propagated
        assert_eq!(tensor.shape(), vec![2, 2]);
        // backing storage length should be 4
        assert_eq!(tensor.storage_ref().len(), 4);
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
        let tensor = Tensor::zeros(vec![2, 2]).unwrap();
        // shape should be correctly propagated
        assert_eq!(tensor.shape(), vec![2, 2]);
        // backing storage length should be 4
        assert_eq!(tensor.storage_ref().len(), 4);
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
}
