use crate::Tensor;
use crate::tensor::shape;
use crate::{Result, TensorError};

// ===== Indexing =====
impl Tensor {
    // it will take the logical index slice and convert into corresponding flat backing index
    fn get_flat_index(&self, idx: &[usize]) -> Result<usize> {
        // validate if the length of given indices is correct
        if self.shape().len() != idx.len() {
            return Err(TensorError::IndexRankMismatch {
                expected: self.shape().len(),
                got: idx.len(),
            }
            .into());
        }
        // validate if all the input indices are bound
        for i in 0..idx.len() {
            if idx[i] >= self.shape()[i] {
                return Err(TensorError::IndexOutOfBounds {
                    dimension_length: self.shape()[i],
                    requested_index: idx[i],
                }
                .into());
            }
        }
        // compute flat index using strides
        let index: usize = idx
            .iter()
            .zip(self.strides().iter())
            .map(|(&i, &j)| i * j)
            .sum();
        Ok(index + self.offset())
    }

    // this method will take indices of the high level and give out data at flat index position
    pub(crate) fn get(&self, idx: &[usize]) -> Result<f32> {
        let index = self.get_flat_index(idx)?;
        let value = self
            .storage_ref()
            .get(index)
            .expect("Tensor::get: flat index out of bounds (internal bug)");
        Ok(*value)
    }

    // this method will take indices of the high level and give out mutable handle for data at flat index position
    pub(crate) fn get_mut(&mut self, idx: &[usize]) -> Result<&mut f32> {
        let index = self.get_flat_index(idx)?;
        let slice = self.storage_mut().as_mut_slice_unique();
        let elem = slice
            .get_mut(index)
            .expect("Tensor::get_mut: calculated flat index is out of bounds");
        Ok(elem)
    }
}

// forms the logical N-D coordinate for a row-major traversal position, based only on shape
pub(crate) fn convert_flat_position_to_logical_nd(mut index: usize, shape: &[usize]) -> Vec<usize> {
    // initiate a vector of the same length as shape
    let mut idx = vec![0; shape.len()];
    let numel = shape::compute_numel(shape).expect("input shape is invalid");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MtError;
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

    #[test]
    fn unravel_index_working_as_expected() {
        assert_eq!(convert_flat_position_to_logical_nd(0, &[2, 2]), vec![0, 0]);
        assert_eq!(convert_flat_position_to_logical_nd(2, &[2, 2]), vec![1, 0]);
    }

    #[test]
    #[should_panic]
    fn unravel_index_panics_with_index_greater_than_numel() {
        assert_eq!(convert_flat_position_to_logical_nd(5, &[2, 2]), vec![1, 1]);
    }
}
