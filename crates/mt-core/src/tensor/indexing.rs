use crate::Tensor;
use crate::tensor::shape;
use crate::{Result, TensorError};

// ===== Indexing =====
impl Tensor {
    /// Converts a logical index into the corresponding flat backing storage index.
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

    /// Returns the value at a logical tensor index.
    pub(crate) fn get(&self, idx: &[usize]) -> Result<f32> {
        let index = self.get_flat_index(idx)?;
        let value = self
            .storage_ref()
            .get(index)
            .expect("Tensor::get: flat index out of bounds (internal bug)");
        Ok(*value)
    }

    pub(crate) fn get_with_flat(&self, flat: usize) -> Result<f32> {
        let value = self
            .storage_ref()
            .get(flat)
            .expect("Tensor::get: flat index out of bounds (internal bug)");
        Ok(*value)
    }

    /// # Safety
    ///
    /// Caller must guarantee that `flat < self.storage_ref().len()`.
    /// The pre-validation should lie in the tensor constructor.
    pub(crate) unsafe fn get_with_flat_unchecked(&self, flat: usize) -> f32 {
        unsafe { *self.storage_ref().get_unchecked(flat) }
    }

    /// Returns a mutable handle to the value at a logical tensor index.
    pub(crate) fn get_mut(&mut self, idx: &[usize]) -> Result<&mut f32> {
        let index = self.get_flat_index(idx)?;
        let slice = self.storage_mut().as_mut_slice_unique();
        let elem = slice
            .get_mut(index)
            .expect("Tensor::get_mut: calculated flat index is out of bounds");
        Ok(elem)
    }
}

/// Internal kernel helper.
///
/// Advances `l_idx` to the next logical coordinate in row-major order and
/// updates `a_flat` and `b_flat` to remain consistent with that new logical index.
///
/// # Preconditions
///
/// - `l_idx.len() == result_shape.len()`
/// - `a_strides.len() == result_shape.len()`
/// - `b_strides.len() == result_shape.len()`
/// - `a_flat` and `b_flat` must correspond to the current `l_idx`
/// - `l_idx` must not already be the last valid index in `result_shape`
///
/// # Panics
///
/// Panics if called when `l_idx` is already the last valid logical index.
pub(crate) fn compute_index_on_increment(
    l_idx: &mut [usize],
    a_flat: &mut usize,
    b_flat: &mut usize,
    result_shape: &[usize],
    a_strides: &[usize],
    b_strides: &[usize],
) {
    // traverse the dimensions starting from the right most dimension
    for dim in (0..result_shape.len()).rev() {
        // dimension limit reached/overflown
        // l_idx = logical index, a_flat = flat index for tensor a, b_flat = flat index for tensor b
        if l_idx[dim] == result_shape[dim] - 1 {
            // -> last index already processed, should not be the case
            if dim == 0 {
                panic!("last index already reached, should be controlled by the caller");
            }
            // reset the overflown dimension to zero
            l_idx[dim] = 0;
            // since the next dimension will increment flat index wrt its strides, rewind the contribution of old dimension from the flat count
            *a_flat -= a_strides[dim] * (result_shape[dim] - 1);
            *b_flat -= b_strides[dim] * (result_shape[dim] - 1);
            continue;
        }
        l_idx[dim] += 1;
        // add strides to flats for the dimension dim, ( the steps needed for this dimension to be incremented)
        *a_flat += a_strides[dim];
        *b_flat += b_strides[dim];
        break;
    }
}

/// Forms the logical N-D coordinate for a row-major traversal position.
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

    #[test]
    fn compute_index_on_increment_positive() {
        let mut l_idx = [0usize, 0];
        let mut a_flat = 0usize;
        let mut b_flat = 0usize;
        let result_shape = [2, 3];
        let numel = shape::compute_numel(&result_shape).unwrap();
        let l_idx_expected = vec![[0usize, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]];

        for i in 0usize..numel {
            assert_eq!(a_flat, i);
            assert_eq!(b_flat, i);
            assert_eq!(l_idx_expected[i], l_idx);
            if i == numel - 1 {
                return;
            }
            compute_index_on_increment(
                &mut l_idx,
                &mut a_flat,
                &mut b_flat,
                &result_shape,
                &[3, 1],
                &[3, 1],
            );
        }
    }

    #[test]
    #[should_panic]
    fn compute_index_on_increment_negative() {
        let mut l_idx = [0usize, 2];
        let mut a_flat = 0usize;
        let mut b_flat = 0usize;
        let result_shape = [2, 3];
        let numel = shape::compute_numel(&result_shape).unwrap();

        for i in 0usize..numel {
            assert_eq!(a_flat, i);
            assert_eq!(b_flat, i);
            if i == numel - 1 {
                return;
            }
            compute_index_on_increment(
                &mut l_idx,
                &mut a_flat,
                &mut b_flat,
                &result_shape,
                &[3, 1],
                &[3, 1],
            );
        }
    }
}
