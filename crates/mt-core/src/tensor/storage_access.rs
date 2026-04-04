use crate::Tensor;
use crate::error::{Result, TensorError};
use crate::tensor::shape;

// ===== Storage operations =====
impl Tensor {
    // this method gives us a mutable hanlde to guaranted unique contiguous buffer
    // only works for contiguous buffer
    pub(crate) fn as_mut_slice_contiguous_unique(&mut self) -> Result<&mut [f32]> {
        // check if the storage is contiguous and offset is zero
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous.into());
        }
        if self.offset() != 0 {
            return Err(TensorError::OffsetNotZero.into());
        }
        // check if the length of logical elements in tensor is same as no. of elements in storage
        // imp check for fast path storage api (contiguous) to work
        if self.storage_ref().len() != shape::compute_numel(&self.shape())? {
            return Err(TensorError::InvalidLayout.into());
        }
        Ok(self.storage_mut().as_mut_slice_unique())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Storage;
    #[test]
    fn get_mut_slice_from_tensor_for_unshared() {
        let store = Storage::zeros(4);
        let ptr = store.as_ptr();
        let mut obj = Tensor::from_parts_unchecked(store, vec![2, 2], vec![2, 1], 0);
        let unique = obj
            .as_mut_slice_contiguous_unique()
            .expect("should provide unique mutable slice");
        let ptr_first_mut = unique.as_ptr();
        assert_eq!(ptr, ptr_first_mut);
    }
    #[test]
    fn get_mut_slice_from_tensor_for_already_shared() {
        let store = Storage::zeros(4);
        let ptr = store.as_ptr();
        let store_clone = store.clone();
        let mut obj = Tensor::from_parts_unchecked(store_clone, vec![2, 2], vec![2, 1], 0);
        let unique_first = obj.as_mut_slice_contiguous_unique().unwrap();
        let ptr_first_mut = unique_first.as_ptr();
        assert_ne!(ptr, ptr_first_mut);
    }
}
