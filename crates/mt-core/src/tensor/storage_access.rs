use crate::Tensor;
use crate::tensor::shape;
use crate::error::{TensorError,Result};

// ===== Storage operations =====
impl Tensor {
    // this method gives us a mutable hanlde to guaranted unique contiguous buffer
    // only works for contiguous buffer
    pub(crate) fn as_mut_slice_contiguous_unique(&mut self) -> Result<&mut [f32]> {
        // check if the storage is contiguous and offset is zero
        if !self.is_contiguous() || self.offset() != 0 {
            return Err(TensorError::NotContiguous.into());
        }
        // check if the length of logical elements in tensor is same as no. of elements in storage
        // imp check for fast path storage api (contiguous) to work
        if self.storage().len() != shape::compute_numel(&self.shape())? {
            return Err(TensorError::InvalidLayout.into());
        }
        Ok(self.storage_mut().as_mut_slice_unique())
    }
}
