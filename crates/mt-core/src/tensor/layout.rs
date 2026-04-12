use crate::Tensor;
use crate::tensor::shape;

impl Tensor {
    /// Checks whether the tensor's underlying storage layout is contiguous.
    pub(crate) fn is_contiguous(&self) -> bool {
        self.strides() == shape::contiguous_strides(&self.shape())
    }
}
