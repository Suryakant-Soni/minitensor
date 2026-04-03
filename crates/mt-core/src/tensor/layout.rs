use crate::Tensor;
use crate::tensor::shape;

impl Tensor {
    // this method checks if the tensor underlying storage is contiguous or not
    pub(crate) fn is_contiguous(&self) -> bool {
        self.strides() == shape::contiguous_strides(&self.shape())
    }
}
