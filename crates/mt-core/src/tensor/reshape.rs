use crate::Tensor;
use crate::error::*;
use crate::tensor::shape;

pub(crate) struct ReshapeMetadata {
    strides: Vec<usize>,
    offset: usize,
}

impl ReshapeMetadata {
    #[inline]
    pub(crate) fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }
}
impl Tensor {
    pub(crate) fn internal_reshape_to_new_tensor(
        &self,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Tensor {
        Self::from_parts_unchecked(self.storage_clone(), shape.to_vec(), strides.to_vec(), offset)
    }

    // reshape and form a new tensor with new shape and same old storage
    pub(crate) fn reshape(&self, new_shape: &[usize]) -> Result<Tensor> {
        let md = compute_reshape_metadata(self, new_shape)?;
        Ok(self.internal_reshape_to_new_tensor(new_shape, md.strides(), md.offset()))
    }
}

pub(crate) fn compute_reshape_metadata(
    old_tensor: &Tensor,
    new_shape: &[usize],
) -> Result<ReshapeMetadata> {
    // validate if the new shape can fit
    if old_tensor.numel()? != shape::compute_numel(new_shape)? {
        return Err(MtError::invalid_input("new shape is incompatible"));
    }
    if !old_tensor.is_contiguous() {
        return Err(MtError::processing_aborted(
            "reshape supported for only contiguous tensors",
        ));
    }
    let new_strides = shape::contiguous_strides(new_shape);
    let obj = ReshapeMetadata {
        strides: new_strides,
        offset: old_tensor.offset(),
    };
    Ok(obj)
}
