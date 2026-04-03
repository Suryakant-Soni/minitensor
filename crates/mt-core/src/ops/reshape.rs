use crate::Tensor;
use crate::error::*;

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

pub(crate) fn compute_reshape_metadata(
    old_tensor: &Tensor,
    new_shape: &[usize],
) -> Result<ReshapeMetadata> {
    // validate if the new shape can fit
    if old_tensor.numel()? != Tensor::compute_numel(new_shape)? {
        return Err(MtError::invalid_input("new shape is incompatible"));
    }
    if !old_tensor.is_contiguous() {
        return Err(MtError::processing_aborted(
            "reshape supported for only contiguous tensors",
        ));
    }
    let new_strides = Tensor::contiguous_strides(new_shape);
    let obj = ReshapeMetadata {
        strides: new_strides,
        offset: old_tensor.offset(),
    };
    Ok(obj)
}
