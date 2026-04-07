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
        Self::from_parts_unchecked(
            self.storage_clone(),
            shape.to_vec(),
            strides.to_vec(),
            offset,
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn compute_reshape_metadata_success_for_contiguous_tensor() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let md = compute_reshape_metadata(&tensor, &[4]).unwrap();

        assert_eq!(md.strides(), &[1]);
        assert_eq!(md.offset(), 0);
    }

    #[test]
    fn compute_reshape_metadata_fails_when_numel_is_incompatible() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let result = compute_reshape_metadata(&tensor, &[3]);

        assert!(result.is_err());
    }

    #[test]
    fn compute_reshape_metadata_fails_for_non_contiguous_tensor() {
        let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        // Same storage, but intentionally non-contiguous layout.
        let non_contiguous = base.internal_reshape_to_new_tensor(&[2, 2], &[1, 2], 0);

        assert!(!non_contiguous.is_contiguous());

        let result = compute_reshape_metadata(&non_contiguous, &[4]);

        assert!(result.is_err());
    }

    #[test]
    fn reshape_successfully_creates_new_tensor_with_new_shape() {
        let tensor = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();

        let reshaped = tensor.reshape(&[4]).unwrap();

        assert_eq!(reshaped.shape(), &[4]);
        assert_eq!(reshaped.strides(), &[1]);
        assert_eq!(reshaped.offset(), 0);

        // Value order should remain row-major contiguous order.
        assert_eq!(reshaped.get(&[0]).unwrap(), 10.0);
        assert_eq!(reshaped.get(&[1]).unwrap(), 20.0);
        assert_eq!(reshaped.get(&[2]).unwrap(), 30.0);
        assert_eq!(reshaped.get(&[3]).unwrap(), 40.0);

        // Original tensor should remain unchanged.
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.strides(), &[2, 1]);
    }

    #[test]
    fn internal_reshape_to_new_tensor_uses_exact_metadata_passed() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let new_tensor = tensor.internal_reshape_to_new_tensor(&[4], &[1], 0);

        assert_eq!(new_tensor.shape(), &[4]);
        assert_eq!(new_tensor.strides(), &[1]);
        assert_eq!(new_tensor.offset(), 0);

        assert_eq!(new_tensor.get(&[0]).unwrap(), 1.0);
        assert_eq!(new_tensor.get(&[1]).unwrap(), 2.0);
        assert_eq!(new_tensor.get(&[2]).unwrap(), 3.0);
        assert_eq!(new_tensor.get(&[3]).unwrap(), 4.0);
    }

    #[test]
    fn compute_reshape_metadata_preserves_offset() {
        let base =
            Tensor::from_vec(vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0], vec![3, 2]).unwrap();
        // Contiguous subview starting from flat offset 2.
        // Shape [2, 2], strides [2, 1], offset 2
        let sliced = base.internal_reshape_to_new_tensor(&[2, 2], &[2, 1], 2);
        assert!(sliced.is_contiguous());
        let md = compute_reshape_metadata(&sliced, &[4]).unwrap();
        assert_eq!(md.strides(), &[1]);
        assert_eq!(md.offset(), 2);
    }

    #[test]
    fn reshape_preserves_offset_for_contiguous_offset_view() {
        let base =
            Tensor::from_vec(vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0], vec![3, 2]).unwrap();

        let sliced = base.internal_reshape_to_new_tensor(&[2, 2], &[2, 1], 2);

        let reshaped = sliced.reshape(&[4]).unwrap();

        assert_eq!(reshaped.shape(), &[4]);
        assert_eq!(reshaped.strides(), &[1]);
        assert_eq!(reshaped.offset(), 2);
    }
}
