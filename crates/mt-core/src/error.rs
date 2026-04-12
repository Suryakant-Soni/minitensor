pub type Result<T> = std::result::Result<T, MtError>;

#[derive(Debug)]
pub enum MtError {
    Tensor(TensorError),
    Storage(StorageError),
    Operations(OpError),
    InvalidInput { reason: String },
    ProcessingAborted { reason: String },
}

impl MtError {
    pub(crate) fn invalid_input(reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            reason: reason.into(),
        }
    }
    pub(crate) fn processing_aborted(reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            reason: reason.into(),
        }
    }
}

pub(crate) fn ensure_non_empty<T>(name: &str, slice: &[T]) -> Result<()> {
    if slice.is_empty() {
        return Err(MtError::invalid_input(format!("{name} cannot be empty")));
    }
    Ok(())
}

#[derive(Debug)]
pub enum TensorError {
    NumelOverflow,
    InvalidLayout,
    ShapeMismatch,
    NotContiguous,
    OffsetNotZero,
    ShapeDataLenMismatch {
        expected: usize,
        got: usize,
    },
    IndexRankMismatch {
        expected: usize,
        got: usize,
    },
    IndexOutOfBounds {
        dimension_length: usize,
        requested_index: usize,
    },
}

#[derive(Debug)]
pub enum OpError {
    BroadcastIncompatible,
}

#[derive(Debug)]
pub enum StorageError {}

/// Converts tensor-specific errors into the generic `MtError` type.
impl From<TensorError> for MtError {
    fn from(err: TensorError) -> Self {
        MtError::Tensor(err)
    }
}

impl From<StorageError> for MtError {
    fn from(err: StorageError) -> Self {
        MtError::Storage(err)
    }
}
impl From<OpError> for MtError {
    fn from(err: OpError) -> Self {
        MtError::Operations(err)
    }
}
