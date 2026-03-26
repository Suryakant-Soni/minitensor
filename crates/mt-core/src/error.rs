pub type Result<T> = std::result::Result<T, MtError>;

#[derive(Debug)]
pub enum MtError{
    Tensor(TensorError),
    Storage(StorageError),
}

#[derive(Debug)]
pub enum TensorError{
    NumelOverflow,
    ShapeDataLenMismatch{expected: usize, got: usize},
    IndexRankMismatch{expected: usize, got: usize},
    IndexOutOfBounds{dimension_length: usize,requested_index : usize},
}

#[derive(Debug)]
pub enum StorageError{
}

// to support auto convert of specific errors to generic error MtError Object
impl From<TensorError> for MtError {
    fn from(err: TensorError) -> Self{
        MtError::Tensor(err)
    }
}

impl From<StorageError> for MtError{
    fn from(err: StorageError) -> Self{
        MtError::Storage(err)
    }
}