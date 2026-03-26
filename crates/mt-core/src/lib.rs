pub(crate) mod storage;
pub mod tensor;
pub mod error;
pub mod ops;

pub use tensor::Tensor;
pub use error::{MtError,Result};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
    }
}
