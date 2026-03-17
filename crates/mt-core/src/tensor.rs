// mental model - tensor.rs file will not update or change the warehouse blocks but it will always tell you how to interpret and what are the viewing rules of the raw memory of storage.rs
// tensor will never own the layout, it defines view mappping using tools like strides and logical indexing
// tensor will only describe how to read the line in N-D grid
// Storage = warehouse row
// Shape = grid overlay placed on top of warehouse
// Strides = walking rule inside warehouse
// Offset = starting block
//Tensor = (Storage + interpretation rules)

use crate::error::*;
use crate::storage::Storage;
struct Tensor {
    storage: Storage,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        // validate product(shape) == data.len() // compute strides
        // validating that data length should be equal to numel
        let numel = Self::compute_numel(&shape)?;
        if data.len() != numel {
            return Err(TensorError::IndexRankMismatch {
                expected: numel,
                got: data.len(),
            })?;
        }
        let obj = Self {
            storage: Storage::from_vec(data),
            offset: 0,
            strides: Self::contiguous_strides(&shape),
            shape: shape,
        };
        Ok(obj)
    }
    fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let len = shape.len();
        let mut strides = vec![0usize; len];
        if len == 0 {
            return strides;
        }
        strides[len - 1] = 1;
        for i in (0..len - 1).rev() {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        strides
    }
    fn compute_numel(shape: &[usize]) -> Result<usize> {
        let mut res = 1usize;
        for &elem in shape {
            res = res.checked_mul(elem).ok_or(TensorError::NumelOverflow)?;
        }
        Ok(res)
    }
    pub fn zeros(shape: Vec<usize>) -> Result<Self> {
        // allocate storage
        let obj = Self {
            storage: Storage::zeros(Self::compute_numel(&shape)?),
            offset: 0,
            strides: Self::contiguous_strides(&shape),
            shape,
        };
        Ok(obj)
    }
    // this method will take indices of the high level and give out data at flat index position
    pub fn get(&self, idx: &[usize]) -> Result<f32> {
        // validate if the length of given indices is correct
        if self.shape.len() != idx.len() {
            return Err(TensorError::IndexRankMismatch {
                expected: self.shape.len(),
                got: idx.len(),
            })?;
        }
        // validate if all the input indices are bound
        for i in 0..idx.len(){
            if idx[i] >= self.shape[i]{
                return Err(TensorError::IndexNotBound {
                    max_index_length : self.shape[i]-1,
                })?;
            }
        }
        // compute flat index using strides + offset
        let index: usize = idx.iter().zip(self.strides.iter()).map(|(&i,&j)| i * j).sum();
        
        let value = self.storage.get(index + self.offset).expect("Tensor::get: flat index out of bounds (internal bug)");
        Ok(*value)
    }
}

//          pub fn get_mut(&mut self, idx: &[usize]) -> &mut f32 {
//              // use storage.as_mut_slice_unique()
//              }
//             }
