// mental model - tensor.rs file will not update or change the warehouse blocks but it will always tell you how to interpret and what are the viewing rules of the raw memory of storage.rs 
// tensor will never own the layout, it defines view mappping using tools like strides and logical indexing
// tensor will only describe how to read the line in N-D grid
// Storage = warehouse row
// Shape = grid overlay placed on top of warehouse
// Strides = walking rule inside warehouse
// Offset = starting block
//Tensor = (Storage + interpretation rules)

use crate::storage::Storage;
use crate::error::*;
struct Tensor{
storage : Storage,
shape : Vec<usize>,
strides : Vec<usize>,
offset : usize,
}

impl Tensor { 
    
    pub fn from_vec(&mut self, data: Vec<f32>, shape: Vec<usize>) -> Result<Self> { // validate product(shape) == data.len() // compute strides 
        // validating that data length should be equal to numel
        let numel = Self::compute_numel(&shape)?;
        if data.len() != numel{
            return Err(TensorError::ShapeDataLenMismatch { expected: numel, got: data.len() })?;
        }
        let mut obj = Self{
            storage: Storage::from_vec(data),
            offset: 0,
            shape: shape,
        // compute strides
        };

         Ok(obj) 
    } 
    fn contiguous_strides(shape: &[usize]) -> [usize]{
        
    }
    fn compute_numel(shape: &[usize]) -> Result<usize>{
        let mut res = 1usize;
            for &elem in shape{
                res = res.checked_mul(elem).ok_or(TensorError::NumelOverflow)?;
            }
            Ok(res)
    }
} 


    // pub fn zeros(shape: Vec<usize>) -> Self { // allocate storage 
    //     }
    //      pub fn get(&self, idx: &[usize]) -> f32 { 
    //         // compute flat index using strides + offset
    //          } 
    //          pub fn get_mut(&mut self, idx: &[usize]) -> &mut f32 {
    //              // use storage.as_mut_slice_unique() 
    //              } 
    //             }