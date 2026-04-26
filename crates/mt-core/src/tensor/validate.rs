use crate::{OpError, Result, TensorError};
#[inline]
pub fn non_empty_shape(shape : &[usize]) -> Result<()>{
    if shape.iter().any(|&d| d== 0){
        return Err(TensorError::ZeroSizedTensor.into())
    }
    Ok(())
}

#[inline]
pub fn validate_axis(shape : &[usize],axis:usize) -> Result<()>{
 if axis >= shape.len(){
    return Err(OpError::AxisIndexInvalid.into())
 }
 Ok(())
}