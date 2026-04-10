use crate::error::OpError;
use crate::{Result, Storage, Tensor};

// Broadcast views are read-only views
pub(crate) struct BroadcastView {
    strides: Vec<usize>,
    offset: usize,
}

// Broadcast views are read-only views
pub(crate) struct BinaryBroadcastMetadata {
    result_shape: Vec<usize>,
    a: BroadcastView,
    b: BroadcastView,
}

impl BinaryBroadcastMetadata {
    #[inline]
    pub(crate) fn result_shape(&self) -> &[usize] {
        &self.result_shape
    }

    #[inline]
    pub(crate) fn a(&self) -> &BroadcastView {
        &self.a
    }

    #[inline]
    pub(crate) fn b(&self) -> &BroadcastView {
        &self.b
    }
}

impl BroadcastView {
    #[inline]
    pub(crate) fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }
}

// assumes : tensor a and b inputs are already validated to be consistent tensor data
pub(crate) fn compute_broadcast_metadata(
    a: &Tensor,
    b: &Tensor,
) -> Result<BinaryBroadcastMetadata> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let a_strides_old = a.strides();
    let b_strides_old = b.strides();
    let a_rank = a_shape.len();
    let b_rank = b_shape.len();
    let rank = a_rank.max(b_rank);
    let mut result_shape = vec![0; rank];
    let mut a_strides = vec![0; rank];
    let mut b_strides = vec![0; rank];
    let a_pad = rank - a_rank;
    let b_pad = rank - b_rank;
    for i in (0..rank).rev() {
        let ai = if i >= a_pad { Some(i - a_pad) } else { None };
        let bi = if i >= b_pad { Some(i - b_pad) } else { None };
        let a_dim = match ai {
            Some(idx) => a_shape[idx],
            None => 1,
        };
        let b_dim = match bi {
            Some(idx) => b_shape[idx],
            None => 1,
        };
        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            return Err(OpError::BroadcastIncompatible.into());
        }
        result_shape[i] = a_dim.max(b_dim);
        // a is broadcasted for ith dimension
        a_strides[i] = match ai {
            None => 0,
            Some(idx) => {
                if b_dim > a_dim && a_dim == 1 {
                    0
                } else {
                    a_strides_old[idx]
                }
            }
        };
        // b is broadcasted for ith dimension
        b_strides[i] = match bi {
            None => 0,
            Some(idx) => {
                if a_dim > b_dim && b_dim == 1 {
                    0
                } else {
                    b_strides_old[idx]
                }
            }
        };
    }

    let a = BroadcastView {
        offset: a.offset(),
        strides: a_strides,
    };
    let b = BroadcastView {
        offset: b.offset(),
        strides: b_strides,
    };
    Ok(BinaryBroadcastMetadata { result_shape, a, b })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MtError;
    #[test]
    fn broadcast_working() {
        let tensor_a = Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7, 1.4, 0.9], vec![2, 3]).unwrap();
        let tensor_b = Tensor::from_vec(vec![0.8, 2.1, 4.3, 2.2, 1.4, 0.9], vec![2, 1, 3]).unwrap();
        let md = compute_broadcast_metadata(&tensor_a, &tensor_b).unwrap();
        assert_eq!(md.result_shape(), &[2, 2, 3]);
        assert_eq!(md.a().strides(), &[0, 3, 1]);
        assert_eq!(md.b().strides(), &[3, 0, 1]);
        assert_eq!(md.a().offset(), tensor_a.offset());
        assert_eq!(md.b().offset(), tensor_b.offset());
    }

    #[test]
    fn broadcast_incompatible() {
        let tensor_a =
            Tensor::from_vec(vec![2.3, 3.2, 5.3, 4.7, 1.4, 0.9, 0.0, 2.2], vec![2, 4]).unwrap();
        let tensor_b = Tensor::from_vec(vec![0.8, 2.1, 4.3, 2.2, 1.4, 0.9], vec![2, 1, 3]).unwrap();
        let md = compute_broadcast_metadata(&tensor_a, &tensor_b);
        assert!(matches!(
            md,
            Err(MtError::Operations(OpError::BroadcastIncompatible))
        ));
    }
}
