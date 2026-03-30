use std::sync::Arc;
// mental model - the raw products can be arranged in warehouse is a row or in a simple line of blocks,
// it does not matter if they are advertised and ordered or how consumed at other layers, but to keep them in warehouse
// they will always be kept in the simplest deterministic way, a fixed row so we are always deterministic of our warehouse size

// physical backing buffer for tensors

// - flat storage, can be shared using views via ARC
// - fixed size buffer no resizing or growing buffer ( more deterministic & memory efficient)
// - read only be default, mutation possible via `make_unique` and `as_mut_slice_unique`

#[derive(Clone)]
pub(crate) struct Storage {
    buf: Arc<[f32]>,
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage")
            .field("len", &self.len())
            .field("data_id", &self.data_id())
            .finish()
    }
}

impl Storage {
    pub(crate) fn from_vec(vec: Vec<f32>) -> Self {
        Self {
            buf: Arc::<[f32]>::from(vec),
        }
    }

    pub(crate) fn zeros(len: usize) -> Self {
        Self::from_vec(vec![0.0; len])
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[f32] {
        &self.buf
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const f32 {
        self.buf.as_ptr()
    }

    #[inline]
    pub(crate) fn ptr_eq(&self, other: &Storage) -> bool {
        Arc::ptr_eq(&self.buf, &other.buf)
    }

    #[inline]
    pub(crate) fn data_id(&self) -> usize {
        self.as_ptr() as usize
    }

    pub(crate) fn get(&self, index: usize) -> Option<&f32> {
        self.buf.get(index)
    }

    // if the buffer is shared, it clones the entire buffer since we are using
    // copy-on-write principle
    // make unique will use copy-on-write if there is not other shared reference of this arc
    // if multiple references exists - it will copy the data and create new allocation bundle it in arc
    // and push it to buf field of same struct instance (self)

    pub(crate) fn make_unique(&mut self) {
        let is_some = Arc::get_mut(&mut self.buf).is_some();
        if is_some {
            // this means that you can get mutable access since there is not other reference of this arc
            return;
        }
        // clone data into new allocation area
        let cloned = self.buf.as_ref().to_vec();
        self.buf = Arc::<[f32]>::from(cloned);
        debug_assert!(Arc::get_mut(&mut self.buf).is_some());
    }

    // get a mutable slice to the unique backing buffer, cloning first if it was shared
    pub(crate) fn as_mut_slice_unique(&mut self) -> &mut [f32] {
        self.make_unique();
        Arc::get_mut(&mut self.buf)
            .expect("Arc is still shared (not unique) even after make_unique()")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Construction behaviour
    #[test]
    fn from_vec_preserves_length_and_values() {
        let storage = Storage::from_vec(vec![0.95, 1.19, 2.44, 3.01]);
        assert_eq!(storage.len(), 4);
        assert_eq!(storage.get(0), Some(&0.95));
        assert_eq!(storage.get(1), Some(&1.19));
        assert_eq!(storage.get(2), Some(&2.44));
        assert_eq!(storage.get(3), Some(&3.01));
        assert_eq!(storage.get(4), None);
        assert_eq!(storage.is_empty(), false);
    }

    #[test]
    fn from_vec_empty_vec_creates_empty_storage() {
        let storage = Storage::from_vec(vec![]);
        assert_eq!(storage.len(), 0);
        assert_eq!(storage.get(0), None);
        assert_eq!(storage.is_empty(), true);
    }

    #[test]
    fn zeros_check_length_check_values() {
        let store = Storage::zeros(3);
        assert_eq!(store.len(), 3);
        assert_eq!(store.get(0), Some(&0.0));
        assert_eq!(store.get(1), Some(&0.0));
        assert_eq!(store.get(3), None);
        assert_eq!(store.is_empty(), false);
    }

    #[test]
    fn zeros_check_zero_len_storage() {
        let store = Storage::zeros(0);
        assert_eq!(store.len(), 0);
        assert_eq!(store.get(0), None);
        assert_eq!(store.is_empty(), true);
    }

    // COW copy on write behaviour

    #[test]
    fn as_mut_slice_unique_returns_same_for_unshared() {
        let mut store = Storage::zeros(2);
        let ptr = store.as_ptr();
        let unique = store.as_mut_slice_unique();
        let ptr_first_mut = unique.as_ptr();
        assert_eq!(ptr, ptr_first_mut);
    }

    #[test]
    fn as_mut_slice_unique_returns_new_for_shared() {
        let mut store = Storage::zeros(2);
        let ptr = store.as_ptr();
        let unique_first = store.as_mut_slice_unique();
        let ptr_first_mut = unique_first.as_ptr();
        assert_eq!(ptr, ptr_first_mut);
        // get the mutable share
        let store_mut = store.clone();
        let unique_sec = store.as_mut_slice_unique();
        let ptr_second_mut = unique_sec.as_ptr();
        let clone_ptr = store_mut.as_ptr();
        assert_ne!(ptr, ptr_second_mut);
        assert_ne!(clone_ptr, ptr_second_mut);
    }
}
