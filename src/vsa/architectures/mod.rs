//! Implementation of different architectures.

mod bsc;
mod map;

pub use self::bsc::BinarySpatterCode;
pub use self::map::MultiplyAddPermute;

/// A vector symbolic architecture.
pub trait VectorSymbolicArchitecture: Clone {
    /// The underlying storage of a single vector.
    type Storage: Storage;
    /// The underlying storage of a multi vector.
    type StorageMulti: Storage;

    /// Create a random vector in the architecture.
    fn random(&self, size: usize) -> Self::Storage;
    /// Normalize a multi-vector.
    fn normalize(&self, storage: Self::StorageMulti) -> Self::Storage;
    /// Bundle two vectors.
    fn bundle_multi(&self, a: &Self::Storage, b: &Self::Storage) -> Self::StorageMulti;
    /// Bundle a vector and normalize it.
    fn bundle(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        self.normalize(self.bundle_multi(a, b))
    }

    /// Bind two vectors.
    fn bind(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
    /// Permute a vector.
    fn permute(a: &Self::Storage, shifts: usize) -> Self::Storage;
    /// Inverse a vector.
    fn inverse(a: &Self::Storage) -> Self::Storage;
    /// Calculate the cosine similarity.
    fn cosine_similarity(a: &Self::Storage, b: &Self::Storage) -> f64;
}

/// A underyling data type.
#[allow(clippy::len_without_is_empty)]
pub trait Storage: Clone + PartialEq + std::ops::Index<usize, Output = Self::Primitive> {
    /// The underyling primitive type of the storage which can be read.
    type Primitive;

    /// The length of the storage.
    fn len(&self) -> usize;

    /// Checks if two storages are compatible.
    fn enforce_constraints(&self, other: &Self);
}

impl<R: UIntResolution> Storage for bitvec::vec::BitVec<R, bitvec::order::Lsb0> {
    type Primitive = bool;

    fn len(&self) -> usize {
        bitvec::vec::BitVec::len(self)
    }

    fn enforce_constraints(&self, other: &Self) {
        debug_assert_eq!(
            self.len(),
            other.len(),
            "cannot operate on vectors with different sizes"
        );
        debug_assert!(!self.is_empty(), "cannot operate on vectors with size 0");
    }
}
impl<R: Resolution> Storage for Vec<R> {
    type Primitive = R;

    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn enforce_constraints(&self, other: &Self) {
        debug_assert_eq!(
            self.len(),
            other.len(),
            "cannot operate on vectors with different sizes"
        );
        debug_assert!(!self.is_empty(), "cannot operate on vectors with size 0");
    }
}

/// A resolution of a data type.
pub trait Resolution: Clone + Copy + PartialEq {}

/// A resolution of a data type limited to unsigned integers.
pub trait UIntResolution: Resolution + bitvec::store::BitStore {}

impl Resolution for u8 {}
impl UIntResolution for u8 {}

impl Resolution for u32 {}
impl UIntResolution for u32 {}
