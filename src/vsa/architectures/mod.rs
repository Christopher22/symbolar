//! Implementation of different architectures.

mod bsc;
mod map;

use bitvec::vec::BitVec;
use rand::RngExt;

pub use self::bsc::BinarySpatterCode;
pub use self::map::{MultiplyAddPermute, PlusMinusOnes};

/// A vector symbolic architecture.
pub trait VectorSymbolicArchitecture: Clone {
    /// The underlying storage of a single vector.
    type Storage: PrimaryStorage;
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
    /// Calculate a appopiate similarity for the architecture.
    fn similarity(a: &Self::Storage, b: &Self::Storage) -> f64;
}

/// A vector symbolic architecture where the bind operation is self-inverse.
pub trait SelfInverseVectorSymbolicArchitecture: VectorSymbolicArchitecture {}

/// A underyling data type.
#[allow(clippy::len_without_is_empty)]
pub trait Storage:
    std::fmt::Debug + Clone + PartialEq + std::ops::Index<usize, Output = Self::Primitive>
{
    /// The underyling primitive type of the storage which can be read.
    type Primitive;

    /// The length of the storage.
    fn len(&self) -> usize;

    /// Checks if two storages are compatible.
    fn enforce_constraints(&self, other: &Self);
}

/// A storage type used for "standard" vector content.
pub trait PrimaryStorage: Storage {
    /// Create random data with the specific size.
    fn random<Rng: rand::Rng>(rng: &mut Rng, size: usize) -> Self;

    /// Parse from numeric data.
    fn parse(s: &[f64]) -> Self;
}

impl<R: UIntResolution> Storage for BitVec<R, bitvec::order::Lsb0> {
    type Primitive = bool;

    fn len(&self) -> usize {
        BitVec::len(self)
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

impl<R: UIntResolution> PrimaryStorage for BitVec<R, bitvec::order::Lsb0> {
    fn random<Rng: rand::Rng>(rng: &mut Rng, size: usize) -> Self {
        let mut out = BitVec::with_capacity(size);
        let words = size / 64;

        for _ in 0..words {
            let word: u64 = rng.random();
            for bit in 0..64 {
                out.push(((word >> bit) & 1) == 1);
            }
        }

        let remaining = size % 64;
        if remaining > 0 {
            let word: u64 = rng.random();
            for bit in 0..remaining {
                out.push(((word >> bit) & 1) == 1);
            }
        }

        out
    }

    fn parse(s: &[f64]) -> Self {
        s.iter().map(|v| *v >= 0.5).collect()
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
pub trait Resolution: std::fmt::Debug + Clone + Copy + PartialEq {
    /// The identity element for the resolution.
    const IDENTITY: Self;
}

/// A resolution of a data type limited to (positive and negative) integers.
pub trait IntResolution:
    Resolution + Ord + Eq + std::ops::Neg<Output = Self> + std::ops::Add<Output = Self>
{
}

/// A resolution of a data type limited to unsigned integers.
pub trait UIntResolution: Resolution + bitvec::store::BitStore + Ord + Eq {}

impl Resolution for u8 {
    const IDENTITY: Self = 1;
}
impl UIntResolution for u8 {}

impl Resolution for i8 {
    const IDENTITY: Self = 1;
}
impl IntResolution for i8 {}

impl Resolution for u16 {
    const IDENTITY: Self = 1;
}
impl UIntResolution for u16 {}

impl Resolution for i16 {
    const IDENTITY: Self = 1;
}
impl IntResolution for i16 {}

impl Resolution for u32 {
    const IDENTITY: Self = 1;
}
impl UIntResolution for u32 {}

impl Resolution for i32 {
    const IDENTITY: Self = 1;
}
impl IntResolution for i32 {}

impl Resolution for u64 {
    const IDENTITY: Self = 1;
}
impl UIntResolution for u64 {}

impl Resolution for i64 {
    const IDENTITY: Self = 1;
}
impl IntResolution for i64 {}

impl Resolution for usize {
    const IDENTITY: Self = 1;
}
impl UIntResolution for usize {}

impl Resolution for isize {
    const IDENTITY: Self = 1;
}
impl IntResolution for isize {}
