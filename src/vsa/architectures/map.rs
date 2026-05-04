use std::sync::Arc;

use crate::vsa::architectures::Storage;

use super::{UIntResolution, VectorSymbolicArchitecture};
use bitvec::prelude::*;
use rand::{RngExt, SeedableRng};

/// Architecture based upon multiple-add-permute.
#[derive(Debug)]
pub struct MultiplyAddPermute<R: UIntResolution = u8, Rng: rand::Rng = rand::rngs::StdRng> {
    resolution: std::marker::PhantomData<fn() -> R>,
    rng: Arc<parking_lot::RwLock<Rng>>,
}

impl<R: UIntResolution, Rng: rand::Rng> Clone for MultiplyAddPermute<R, Rng> {
    fn clone(&self) -> Self {
        Self {
            resolution: std::marker::PhantomData,
            rng: self.rng.clone(),
        }
    }
}

impl<R: UIntResolution, Rng: rand::Rng + SeedableRng> MultiplyAddPermute<R, Rng> {
    /// Create a new architecture with a seed.
    pub fn new(seed: u64) -> Self {
        Self {
            resolution: std::marker::PhantomData,
            rng: Arc::new(parking_lot::RwLock::new(Rng::seed_from_u64(seed))),
        }
    }
}

impl<R: UIntResolution, Rng: rand::Rng + SeedableRng> Default for MultiplyAddPermute<R, Rng> {
    fn default() -> Self {
        Self::new(rand::random())
    }
}

impl<R, Rng> VectorSymbolicArchitecture for MultiplyAddPermute<R, Rng>
where
    R: UIntResolution + From<u8> + PartialEq + Copy,
    Rng: rand::Rng,
{
    /// We treat -1 as 0, so we can use bitvec for storage.
    type Storage = BitVec<R, Lsb0>;
    type StorageMulti = Vec<R>;

    fn normalize(&self, storage: Self::StorageMulti) -> Self::Storage {
        let positive = R::from(2);
        storage.into_iter().map(|v| v == positive).collect()
    }

    fn random(&self, size: usize) -> Self::Storage {
        let mut rng = self.rng.write();
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

    fn bundle(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        a.enforce_constraints(b);

        let mut out = a.clone();
        out &= b.as_bitslice();
        out
    }

    fn bind(a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        a.enforce_constraints(b);

        let mut out = a.clone();
        out ^= b.as_bitslice();
        !out
    }

    fn bundle_multi(&self, a: &Self::Storage, b: &Self::Storage) -> Self::StorageMulti {
        a.enforce_constraints(b);

        a.iter()
            .zip(b.iter())
            .map(|(x, y)| R::from((*x as u8) + (*y as u8)))
            .collect()
    }

    fn permute(a: &Self::Storage, shifts: usize) -> Self::Storage {
        let len = a.len();
        if len == 0 {
            return a.clone();
        }

        let shift = shifts % len;
        if shift == 0 {
            return a.clone();
        }

        let mut out = a.clone();
        out.rotate_right(shift);
        out
    }

    fn inverse(a: &Self::Storage) -> Self::Storage {
        a.clone()
    }

    fn cosine_similarity(a: &Self::Storage, b: &Self::Storage) -> f64 {
        a.enforce_constraints(b);

        let dim = a.len() as f64;
        let mut diff = a.clone();
        diff ^= b.as_bitslice();
        let mismatches = diff.count_ones() as f64;
        let dot = dim - 2.0 * mismatches;
        dot / dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsa::architectures::VectorSymbolicArchitecture;

    #[test]
    fn random_returns_expected_size() {
        let map = MultiplyAddPermute::<u8>::new(7);
        let hv = map.random(256);
        assert_eq!(hv.len(), 256);
    }

    #[test]
    fn bind_behaves_like_bipolar_multiplication() {
        let a: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 0, 1, 0];
        let b: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 1, 0, 0];
        let bound = MultiplyAddPermute::<u8>::bind(&a, &b);

        assert_eq!(bound, bitvec![u8, Lsb0; 1, 0, 0, 1]);
    }

    #[test]
    fn bundle_normalizes_with_negative_tie_break() {
        let map = MultiplyAddPermute::<u8>::new(1);
        let a: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 0, 1, 0];
        let b: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 1, 0, 0];
        let bundled = map.bundle(&a, &b);

        assert_eq!(bundled, bitvec![u8, Lsb0; 1, 0, 0, 0]);
    }

    #[test]
    fn permutation_rolls_right() {
        let a: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 0, 1, 0, 1];
        let permuted = MultiplyAddPermute::<u8>::permute(&a, 2);

        assert_eq!(permuted, bitvec![u8, Lsb0; 0, 1, 1, 0, 1]);
    }

    #[test]
    fn inverse_returns_copy() {
        let a: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 0, 1];
        let inv = MultiplyAddPermute::<u8>::inverse(&a);
        assert_eq!(inv, a);
    }

    #[test]
    fn cosine_similarity_matches_expected_values() {
        let a: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 1, 0, 0];
        let b_same: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 1, 0, 0];
        let b_opposite: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 0, 0, 1, 1];
        let b_half: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 0, 0, 1];

        assert!((MultiplyAddPermute::<u8>::cosine_similarity(&a, &b_same) - 1.0).abs() < 1e-12);
        assert!((MultiplyAddPermute::<u8>::cosine_similarity(&a, &b_opposite) + 1.0).abs() < 1e-12);
        assert!(MultiplyAddPermute::<u8>::cosine_similarity(&a, &b_half).abs() < 1e-12);
    }
}
