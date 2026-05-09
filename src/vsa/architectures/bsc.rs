use std::{borrow::Borrow, sync::Arc};

use bitvec::prelude::*;
use rand::SeedableRng;

use super::{
    PrimaryStorage, SelfInverseVectorSymbolicArchitecture, Storage, UIntResolution,
    VectorSymbolicArchitecture,
};

/// Architecture based upon binary spatter codes.
#[derive(Debug)]
pub struct BinarySpatterCode<R: UIntResolution = u8, Rng: rand::Rng = rand::rngs::StdRng> {
    resolution: std::marker::PhantomData<fn() -> R>,
    rng: Arc<parking_lot::RwLock<Rng>>,
}

impl<R: UIntResolution, Rng: rand::Rng> Clone for BinarySpatterCode<R, Rng> {
    fn clone(&self) -> Self {
        Self {
            resolution: std::marker::PhantomData,
            rng: self.rng.clone(),
        }
    }
}

impl<R: UIntResolution, Rng: rand::Rng + SeedableRng> BinarySpatterCode<R, Rng> {
    /// Create a new architecture with a seed.
    pub fn new(seed: u64) -> Self {
        Self {
            resolution: std::marker::PhantomData,
            rng: Arc::new(parking_lot::RwLock::new(Rng::seed_from_u64(seed))),
        }
    }
}

impl<R: UIntResolution, Rng: rand::Rng + SeedableRng> Default for BinarySpatterCode<R, Rng> {
    fn default() -> Self {
        Self::new(rand::random())
    }
}

impl<R, Rng> VectorSymbolicArchitecture for BinarySpatterCode<R, Rng>
where
    R: UIntResolution,
    Rng: rand::Rng,
{
    type Storage = BitVec<R, Lsb0>;
    type StorageMulti = BitVec<R, Lsb0>;

    fn random(&self, size: usize) -> Self::Storage {
        BitVec::random(&mut self.rng.write(), size)
    }

    fn normalize(&self, storage: Self::StorageMulti) -> Self::Storage {
        storage
    }

    fn bundle(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        self.bundle_multi([a, b].into_iter()).expect("two vectors")
    }

    fn bind(a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        a.enforce_constraints(b);

        let mut out = a.clone();
        out ^= b.as_bitslice();
        out
    }

    fn bundle_multi<I>(&self, mut vectors: impl Iterator<Item = I>) -> Option<Self::StorageMulti>
    where
        I: Borrow<Self::Storage>,
    {
        let first_borrowed = vectors.next()?;
        let first = first_borrowed.borrow();
        let len = first.len();
        let mut ones = vec![0; len];
        let mut total = 1usize;

        for (idx, bit) in first.iter().by_vals().enumerate() {
            ones[idx] += usize::from(bit);
        }

        for vector_borrowed in vectors {
            let vector = vector_borrowed.borrow();
            first.enforce_constraints(vector);
            total += 1;
            for (idx, bit) in vector.iter().by_vals().enumerate() {
                ones[idx] += usize::from(bit);
            }
        }

        if total < 2 {
            return None;
        }

        let half = total / 2;
        let tiebreaker = BitVec::<R, Lsb0>::random(&mut self.rng.write(), len);
        let mut out = BitVec::with_capacity(len);
        for (idx, count) in ones.into_iter().enumerate() {
            if count > half {
                out.push(true);
            } else if count < half || total % 2 == 1 {
                out.push(false);
            } else {
                out.push(tiebreaker[idx]);
            }
        }

        Some(out)
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

    fn similarity(a: &Self::Storage, b: &Self::Storage) -> f64 {
        a.enforce_constraints(b);

        let dim = a.len() as f64;
        let mut diff = a.clone();
        diff ^= b.as_bitslice();
        let mismatches = diff.count_ones() as f64;
        let dot = dim - 2.0 * mismatches;
        dot / dim
    }
}

impl<R, Rng> SelfInverseVectorSymbolicArchitecture for BinarySpatterCode<R, Rng>
where
    R: UIntResolution + From<u8> + PartialEq + Copy,
    Rng: rand::Rng,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsa::architectures::VectorSymbolicArchitecture;

    #[test]
    fn random_returns_expected_size() {
        let bsc = BinarySpatterCode::<u8>::new(7);
        let hv = bsc.random(256);
        assert_eq!(hv.len(), 256);
    }

    #[test]
    fn bind_behaves_like_xor() {
        let a: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 0, 1, 0];
        let b: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 1, 0, 0];
        let bound = BinarySpatterCode::<u8>::bind(&a, &b);

        assert_eq!(bound, bitvec![u8, Lsb0; 0, 1, 1, 0]);
    }

    #[test]
    fn bundle_uses_majority_and_random_tiebreaks() {
        let bsc = BinarySpatterCode::<u8>::new(11);
        let random_source = BinarySpatterCode::<u8>::new(11);
        let a: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 0, 1, 0, 1, 0];
        let b: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 1, 0, 0, 0, 0];

        let bundled = bsc.bundle(&a, &b);
        let random_bits = random_source.random(6);
        let expected: BitVec<u8, Lsb0> = a
            .iter()
            .zip(b.iter())
            .zip(random_bits.iter())
            .map(
                |((left, right), random)| {
                    if *left == *right { *left } else { *random }
                },
            )
            .collect();

        assert_eq!(bundled, expected);
    }

    #[test]
    fn permutation_rolls_right() {
        let a: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 0, 1, 0, 1];
        let permuted = BinarySpatterCode::<u8>::permute(&a, 2);

        assert_eq!(permuted, bitvec![u8, Lsb0; 0, 1, 1, 0, 1]);
    }

    #[test]
    fn inverse_returns_copy() {
        let a: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 0, 1];
        let inv = BinarySpatterCode::<u8>::inverse(&a);
        assert_eq!(inv, a);
    }

    #[test]
    fn cosine_similarity_matches_expected_values() {
        let a: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 1, 0, 0];
        let b_same: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 1, 0, 0];
        let b_opposite: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 0, 0, 1, 1];
        let b_half: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 0, 0, 1];

        assert!((BinarySpatterCode::<u8>::similarity(&a, &b_same) - 1.0).abs() < 1e-12);
        assert!((BinarySpatterCode::<u8>::similarity(&a, &b_opposite) + 1.0).abs() < 1e-12);
        assert!(BinarySpatterCode::<u8>::similarity(&a, &b_half).abs() < 1e-12);
    }
}
