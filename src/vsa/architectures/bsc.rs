use std::sync::Arc;

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
    type Accumulator = BitVec<R, Lsb0>;

    fn random(&self, size: usize) -> Self::Storage {
        BitVec::random(&mut self.rng.write(), size)
    }

    fn denormalize(storage: Self::Storage) -> Self::Accumulator {
        storage
    }

    fn normalize(&self, storage: Self::Accumulator) -> Self::Storage {
        storage
    }

    fn bind(a: &mut Self::Storage, b: &Self::Storage) {
        a.enforce_constraints(b);
        *a ^= b.as_bitslice();
    }

    fn bundle(&self, accumulator: &mut Self::Accumulator, vector: &Self::Storage) {
        let tiebreaker = BitVec::<R, Lsb0>::random(&mut self.rng.write(), vector.len());
        for ((mut acc_bit, vec_bit), tie_bit) in accumulator
            .iter_mut()
            .zip(vector.iter())
            .zip(tiebreaker.iter())
        {
            if *acc_bit != *vec_bit {
                *acc_bit = *tie_bit;
            }
        }
    }

    fn bundle_with_accumulator(
        &self,
        accumulator: &mut Self::Accumulator,
        vector: &Self::Accumulator,
    ) {
        let tiebreaker = BitVec::<R, Lsb0>::random(&mut self.rng.write(), vector.len());
        for ((mut acc_bit, vec_bit), tie_bit) in accumulator
            .iter_mut()
            .zip(vector.iter())
            .zip(tiebreaker.iter())
        {
            if *acc_bit != *vec_bit {
                *acc_bit = *tie_bit;
            }
        }
    }

    fn bind_with_accumulator(a: &mut Self::Accumulator, b: &Self::Storage) {
        a.enforce_constraints(b);
        *a ^= b.as_bitslice();
    }

    fn permute(a: &mut Self::Storage, shifts: usize) {
        let len = a.len();
        if len == 0 {
            return;
        }

        let shift = shifts % len;
        if shift == 0 {
            return;
        }

        a.rotate_right(shift);
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
        let mut bound = a.clone();
        BinarySpatterCode::<u8>::bind(&mut bound, &b);

        assert_eq!(bound, bitvec![u8, Lsb0; 0, 1, 1, 0]);
    }

    #[test]
    fn bundle_uses_majority_and_random_tiebreaks() {
        let bsc = BinarySpatterCode::<u8>::new(11);
        let random_source = BinarySpatterCode::<u8>::new(11);
        let a: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 0, 1, 0, 1, 0];
        let b: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 1, 0, 0, 0, 0];

        let mut bundled = a.clone();
        bsc.bundle(&mut bundled, &b);
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
        let mut a: BitVec<u8, Lsb0> = bitvec![u8, Lsb0; 1, 0, 1, 0, 1];
        BinarySpatterCode::<u8>::permute(&mut a, 2);

        assert_eq!(a, bitvec![u8, Lsb0; 0, 1, 1, 0, 1]);
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
