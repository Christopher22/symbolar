use std::sync::Arc;

use bitvec::prelude::*;
use rand::SeedableRng;

use super::{
    IntResolution, PrimaryStorage, SelfInverseVectorSymbolicArchitecture, Storage, UIntResolution,
    VectorSymbolicArchitecture,
};

/// Vector storage where each element is either +1 or -1, represented as a bit vector.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlusMinusOnes<R: UIntResolution = usize>(BitVec<R, Lsb0>);

impl<R: UIntResolution> PlusMinusOnes<R> {
    const POSITIVE: i8 = 1;
    const NEGATIVE: i8 = -1;
}

impl<R: UIntResolution> Storage for PlusMinusOnes<R> {
    type Primitive = i8;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn enforce_constraints(&self, other: &Self) {
        debug_assert_eq!(
            self.0.len(),
            other.0.len(),
            "cannot operate on vectors with different sizes"
        );
        debug_assert!(!self.0.is_empty(), "cannot operate on vectors with size 0");
    }
}

impl<R: UIntResolution> PrimaryStorage for PlusMinusOnes<R> {
    fn random<Rng: rand::Rng>(rng: &mut Rng, size: usize) -> Self {
        Self(BitVec::<R, Lsb0>::random(rng, size))
    }

    fn parse(s: &[f64]) -> Self {
        Self(s.iter().map(|v| *v >= 0.0).collect())
    }
}

impl<R: UIntResolution> std::ops::Index<usize> for PlusMinusOnes<R> {
    type Output = i8;

    fn index(&self, index: usize) -> &Self::Output {
        match self.0[index] {
            true => &Self::POSITIVE,
            false => &Self::NEGATIVE,
        }
    }
}

/// Architecture based upon multiple-add-permute.
#[derive(Debug)]
pub struct MultiplyAddPermute<
    R: UIntResolution = usize,
    RM: IntResolution = isize,
    Rng: rand::Rng = rand::rngs::StdRng,
> {
    resolution: std::marker::PhantomData<fn(RM) -> R>,
    rng: Arc<parking_lot::RwLock<Rng>>,
}

impl<R: UIntResolution, RM: IntResolution, Rng: rand::Rng> MultiplyAddPermute<R, RM, Rng> {
    fn add_bits(x: bool, y: bool) -> RM {
        (match x {
            true => RM::IDENTITY,
            false => -RM::IDENTITY,
        }) + (match y {
            true => RM::IDENTITY,
            false => -RM::IDENTITY,
        })
    }
}

impl<R: UIntResolution, RM: IntResolution, Rng: rand::Rng + SeedableRng>
    MultiplyAddPermute<R, RM, Rng>
{
    /// Create a new architecture with a seed.
    pub fn new(seed: u64) -> Self {
        Self {
            resolution: std::marker::PhantomData,
            rng: Arc::new(parking_lot::RwLock::new(Rng::seed_from_u64(seed))),
        }
    }
}

impl<R: UIntResolution, RM: IntResolution, Rng: rand::Rng> Clone
    for MultiplyAddPermute<R, RM, Rng>
{
    fn clone(&self) -> Self {
        Self {
            resolution: std::marker::PhantomData,
            rng: self.rng.clone(),
        }
    }
}

impl<R: UIntResolution, RM: IntResolution, Rng: rand::Rng + SeedableRng> Default
    for MultiplyAddPermute<R, RM, Rng>
{
    fn default() -> Self {
        Self::new(rand::random())
    }
}

impl<R: UIntResolution, RM: IntResolution, Rng: rand::Rng> VectorSymbolicArchitecture
    for MultiplyAddPermute<R, RM, Rng>
{
    type Storage = PlusMinusOnes<R>;
    type StorageMulti = Vec<RM>;

    fn normalize(&self, storage: Self::StorageMulti) -> Self::Storage {
        // MAP keeps tie votes (0) as +1 to match TorchHD's sign(0) behavior.
        let zero = -RM::IDENTITY + RM::IDENTITY;
        PlusMinusOnes(storage.into_iter().map(|v| v >= zero).collect())
    }

    fn random(&self, size: usize) -> Self::Storage {
        PlusMinusOnes::random(&mut self.rng.write(), size)
    }

    fn bundle_multi(&self, a: &Self::Storage, b: &Self::Storage) -> Self::StorageMulti {
        a.enforce_constraints(b);
        a.0.iter()
            .zip(b.0.iter())
            .map(|(x, y)| Self::add_bits(*x, *y))
            .collect()
    }

    fn bind(a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        a.enforce_constraints(b);

        let mut out = a.0.clone();
        out ^= b.0.as_bitslice();
        PlusMinusOnes(!out)
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

        let mut out = a.0.clone();
        out.rotate_right(shift);
        PlusMinusOnes(out)
    }

    fn inverse(a: &Self::Storage) -> Self::Storage {
        a.clone()
    }

    fn similarity(a: &Self::Storage, b: &Self::Storage) -> f64 {
        a.enforce_constraints(b);

        let dim = a.len() as f64;
        let mut diff = a.0.clone();
        diff ^= b.0.as_bitslice();
        let mismatches = diff.count_ones() as f64;
        let dot = dim - 2.0 * mismatches;
        dot / dim
    }
}

impl<R: UIntResolution, RM: IntResolution, Rng: rand::Rng> SelfInverseVectorSymbolicArchitecture
    for MultiplyAddPermute<R, RM, Rng>
{
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
}
