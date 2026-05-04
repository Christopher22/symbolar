use super::architectures::VectorSymbolicArchitecture;
use std::{
    num::NonZero,
    ops::{Add, Mul},
};

/// A trait abtracting about dynamic and fixed sizes.
pub trait Size: Copy + Eq {}

pub trait FixedSize: Default + Size {
    const SIZE: usize;
}

/// A vector with a fixed size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fixed<const N: usize>;

impl<const N: usize> Size for Fixed<N> {}
impl<const N: usize> FixedSize for Fixed<N> {
    const SIZE: usize = N;
}

/// A vector with a dynamic size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dynamic(usize);

impl From<NonZero<usize>> for Dynamic {
    fn from(value: NonZero<usize>) -> Self {
        Self(value.get())
    }
}

impl Size for Dynamic {}

/// A vector of a vector symbolic architecture.
#[derive(Debug, Clone)]
pub struct Vector<S: Size, V: VectorSymbolicArchitecture> {
    /// The size of the vector.
    pub size: S,
    vsa: V,
    data: V::Storage,
}

impl<V: VectorSymbolicArchitecture> Vector<Dynamic, V> {
    /// Create a random vector with fixed dimensions.
    pub fn random_fixed<const N: usize>(vsa: &V) -> Vector<Fixed<N>, V> {
        let size = Fixed::<N>;
        let data = vsa.random(N);
        Vector {
            size,
            vsa: vsa.clone(),
            data,
        }
    }

    /// Create a random vector.
    pub fn random(vsa: &V, size: usize) -> Self {
        let data = vsa.random(size);
        Self {
            size: Dynamic(size),
            vsa: vsa.clone(),
            data,
        }
    }
}

impl<'a, S: FixedSize, V: VectorSymbolicArchitecture> Add<&'a Self> for Vector<S, V> {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        let data = self.vsa.bundle(&self.data, &rhs.data);
        Self {
            size: self.size,
            vsa: self.vsa,
            data,
        }
    }
}

impl<'a, S: FixedSize, V: VectorSymbolicArchitecture> Mul<&'a Self> for Vector<S, V> {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        let data = V::bind(&self.data, &rhs.data);
        Self {
            size: self.size,
            vsa: self.vsa,
            data,
        }
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> Vector<S, V> {
    /// Permute the vector.
    pub fn permute(self, shifts: usize) -> Self {
        let data = V::permute(&self.data, shifts);
        Self {
            size: self.size,
            vsa: self.vsa,
            data,
        }
    }
}

impl<'a, V: VectorSymbolicArchitecture> Add<&'a Self> for Vector<Dynamic, V> {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        assert_eq!(self.size, rhs.size, "cannot add vectors of different sizes");
        let data = self.vsa.bundle(&self.data, &rhs.data);
        Self {
            size: self.size,
            vsa: self.vsa,
            data,
        }
    }
}

impl<'a, V: VectorSymbolicArchitecture> Mul<&'a Self> for Vector<Dynamic, V> {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot multiply vectors of different sizes"
        );
        let data = V::bind(&self.data, &rhs.data);
        Self {
            size: self.size,
            vsa: self.vsa,
            data,
        }
    }
}
