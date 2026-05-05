use super::architectures::VectorSymbolicArchitecture;
use std::{
    num::NonZero,
    ops::{Add, Mul},
};

/// A trait abtracting about dynamic and fixed sizes.
pub trait Size: std::fmt::Debug + Copy + Eq {
    /// Get the size as a usize > 0.
    fn size(&self) -> usize;
}

pub trait FixedSize: Default + Size {
    const SIZE: usize;
}

/// A vector with a fixed size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fixed<const N: usize>;

impl<const N: usize> Size for Fixed<N> {
    fn size(&self) -> usize {
        N
    }
}
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

impl Size for Dynamic {
    fn size(&self) -> usize {
        self.0
    }
}

/// A vector of a vector symbolic architecture.
#[derive(Clone)]
pub struct Vector<S: Size, V: VectorSymbolicArchitecture> {
    /// The size of the vector.
    pub size: S,
    vsa: V,
    data: V::Storage,
}

impl<S: Size, V: VectorSymbolicArchitecture> Vector<S, V> {
    /// Create a random vector.
    pub fn random(vsa: &V, size: S) -> Self {
        let data = vsa.random(size.size());
        Self {
            size,
            vsa: vsa.clone(),
            data,
        }
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> std::fmt::Debug for Vector<S, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vector")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> PartialEq for Vector<S, V> {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size && self.data == other.data
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> Eq for Vector<S, V> where V::Storage: Eq {}

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
