use std::{
    num::NonZero,
    ops::{Add, Mul},
};

use crate::{EvaluateOps, architectures::VectorSymbolicArchitecture};

/// A trait abtracting about dynamic and fixed sizes.
pub trait Size: std::fmt::Debug + Copy + Eq {
    /// Get the size as a usize > 0.
    fn size(&self) -> usize;
}

/// A trait for fixed sizes.
pub trait FixedSize: Default + Size {
    /// The size as a compile-time constant.
    const SIZE: usize;
}

/// A vector with a fixed size.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Fixed<const N: usize>;

impl<const N: usize> std::fmt::Debug for Fixed<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", N)
    }
}

impl<const N: usize> std::fmt::Display for Fixed<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", N)
    }
}

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

    /// Permute the vector.
    pub fn permute(self, shifts: usize) -> Self {
        let data = V::permute(&self.data, shifts);
        Self {
            size: self.size,
            vsa: self.vsa,
            data,
        }
    }

    /// Compute the similarity between two vectors.
    pub fn similarity(&self, other: &Self) -> f64 {
        assert_eq!(
            self.size, other.size,
            "cannot compute similarity of vectors of different sizes"
        );
        V::similarity(&self.data, &other.data)
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> EvaluateOps for Vector<S, V> {
    fn add_many<'a, I>(mut values: I) -> Self
    where
        I: ExactSizeIterator<Item = &'a Self>,
        Self: 'a,
    {
        let first = values.next().expect("plus has at least one term");
        if values.len() == 0 {
            return first.clone();
        }

        let storage = first
            .vsa
            .bundle_multi(std::iter::once(&first.data).chain(values.map(|value| &value.data)))
            .expect("plus has at least two compatible terms");

        Self {
            size: first.size,
            vsa: first.vsa.clone(),
            data: first.vsa.normalize(storage),
        }
    }

    fn multiply(lhs: &Self, rhs: &Self) -> Self {
        lhs * rhs
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

impl<S: Size, V: VectorSymbolicArchitecture> Add<Self> for Vector<S, V> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bundle vectors of different sizes"
        );
        let data = self.vsa.bundle(&self.data, &rhs.data);
        Self {
            size: self.size,
            vsa: self.vsa,
            data,
        }
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Add<&'a Self> for Vector<S, V> {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bundle vectors of different sizes"
        );
        let data = self.vsa.bundle(&self.data, &rhs.data);
        Self {
            size: self.size,
            vsa: self.vsa,
            data,
        }
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Add<&'a Vector<S, V>> for &Vector<S, V> {
    type Output = Vector<S, V>;

    fn add(self, rhs: &'a Vector<S, V>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bundle vectors of different sizes"
        );
        let data = self.vsa.bundle(&self.data, &rhs.data);
        Vector {
            size: self.size,
            vsa: self.vsa.clone(),
            data,
        }
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> Mul<Self> for Vector<S, V> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bind vectors of different sizes"
        );
        let data = V::bind(&self.data, &rhs.data);
        Self {
            size: self.size,
            vsa: self.vsa,
            data,
        }
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Mul<&'a Self> for Vector<S, V> {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bind vectors of different sizes"
        );
        let data = V::bind(&self.data, &rhs.data);
        Self {
            size: self.size,
            vsa: self.vsa,
            data,
        }
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Mul<&'a Vector<S, V>> for &Vector<S, V> {
    type Output = Vector<S, V>;

    fn mul(self, rhs: &'a Vector<S, V>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bind vectors of different sizes"
        );
        let data = V::bind(&self.data, &rhs.data);
        Vector {
            size: self.size,
            vsa: self.vsa.clone(),
            data,
        }
    }
}
