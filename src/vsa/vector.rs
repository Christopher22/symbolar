use std::{
    num::NonZero,
    ops::{Add, Mul},
};

use crate::{
    EvaluateOps,
    architectures::{
        NonSelfInverseVectorSymbolicArchitecture, PrimaryStorage, Storage,
        VectorSymbolicArchitecture,
    },
};

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

/// A trait indicating whether a vector is normalized or not.
pub trait VectorType<V: VectorSymbolicArchitecture>:
    std::fmt::Debug
    + Clone
    + PartialEq
    + std::ops::Index<usize, Output = Self::Primitive>
    + Into<NotNormalized<V>>
{
    /// The underlying primitive type of the vector which can be read.
    type Primitive: Copy + PartialEq + PartialOrd;

    /// Create a vector type from the architecture and data.
    fn from(vsa: &V, data: V::Accumulator) -> Self;
}

#[derive(Clone)]
/// A normalized vector payload.
pub struct Normalized<V: VectorSymbolicArchitecture>(pub(crate) V::Storage);

impl<V: VectorSymbolicArchitecture> VectorType<V> for Normalized<V> {
    type Primitive = <V::Storage as Storage>::Primitive;

    fn from(vsa: &V, data: V::Accumulator) -> Self {
        Self(vsa.normalize(data))
    }
}

impl<V> From<Normalized<V>> for NotNormalized<V>
where
    V: VectorSymbolicArchitecture,
{
    fn from(value: Normalized<V>) -> Self {
        Self(V::denormalize(value.0))
    }
}

impl<V: VectorSymbolicArchitecture> PartialEq for Normalized<V> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<V: VectorSymbolicArchitecture> Eq for Normalized<V> where V::Storage: Eq {}

impl<V: VectorSymbolicArchitecture> std::fmt::Debug for Normalized<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Normalized").finish()
    }
}

impl<V: VectorSymbolicArchitecture> std::ops::Index<usize> for Normalized<V> {
    type Output = <V::Storage as Storage>::Primitive;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

#[derive(Clone)]
/// An unnormalized vector payload.
pub struct NotNormalized<V: VectorSymbolicArchitecture>(pub(crate) V::Accumulator);
impl<V: VectorSymbolicArchitecture> VectorType<V> for NotNormalized<V> {
    type Primitive = <V::Accumulator as Storage>::Primitive;

    fn from(_vsa: &V, data: V::Accumulator) -> Self {
        Self(data)
    }
}

impl<V: VectorSymbolicArchitecture> PartialEq for NotNormalized<V> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<V: VectorSymbolicArchitecture> Eq for NotNormalized<V> where V::Accumulator: Eq {}
impl<V: VectorSymbolicArchitecture> std::fmt::Debug for NotNormalized<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("NotNormalized").finish()
    }
}
impl<V: VectorSymbolicArchitecture> std::ops::Index<usize> for NotNormalized<V> {
    type Output = <V::Accumulator as Storage>::Primitive;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

#[derive(Clone)]
/// A typed vector with architecture, size, and normalization state.
pub struct Vector<S: Size, V: VectorSymbolicArchitecture, T: VectorType<V>> {
    /// The size of the vector.
    pub size: S,
    pub(crate) vsa: V,
    pub(crate) data: T,
}

impl<S: Size, V: VectorSymbolicArchitecture, T: VectorType<V>> Vector<S, V, T> {
    #[cfg(feature = "polars")]
    pub(crate) fn new(vsa: V, size: S, data: V::Accumulator) -> Self {
        assert!(
            V::valid_size(size),
            "invalid vector size for the architecture"
        );

        let data = T::from(&vsa, data);
        Self { size, vsa, data }
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> Vector<S, V, Normalized<V>> {
    /// Create a random vector.
    pub fn random(vsa: &V, size: S) -> Option<Self> {
        if !V::valid_size(size) {
            return None;
        }
        let data = vsa.random(size.size());
        Some(Self {
            size,
            vsa: vsa.clone(),
            data: Normalized(data),
        })
    }

    /// Permute the vector.
    pub fn permute(self, shifts: usize) -> Self {
        let mut data = self.data.0;
        V::permute(&mut data, shifts);
        Self {
            size: self.size,
            vsa: self.vsa,
            data: Normalized(data),
        }
    }

    /// Compute the similarity between two vectors.
    pub fn similarity(&self, other: &Self) -> f64 {
        assert_eq!(
            self.size, other.size,
            "cannot compute similarity of vectors of different sizes"
        );
        V::similarity(&self.data.0, &other.data.0)
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> Vector<S, V, NotNormalized<V>> {
    /// Normalize the vector.
    pub fn normalize(self) -> Vector<S, V, Normalized<V>> {
        let normalized = self.vsa.normalize(self.data.0);
        Vector {
            size: self.size,
            vsa: self.vsa,
            data: Normalized(normalized),
        }
    }

    /// Bind the accumulator with a normalized vector.
    /// You should prefer the Mul operator instead.
    pub fn bind(self, other: &Vector<S, V, Normalized<V>>) -> Self {
        assert_eq!(
            self.size, other.size,
            "cannot bind vectors of different sizes"
        );

        let mut data = self.data.0;
        V::bind_with_accumulator(&mut data, &other.data.0);

        Vector {
            size: self.size,
            vsa: self.vsa,
            data: NotNormalized(data),
        }
    }
}

impl<V: VectorSymbolicArchitecture> Vector<Dynamic, V, Normalized<V>> {
    /// Try to parse a vector from primitives.
    pub fn parse(vsa: V, data: &[f64]) -> Option<Self> {
        let size = Dynamic(data.len());
        let data = V::Storage::parse(data);
        if !V::valid_size(size) {
            return None;
        }
        Some(Self {
            size,
            vsa,
            data: Normalized(data),
        })
    }
}

impl<const N: usize, V: VectorSymbolicArchitecture> Vector<Fixed<N>, V, Normalized<V>> {
    /// Try to parse a vector from primitives.
    pub fn parse(vsa: V, data: &[f64]) -> Option<Self> {
        let size = Fixed::<N>;
        if data.len() != N || !V::valid_size(size) {
            return None;
        }
        let data = V::Storage::parse(data);
        Some(Self {
            size,
            vsa,
            data: Normalized(data),
        })
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> EvaluateOps for Vector<S, V, Normalized<V>> {
    fn add_many<'a, I>(mut values: I) -> Self
    where
        I: ExactSizeIterator<Item = &'a Self>,
        Self: 'a,
    {
        let first = values.next().expect("plus has at least one term");
        if values.len() == 0 {
            return first.clone();
        }

        let mut result = V::denormalize(first.data.0.clone());
        for value in values {
            first.vsa.bundle(&mut result, &value.data.0);
        }

        Self {
            size: first.size,
            vsa: first.vsa.clone(),
            data: Normalized(first.vsa.normalize(result)),
        }
    }

    fn multiply(lhs: &Self, rhs: &Self) -> Self {
        lhs * rhs
    }
}

impl<S: Size, V: VectorSymbolicArchitecture, T: VectorType<V>> std::fmt::Debug for Vector<S, V, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vector")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl<S: Size, V: VectorSymbolicArchitecture, T: VectorType<V>> PartialEq for Vector<S, V, T> {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size && self.data == other.data
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> Eq for Vector<S, V, Normalized<V>> where V::Storage: Eq {}
impl<S: Size, V: VectorSymbolicArchitecture> Eq for Vector<S, V, NotNormalized<V>> where
    V::Accumulator: Eq
{
}

impl<S: Size, V: VectorSymbolicArchitecture> Add<Self> for Vector<S, V, Normalized<V>> {
    type Output = Vector<S, V, NotNormalized<V>>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bundle vectors of different sizes"
        );

        let mut data = V::denormalize(self.data.0);
        self.vsa.bundle(&mut data, &rhs.data.0);
        Vector {
            size: self.size,
            vsa: self.vsa,
            data: NotNormalized(data),
        }
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Add<&'a Self> for Vector<S, V, Normalized<V>> {
    type Output = Vector<S, V, NotNormalized<V>>;

    fn add(self, rhs: &'a Self) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bundle vectors of different sizes"
        );
        let mut data = V::denormalize(self.data.0);
        self.vsa.bundle(&mut data, &rhs.data.0);
        Vector {
            size: self.size,
            vsa: self.vsa,
            data: NotNormalized(data),
        }
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Add<&'a Vector<S, V, Normalized<V>>>
    for &Vector<S, V, Normalized<V>>
{
    type Output = Vector<S, V, NotNormalized<V>>;

    fn add(self, rhs: &'a Vector<S, V, Normalized<V>>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bundle vectors of different sizes"
        );
        let mut data = V::denormalize(self.data.0.clone());
        self.vsa.bundle(&mut data, &rhs.data.0);
        Vector {
            size: self.size,
            vsa: self.vsa.clone(),
            data: NotNormalized(data),
        }
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> Add<Vector<S, V, NotNormalized<V>>>
    for Vector<S, V, Normalized<V>>
{
    type Output = Vector<S, V, NotNormalized<V>>;

    fn add(self, rhs: Vector<S, V, NotNormalized<V>>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bundle vectors of different sizes"
        );
        let mut data = rhs.data.0;
        self.vsa.bundle(&mut data, &self.data.0);
        Vector {
            size: self.size,
            vsa: self.vsa,
            data: NotNormalized(data),
        }
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Add<&'a Vector<S, V, NotNormalized<V>>>
    for Vector<S, V, Normalized<V>>
{
    type Output = Vector<S, V, NotNormalized<V>>;

    fn add(self, rhs: &'a Vector<S, V, NotNormalized<V>>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bundle vectors of different sizes"
        );
        let mut data = rhs.data.0.clone();
        self.vsa.bundle(&mut data, &self.data.0);

        Vector {
            size: self.size,
            vsa: self.vsa,
            data: NotNormalized(data),
        }
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Add<&'a Vector<S, V, NotNormalized<V>>>
    for &Vector<S, V, Normalized<V>>
{
    type Output = Vector<S, V, NotNormalized<V>>;

    fn add(self, rhs: &'a Vector<S, V, NotNormalized<V>>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bundle vectors of different sizes"
        );
        let mut data = rhs.data.0.clone();
        self.vsa.bundle(&mut data, &self.data.0);
        Vector {
            size: self.size,
            vsa: self.vsa.clone(),
            data: NotNormalized(data),
        }
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> Add<Vector<S, V, Normalized<V>>>
    for Vector<S, V, NotNormalized<V>>
{
    type Output = Vector<S, V, NotNormalized<V>>;

    fn add(self, rhs: Vector<S, V, Normalized<V>>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bundle vectors of different sizes"
        );
        let mut data = self.data.0;
        self.vsa.bundle(&mut data, &rhs.data.0);
        Vector {
            size: self.size,
            vsa: self.vsa,
            data: NotNormalized(data),
        }
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Add<&'a Vector<S, V, Normalized<V>>>
    for Vector<S, V, NotNormalized<V>>
{
    type Output = Vector<S, V, NotNormalized<V>>;

    fn add(self, rhs: &'a Vector<S, V, Normalized<V>>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bundle vectors of different sizes"
        );
        let mut data = self.data.0;
        self.vsa.bundle(&mut data, &rhs.data.0);
        Vector {
            size: self.size,
            vsa: self.vsa,
            data: NotNormalized(data),
        }
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Add<&'a Vector<S, V, Normalized<V>>>
    for &Vector<S, V, NotNormalized<V>>
{
    type Output = Vector<S, V, NotNormalized<V>>;

    fn add(self, rhs: &'a Vector<S, V, Normalized<V>>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bundle vectors of different sizes"
        );
        let mut data = self.data.0.clone();
        self.vsa.bundle(&mut data, &rhs.data.0);
        Vector {
            size: self.size,
            vsa: self.vsa.clone(),
            data: NotNormalized(data),
        }
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> Mul<Self> for Vector<S, V, Normalized<V>> {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bind vectors of different sizes"
        );
        V::bind(&mut self.data.0, &rhs.data.0);
        self
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Mul<&'a Self> for Vector<S, V, Normalized<V>> {
    type Output = Self;

    fn mul(mut self, rhs: &'a Self) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bind vectors of different sizes"
        );
        V::bind(&mut self.data.0, &rhs.data.0);
        self
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Mul<&'a Vector<S, V, Normalized<V>>>
    for &Vector<S, V, Normalized<V>>
{
    type Output = Vector<S, V, Normalized<V>>;

    fn mul(self, rhs: &'a Vector<S, V, Normalized<V>>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bind vectors of different sizes"
        );
        let mut data = self.data.0.clone();
        V::bind(&mut data, &rhs.data.0);
        Vector {
            size: self.size,
            vsa: self.vsa.clone(),
            data: Normalized(data),
        }
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> Mul<Vector<S, V, NotNormalized<V>>>
    for Vector<S, V, Normalized<V>>
{
    type Output = Vector<S, V, NotNormalized<V>>;

    fn mul(self, rhs: Vector<S, V, NotNormalized<V>>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bind vectors of different sizes"
        );
        rhs.bind(&self)
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Mul<&'a Vector<S, V, NotNormalized<V>>>
    for Vector<S, V, Normalized<V>>
{
    type Output = Vector<S, V, NotNormalized<V>>;

    fn mul(self, rhs: &'a Vector<S, V, NotNormalized<V>>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bind vectors of different sizes"
        );
        rhs.clone().bind(&self)
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Mul<&'a Vector<S, V, NotNormalized<V>>>
    for &Vector<S, V, Normalized<V>>
{
    type Output = Vector<S, V, NotNormalized<V>>;

    fn mul(self, rhs: &'a Vector<S, V, NotNormalized<V>>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bind vectors of different sizes"
        );
        rhs.clone().bind(self)
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> Mul<Vector<S, V, Normalized<V>>>
    for Vector<S, V, NotNormalized<V>>
{
    type Output = Vector<S, V, NotNormalized<V>>;

    fn mul(self, rhs: Vector<S, V, Normalized<V>>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bind vectors of different sizes"
        );
        self.bind(&rhs)
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Mul<&'a Vector<S, V, Normalized<V>>>
    for Vector<S, V, NotNormalized<V>>
{
    type Output = Vector<S, V, NotNormalized<V>>;

    fn mul(self, rhs: &'a Vector<S, V, Normalized<V>>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bind vectors of different sizes"
        );
        self.bind(rhs)
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Mul<&'a Vector<S, V, Normalized<V>>>
    for &Vector<S, V, NotNormalized<V>>
{
    type Output = Vector<S, V, NotNormalized<V>>;

    fn mul(self, rhs: &'a Vector<S, V, Normalized<V>>) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "cannot bind vectors of different sizes"
        );
        self.clone().bind(rhs)
    }
}

impl<S: Size, V: NonSelfInverseVectorSymbolicArchitecture> std::ops::Neg
    for Vector<S, V, Normalized<V>>
{
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        V::inverse(&mut self.data.0);
        self
    }
}

impl<S: Size, V: NonSelfInverseVectorSymbolicArchitecture> std::ops::Neg
    for &Vector<S, V, Normalized<V>>
{
    type Output = Vector<S, V, Normalized<V>>;

    fn neg(self) -> Self::Output {
        let mut data = self.data.0.clone();
        V::inverse(&mut data);
        Vector {
            size: self.size,
            vsa: self.vsa.clone(),
            data: Normalized(data),
        }
    }
}

#[derive(Debug)]
pub struct VectorIterator<'a, S: Size, V: VectorSymbolicArchitecture, T: VectorType<V>> {
    vector: &'a Vector<S, V, T>,
    index: usize,
}

impl<'a, S: Size, V: VectorSymbolicArchitecture, T: VectorType<V>> Iterator
    for VectorIterator<'a, S, V, T>
{
    type Item = T::Primitive;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.vector.size.size() {
            return None;
        }
        let value = self.vector.data[self.index];
        self.index += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vector.size.size() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture, T: VectorType<V>> ExactSizeIterator
    for VectorIterator<'a, S, V, T>
{
}

impl<'a, S: Size, V: VectorSymbolicArchitecture, T: VectorType<V>> IntoIterator
    for &'a Vector<S, V, T>
{
    type Item = T::Primitive;
    type IntoIter = VectorIterator<'a, S, V, T>;

    fn into_iter(self) -> Self::IntoIter {
        VectorIterator {
            vector: self,
            index: 0,
        }
    }
}
